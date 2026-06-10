import numpy as np
import torch

from src.factor_graph import FactorGraph


class D4RTFactorGraph(FactorGraph):
    """Factor graph whose reprojection targets come from D4RT tracks."""

    def __init__(self, video, d4rt_frontend, device="cuda:0", max_factors=-1):
        super().__init__(
            video,
            update_op=None,
            device=device,
            corr_impl="d4rt",
            max_factors=max_factors,
        )
        self.d4rt = d4rt_frontend
        self.d4rt_cfg = video.cfg.get("tracking", {}).get("d4rt", {})
        self.use_depth_init = bool(self.d4rt_cfg.get("use_depth_init", False))
        self.use_depth_reg = bool(self.d4rt_cfg.get("use_depth_reg", False))
        self.use_pose_init = bool(self.d4rt_cfg.get("use_pose_init", False))
        self.use_edge_selection = bool(self.d4rt_cfg.get("use_edge_selection", False))
        self.binary_weight = float(self.d4rt_cfg.get("binary_weight", 1.0))

        if self.use_depth_reg:
            video.cfg["tracking"]["uncertainty_params"]["gamma_depth"] = float(
                self.d4rt_cfg.get("depth_reg_lambda", 0.05)
            )

        if not hasattr(video, "d4rt_depth_initialized"):
            video.d4rt_depth_initialized = torch.zeros(
                video.poses.shape[0], device=device, dtype=torch.bool
            )
        if not hasattr(video, "d4rt_pose_initialized"):
            video.d4rt_pose_initialized = torch.zeros(
                video.poses.shape[0], device=device, dtype=torch.bool
            )

    def _filter_repeated_edges(self, ii, jj):
        if ii.numel() == 0:
            return ii, jj
        if self.ii.numel() == 0 and self.ii_inac.numel() == 0:
            return ii, jj

        device = ii.device
        n = max(
            self.ii.max().item() if self.ii.numel() > 0 else -1,
            self.ii_inac.max().item() if self.ii_inac.numel() > 0 else -1,
            ii.max().item(),
        ) + 1
        m = max(
            self.jj.max().item() if self.jj.numel() > 0 else -1,
            self.jj_inac.max().item() if self.jj_inac.numel() > 0 else -1,
            jj.max().item(),
        ) + 1

        used = torch.zeros((n, m), dtype=torch.bool, device=device)
        used[self.ii, self.jj] = True
        used[self.ii_inac, self.jj_inac] = True
        keep = ~used[ii, jj]
        return ii[keep], jj[keep]

    @torch.no_grad()
    def add_factors(self, ii, jj, remove=False):
        """Add D4RT reprojection edges without building DROID corr volumes."""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)
        else:
            ii = ii.to(device=self.device, dtype=torch.long)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)
        else:
            jj = jj.to(device=self.device, dtype=torch.long)

        ii, jj = self._filter_repeated_edges(ii.reshape(-1), jj.reshape(-1))
        if ii.shape[0] == 0:
            return

        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors and remove:
            ix = torch.arange(len(self.age), device=self.device)[torch.argsort(self.age)]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        with torch.amp.autocast("cuda", enabled=False):
            coords, _ = self.video.reproject(ii, jj)
            target, valid = self.d4rt.query_edges(ii, jj, self.coords0, fallback_target=coords)
            weight = self._valid_to_binary_weight(valid, target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)
        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    def _valid_to_binary_weight(self, valid, target):
        weight = torch.ones_like(target, dtype=torch.float) * self.binary_weight
        return weight * valid.unsqueeze(-1).float()

    @torch.no_grad()
    def _score_edges_by_d4rt(self, ii, jj):
        if ii.numel() == 0:
            return torch.empty(0, device=self.device)
        coords, _ = self.video.reproject(ii, jj)
        _, valid = self.d4rt.query_edges(ii, jj, self.coords0, fallback_target=coords)
        return valid.float().mean(dim=(0, 2, 3))

    @torch.no_grad()
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        if not self.use_edge_selection:
            return super().add_proximity_factors(t0, t1, rad, nms, beta, thresh, remove)

        t = self.video.counter.value
        es = []
        for i in range(t0, t):
            for j in range(max(i - rad - 1, 0), i):
                es.append((i, j))
                es.append((j, i))

        ix = torch.arange(t0, t, device=self.device)
        jx = torch.arange(t1, t, device=self.device)
        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        keep = (ii != jj) & (ii - rad >= jj)
        ii, jj = ii[keep], jj[keep]
        if ii.numel() > 0:
            scores = self._score_edges_by_d4rt(ii, jj)
            order = torch.argsort(scores, descending=True)
            min_score = float(self.d4rt_cfg.get("edge_min_valid_ratio", 0.05))
            for k in order.tolist():
                if scores[k].item() < min_score or len(es) >= self.max_factors:
                    break
                i, j = ii[k], jj[k]
                es.append((i, j))
                es.append((j, i))

        if len(es) == 0:
            return

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)

    @torch.no_grad()
    def add_backend_proximity_factors(self, t_start, t_end, nms, radius, thresh, max_factors, beta, t_start_loop=None, loop=False):
        if not self.use_edge_selection:
            return super().add_backend_proximity_factors(
                t_start, t_end, nms, radius, thresh, max_factors, beta, t_start_loop, loop
            )

        if t_start_loop is None or not loop:
            t_start_loop = t_start

        es = []
        for i in range(t_start_loop, t_end):
            for j in range(max(i - radius - 1, 0), i):
                es.append((i, j))
                es.append((j, i))

        ix = torch.arange(t_start_loop, t_end, device=self.device)
        jx = torch.arange(t_start, t_end, device=self.device)
        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        keep = (ii != jj) & (ii - radius >= jj)
        if loop:
            keep &= (ii - jj > 20)
        ii, jj = ii[keep], jj[keep]

        if ii.numel() > 0:
            scores = self._score_edges_by_d4rt(ii, jj)
            order = torch.argsort(scores, descending=True)
            min_score = float(self.d4rt_cfg.get("edge_min_valid_ratio", 0.05))
            for k in order.tolist():
                if scores[k].item() < min_score or len(es) >= max_factors:
                    break
                i, j = ii[k], jj[k]
                es.append((i, j))
                if not loop:
                    es.append((j, i))

        if len(es) < 3:
            return 0

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove=True)
        return len(self.ii)

    def _apply_d4rt_initialization(self, frame_ids):
        if frame_ids.numel() == 0:
            return

        unique_ids = torch.unique(frame_ids).to(self.device)

        if (self.use_depth_init or self.use_depth_reg) and self.d4rt.has_depth:
            pending = unique_ids[~self.video.d4rt_depth_initialized[unique_ids]]
            if pending.numel() > 0:
                depth, valid = self.d4rt.query_depth(pending, self.video.disps.shape[-2:])
                if depth is not None:
                    disp = torch.where(valid, 1.0 / depth.clamp(min=1e-6), self.video.disps[pending])
                    disp_up = disp.repeat_interleave(
                        self.video.down_scale, 1
                    ).repeat_interleave(self.video.down_scale, 2)[:, : self.video.ht, : self.video.wd]
                    with self.video.get_lock():
                        if self.use_depth_init:
                            self.video.disps[pending] = disp
                            self.video.disps_up[pending] = disp_up
                        if self.use_depth_reg:
                            self.video.mono_disps[pending] = disp
                            self.video.mono_disps_up[pending] = disp_up
                        self.video.d4rt_depth_initialized[pending] = True

        if self.use_pose_init and self.d4rt.has_pose:
            pending = unique_ids[~self.video.d4rt_pose_initialized[unique_ids]]
            if pending.numel() > 0:
                poses = self.d4rt.query_pose(pending)
                if poses is not None:
                    with self.video.get_lock():
                        self.video.poses[pending] = poses
                        self.video.d4rt_pose_initialized[pending] = True

    @torch.no_grad()
    def update(
        self,
        t0=None,
        t1=None,
        itrs=2,
        use_inactive=False,
        EP=1e-7,
        motion_only=False,
        enable_update_uncer=False,
        enable_udba=False,
        visualization_stage=False,
    ):
        """Run BA directly on D4RT reprojection targets."""

        if self.ii.numel() == 0:
            return

        with torch.amp.autocast("cuda", enabled=False):
            self._apply_d4rt_initialization(torch.cat([self.ii, self.jj], 0))
            coords, mask_geom = self.video.reproject(self.ii, self.jj)
            target, mask_d4rt = self.d4rt.query_edges(
                self.ii, self.jj, self.coords0, fallback_target=coords
            )
            valid = mask_geom & mask_d4rt
            self.target = target.to(dtype=torch.float)
            self.weight = self._valid_to_binary_weight(valid, self.target)

            if use_inactive:
                m = (self.ii_inac >= (t0 or 1) - 3) & (self.jj_inac >= (t0 or 1) - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:, m], self.target], 1)
                weight = torch.cat([self.weight_inac[:, m], self.weight], 1)
            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

            if t0 is None:
                t0 = max(1, ii.min().item() + 1)

            damping = 0.2 * self.damping[torch.unique(ii)].contiguous() + EP

            self.video.ba(
                target,
                weight,
                damping,
                ii,
                jj,
                t0,
                t1,
                iters=itrs,
                lm=1e-4,
                ep=0.1,
                lr=0.0,
                weight_decay=0.0,
                motion_only=motion_only,
                enable_update_uncer=False,
                enable_udba=False,
                visualization_stage=False,
            )

            unique_ii = torch.unique(ii)
            self.video.disps_up[unique_ii] = torch.nn.functional.interpolate(
                self.video.disps[unique_ii].unsqueeze(1),
                size=(self.video.ht, self.video.wd),
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)

        self.age += 1

    @torch.no_grad()
    def update_lowmem(
        self,
        t0=None,
        t1=None,
        itrs=2,
        use_inactive=False,
        EP=1e-7,
        steps=8,
        enable_wq=True,
        enable_update_uncer=False,
        enable_udba=False,
        visualization_stage=False,
        save_edges_weights=False,
    ):
        for _ in range(steps):
            self.update(
                t0=t0,
                t1=t1,
                itrs=itrs,
                use_inactive=use_inactive,
                EP=EP,
                motion_only=False,
                enable_update_uncer=False,
                enable_udba=False,
                visualization_stage=False,
            )

        if save_edges_weights:
            np.savez(
                "all_edges_d4rt.npz",
                weight=self.weight.cpu().numpy(),
                ii=self.ii.cpu().numpy(),
                jj=self.jj.cpu().numpy(),
            )
