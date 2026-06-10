import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class D4RTFrontend:
    """Offline D4RT observation provider for SLAM BA."""

    TARGET_KEYS = ("targets", "target_2d", "tracks", "d4rt_targets")
    VALID_KEYS = ("valids", "valid", "masks", "d4rt_valids")
    DEPTH_KEYS = ("depths", "depth", "d4rt_depths")
    POSE_KEYS = ("poses", "camera_poses", "d4rt_poses")

    def __init__(self, cfg, device="cuda:0", ht=None, wd=None, down_scale=8):
        self.cfg = cfg.get("tracking", {}).get("d4rt", {})
        self.device = device
        self.ht = ht
        self.wd = wd
        self.down_scale = down_scale

        self.cache_path = self.cfg.get("cache_path", "")
        self.cache_dir = self.cfg.get("cache_dir", "")
        self.scene = cfg.get("scene", "")
        self.allow_fallback = bool(self.cfg.get("allow_reprojection_fallback", False))
        self.target_coord_scale = self.cfg.get("target_coord_scale", "auto")

        self.targets = None
        self.valids = None
        self.depths = None
        self.poses = None

        if self.cfg.get("mode", "offline") != "offline":
            raise NotImplementedError(
                "D4RT online inference is not wired yet. Use tracking.d4rt.mode=offline "
                "with a cache npz containing D4RT targets/depths."
            )

        path = self._resolve_cache_path()
        if path:
            self._load_npz(path)
        elif not self.allow_fallback:
            raise FileNotFoundError(
                "D4RT mode is active but no cache file was found. Set "
                "tracking.d4rt.cache_path or tracking.d4rt.cache_dir, or set "
                "allow_reprojection_fallback=true for plumbing tests only."
            )

    @property
    def has_depth(self):
        return self.depths is not None

    @property
    def has_pose(self):
        return self.poses is not None

    def _resolve_cache_path(self) -> str:
        if self.cache_path:
            return self.cache_path if os.path.exists(self.cache_path) else ""

        if not self.cache_dir:
            return ""

        candidates = []
        if self.scene:
            candidates.extend([
                os.path.join(self.cache_dir, f"{self.scene}.npz"),
                os.path.join(self.cache_dir, self.scene, "d4rt_cache.npz"),
            ])
        candidates.append(os.path.join(self.cache_dir, "d4rt_cache.npz"))

        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    @staticmethod
    def _first_key(data, keys):
        for key in keys:
            if key in data:
                return key
        return None

    def _load_npz(self, path: str):
        data = np.load(path)
        target_key = self._first_key(data, self.TARGET_KEYS)
        valid_key = self._first_key(data, self.VALID_KEYS)
        depth_key = self._first_key(data, self.DEPTH_KEYS)
        pose_key = self._first_key(data, self.POSE_KEYS)

        if target_key is None and not self.allow_fallback:
            raise KeyError(
                f"{path} does not contain D4RT targets. Expected one of "
                f"{self.TARGET_KEYS}."
            )

        if target_key is not None:
            self.targets = torch.from_numpy(data[target_key]).float()
        if valid_key is not None:
            self.valids = torch.from_numpy(data[valid_key]).bool()
        if depth_key is not None:
            self.depths = torch.from_numpy(data[depth_key]).float()
        if pose_key is not None:
            self.poses = torch.from_numpy(data[pose_key]).float()

    def query_edges(self, ii, jj, coords0, fallback_target=None):
        """Return D4RT 2D targets for factor edges."""

        if self.targets is None:
            if self.allow_fallback and fallback_target is not None:
                valid = torch.ones_like(fallback_target[..., 0], dtype=torch.bool)
                return fallback_target, valid
            raise RuntimeError("D4RT targets are unavailable.")

        targets = self.targets.to(self.device)
        edge_targets = targets[ii.long(), jj.long()]
        edge_targets = self._as_bhw2(edge_targets)
        source_hw = edge_targets.shape[1:3]
        edge_targets = self._resize_bhw2(edge_targets, coords0.shape[:2])
        edge_targets = self._scale_target_coords(edge_targets, source_hw, coords0.shape[:2])

        if self.valids is not None:
            valids = self.valids.to(self.device)[ii.long(), jj.long()]
            valids = self._as_bhw(valids)
            valids = self._resize_mask(valids, coords0.shape[:2])
        else:
            valids = torch.ones(
                edge_targets.shape[:-1], device=self.device, dtype=torch.bool
            )

        ht, wd = coords0.shape[:2]
        in_bounds = (
            (edge_targets[..., 0] >= 0)
            & (edge_targets[..., 0] <= wd - 1)
            & (edge_targets[..., 1] >= 0)
            & (edge_targets[..., 1] <= ht - 1)
        )
        valids = valids & in_bounds & torch.isfinite(edge_targets).all(dim=-1)

        return edge_targets.unsqueeze(0), valids.unsqueeze(0)

    def query_depth(self, frame_ids, out_hw: Tuple[int, int]):
        if self.depths is None:
            return None, None

        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long)
        depth = self.depths[frame_ids.cpu()].to(self.device).float()
        depth = self._as_bhw(depth)
        if depth.shape[-2:] != out_hw:
            depth = F.interpolate(
                depth.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=True
            ).squeeze(1)

        valid = torch.isfinite(depth) & (depth > 1e-6)
        return depth, valid

    def query_pose(self, frame_ids):
        if self.poses is None:
            return None
        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long)
        return self.poses[frame_ids.cpu()].to(self.device).float()

    @staticmethod
    def _as_bhw2(x):
        if x.ndim == 4 and x.shape[-1] == 2:
            return x.float()
        if x.ndim == 4 and x.shape[1] == 2:
            return x.permute(0, 2, 3, 1).contiguous().float()
        raise ValueError(f"Expected edge targets as [N,H,W,2] or [N,2,H,W], got {tuple(x.shape)}")

    @staticmethod
    def _as_bhw(x):
        if x.ndim == 3:
            return x
        if x.ndim == 4 and x.shape[1] == 1:
            return x[:, 0]
        if x.ndim == 4 and x.shape[-1] == 1:
            return x[..., 0]
        raise ValueError(f"Expected masks/depths as [N,H,W], got {tuple(x.shape)}")

    @staticmethod
    def _resize_bhw2(x, out_hw):
        if x.shape[1:3] == out_hw:
            return x
        x_chw = x.permute(0, 3, 1, 2).contiguous()
        x_chw = F.interpolate(x_chw, size=out_hw, mode="bilinear", align_corners=True)
        return x_chw.permute(0, 2, 3, 1).contiguous()

    def _scale_target_coords(self, target, source_hw, out_hw):
        if source_hw == out_hw:
            return target

        mode = self.target_coord_scale
        if mode == "none":
            return target
        if mode == "normalized_to_tracking":
            target = target.clone()
            target[..., 0] = (target[..., 0] + 1.0) * 0.5 * float(out_hw[1] - 1)
            target[..., 1] = (target[..., 1] + 1.0) * 0.5 * float(out_hw[0] - 1)
            return target

        should_scale = mode in ("auto", "full_to_tracking")
        if mode == "auto":
            should_scale = source_hw != out_hw

        if not should_scale:
            return target

        scale_y = float(out_hw[0]) / float(source_hw[0])
        scale_x = float(out_hw[1]) / float(source_hw[1])
        target = target.clone()
        target[..., 0] *= scale_x
        target[..., 1] *= scale_y
        return target

    @staticmethod
    def _resize_mask(mask, out_hw):
        if mask.shape[-2:] == out_hw:
            return mask
        resized = F.interpolate(mask.float().unsqueeze(1), size=out_hw, mode="nearest")
        return resized[:, 0] > 0.5
