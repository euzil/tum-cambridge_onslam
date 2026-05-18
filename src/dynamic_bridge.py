"""
dynamic_bridge.py
=================
DROID-W 与 dynamic_prediction 模块之间的集成桥接层。

集成方式：
    在 SLAM.terminate() 结束后调用
    DynamicBridge(cfg, save_dir).run_from_video_npz(video_npz_path)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage


class DynamicBridge:
    """
    DROID-W 动态预测集成桥接器。

    Parameters
    ----------
    cfg             : SLAM 配置字典
    save_dir        : SLAM 输出目录（包含 video.npz）
    window_size     : 滑动窗口大小（帧数）
    predict_steps   : 预测未来帧数
    uncer_thresh    : 不确定性阈值
    enable_vis      : 是否启用 Open3D 实时可视化
    fps             : 可视化帧率
    """

    def __init__(
        self,
        cfg: dict,
        save_dir: str,
        window_size: int = 8,
        predict_steps: int = 2,
        uncer_thresh: float = 0.8,
        enable_vis: bool = False,
        fps: float = 10.0,
    ) -> None:
        self.cfg = cfg
        self.save_dir = save_dir
        self.window_size = window_size
        self.predict_steps = predict_steps
        self.uncer_thresh = uncer_thresh
        self.enable_vis = enable_vis
        self.fps = fps

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run_from_video_npz(self, video_npz_path: Optional[str] = None) -> dict:
        """从保存的 video.npz 提取动态点并运行预测，然后生成 GIF。"""
        if video_npz_path is None:
            video_npz_path = os.path.join(self.save_dir, "video.npz")

        if not os.path.exists(video_npz_path):
            print(f"[DynamicBridge] video.npz 不存在：{video_npz_path}，跳过动态预测。")
            return {}

        print(f"[DynamicBridge] 从 {video_npz_path} 加载动态点 ...")

        from dynamic_prediction.d4rt_bridge import DROIDWBridge
        from dynamic_prediction.sliding_window import SlidingWindowPredictor

        bridge = DROIDWBridge(uncer_thresh=self.uncer_thresh)
        dw_data = bridge.load(video_npz_path)
        frames, pids_list = bridge.extract_dynamic_points(dw_data)

        n_frames = len(frames)
        n_dyn_total = sum(len(f) for f in frames)
        print(
            f"[DynamicBridge] 共 {n_frames} 个关键帧，"
            f"平均每帧 {n_dyn_total / max(n_frames, 1):.0f} 个动态点"
        )

        predictor = SlidingWindowPredictor(
            window_size=self.window_size,
            predict_steps=self.predict_steps,
        )

        visualizer = self._make_visualizer()
        results = self._prediction_loop(frames, pids_list, predictor, visualizer)
        if visualizer is not None:
            visualizer.destroy()

        # 自动生成预测 GIF
        self.generate_prediction_gif(video_npz_path)

        return results

    # ------------------------------------------------------------------
    # GIF 生成（核心新功能）
    # ------------------------------------------------------------------

    def generate_prediction_gif(
        self,
        video_npz_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        gif_name: str = "dynamic_prediction.gif",
        duration_ms: int = 150,
        max_history: int = 6,
        dot_radius: float = 3.0,
    ) -> str:
        """
        把动态预测结果叠加到原始图像上并生成 GIF。

        每帧图像上画出：
          🔴 红色实心点  —— 当前帧检测到的动态像素（投影到 2D）
          🔵 蓝色折线    —— 过去轨迹（投影到 2D）
          🟡 黄色空心圆  —— 预测的未来落点（第 1 步，投影到 2D）
        """
        if video_npz_path is None:
            video_npz_path = os.path.join(self.save_dir, "video.npz")
        if not os.path.exists(video_npz_path):
            print(f"[DynamicBridge] video.npz 不存在，跳过 GIF 生成。")
            return ""

        if output_dir is None:
            output_dir = os.path.join(self.save_dir, "dynamic_pred_frames")
        os.makedirs(output_dir, exist_ok=True)

        print(f"[DynamicBridge] Rendering prediction GIF frames ...")

        # ---- 加载原始数据 ----
        from dynamic_prediction.d4rt_bridge import DROIDWBridge
        from dynamic_prediction.sliding_window import SlidingWindowPredictor

        npz        = np.load(video_npz_path, allow_pickle=True)
        images     = np.asarray(npz["images"])      # [T, 3, H, W] float32 0~1
        poses      = np.asarray(npz["poses"])       # [T, 4, 4] world-to-camera
        intrinsics = np.asarray(npz["intrinsics"])  # [T, 4] 低分辨率内参

        T, _, H_img, W_img = images.shape
        # 内参是在 1/8 分辨率下标定的，缩放到原图分辨率
        scale_x = W_img / (W_img // 8)  # = 8.0
        scale_y = H_img / (H_img // 8)  # = 8.0

        bridge = DROIDWBridge(uncer_thresh=self.uncer_thresh)
        dw_data = bridge.load(video_npz_path)
        frames, pids_list = bridge.extract_dynamic_points(dw_data)

        predictor = SlidingWindowPredictor(
            window_size=self.window_size,
            predict_steps=self.predict_steps,
        )

        # 预先计算关键帧：动态点最多的5帧 + 第一个PRED帧
        frame_sizes = [len(f) for f in frames]
        top5 = sorted(range(len(frame_sizes)), key=lambda i: frame_sizes[i], reverse=True)[:5]
        first_pred = self.window_size - 1  # 第一个进入PRED状态的帧索引
        keyframes_to_save = set(top5) | {first_pred}

        frame_paths = []
        dpi = 100

        def world_to_pixel(pts_w, pose_cw, fx, fy, cx, cy):
            """世界坐标 [N,3] → 像素坐标 (u[N], v[N], valid[N])"""
            pts_w = np.asarray(pts_w)
            if pts_w.ndim == 1:
                pts_w = pts_w[None, :]
            if len(pts_w) == 0:
                return np.array([]), np.array([]), np.array([], dtype=bool)
            R = pose_cw[:3, :3]
            t = pose_cw[:3, 3]
            pts_c = (R @ pts_w.T).T + t[None, :]       # [N, 3] 相机坐标
            z = pts_c[:, 2]
            valid = z > 0.01
            u = np.where(valid, fx * pts_c[:, 0] / (z + 1e-8) + cx, -1.0)
            v = np.where(valid, fy * pts_c[:, 1] / (z + 1e-8) + cy, -1.0)
            valid &= (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
            return u, v, valid

        for t, (pts_world, pids) in enumerate(zip(frames, pids_list)):
            predictor.add_frame(pts_world, pids)

            predictions = {}
            history     = {}
            if predictor.ready():
                predictions = predictor.predict()
                history     = predictor.get_history()

            # 当前帧原始图像
            img = np.clip(images[t].transpose(1, 2, 0), 0, 1)   # [H, W, 3]

            # 高分辨率内参
            fx_l, fy_l, cx_l, cy_l = intrinsics[t]
            fx = fx_l * scale_x
            fy = fy_l * scale_y
            cx = cx_l * scale_x
            cy = cy_l * scale_y
            pose_cw = poses[t]

            fig, ax = plt.subplots(
                1, 1,
                figsize=(W_img / dpi, H_img / dpi),
                dpi=dpi,
            )
            ax.imshow(img)
            ax.set_xlim(0, W_img)
            ax.set_ylim(H_img, 0)
            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=0.93, bottom=0)

            n_dyn = 0

            # 🔵 历史轨迹（蓝色折线）
            if history:
                for pid, traj in history.items():
                    recent = traj[-max_history:]
                    if len(recent) < 2:
                        continue
                    seg_u, seg_v = [], []
                    for pt in recent:
                        u, v, valid = world_to_pixel(pt, pose_cw, fx, fy, cx, cy)
                        if valid[0]:
                            seg_u.append(float(u[0]))
                            seg_v.append(float(v[0]))
                        else:
                            if len(seg_u) >= 2:
                                ax.plot(seg_u, seg_v,
                                        color="#55AAFF", lw=0.7, alpha=0.55, zorder=2)
                            seg_u, seg_v = [], []
                    if len(seg_u) >= 2:
                        ax.plot(seg_u, seg_v,
                                color="#55AAFF", lw=0.7, alpha=0.55, zorder=2)

            # 🔴 当前帧动态点（红色实心点）
            if len(pts_world) > 0:
                u, v, valid = world_to_pixel(pts_world, pose_cw, fx, fy, cx, cy)
                u_v = u[valid]
                v_v = v[valid]
                n_dyn = int(valid.sum())
                if n_dyn > 0:
                    ax.scatter(u_v, v_v,
                               s=dot_radius ** 2, c="#FF3333",
                               alpha=0.75, linewidths=0, zorder=3)

            # 🟡 预测落点 + 连接线（仅对当前帧存在的点）
            current_pids = set(pids.tolist()) if len(pids) > 0 else set()
            if predictions and current_pids:
                for pid, steps in predictions.items():
                    if pid not in current_pids or not steps:
                        continue
                    # 历史末尾位置
                    hist_end = history.get(pid, [])
                    if not hist_end:
                        continue
                    u0, v0, val0 = world_to_pixel(hist_end[-1], pose_cw, fx, fy, cx, cy)
                    u1, v1, val1 = world_to_pixel(steps[0], pose_cw, fx, fy, cx, cy)
                    if not (val0[0] and val1[0]):
                        continue
                    u0f, v0f = float(u0[0]), float(v0[0])
                    u1f, v1f = float(u1[0]), float(v1[0])
                    # 连接线：历史末尾 → 预测落点（橙色虚线）
                    ax.plot([u0f, u1f], [v0f, v1f],
                            color="#FFA500", lw=0.6, alpha=0.7,
                            linestyle="--", zorder=4)
                    # 预测落点（小黄圈）
                    ax.scatter(u1f, v1f,
                               s=dot_radius ** 2,
                               facecolors="none",
                               edgecolors="#FFEE00",
                               linewidths=0.7,
                               alpha=0.9, zorder=5)

            # 图例
            legend_elems = [
                mpatches.Patch(color="#FF3333", label=f"Dynamic pts ({n_dyn})"),
                mpatches.Patch(color="#55AAFF", label="History"),
                plt.Line2D([0],[0], color="#FFA500", lw=1.2,
                           linestyle="--", label="Pred link"),
                mpatches.Patch(facecolor="none",
                               edgecolor="#FFEE00",
                               linewidth=1.2,
                               label="Pred point"),
            ]
            ax.legend(handles=legend_elems,
                      loc="upper right", fontsize=6,
                      framealpha=0.55, labelspacing=0.3)

            status = "PRED" if predictor.ready() else f"WARM {t+1}/{self.window_size}"
            ax.set_title(
                f"Frame {t+1:03d}/{T}  [{status}]",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="#000000", alpha=0.55),
            )

            frame_path = os.path.join(output_dir, f"frame_{t:04d}.png")
            fig.savefig(frame_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            # 关键帧单独保存（动态点最多的5帧 + 第一个PRED帧）
            if t in keyframes_to_save:
                kf_dir = os.path.join(self.save_dir, "dynamic_keyframes")
                os.makedirs(kf_dir, exist_ok=True)
                fig.savefig(os.path.join(kf_dir, f"keyframe_{t:04d}.png"),
                            dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            frame_paths.append(frame_path)

            print(f"\r  渲染帧 {t+1:03d}/{T} | 动态点={n_dyn:4d}", end="", flush=True)

        print()

        if not frame_paths:
            return ""

        gif_path = os.path.join(self.save_dir, gif_name)
        pil_frames = [PILImage.open(p).convert("RGBA") for p in frame_paths]
        base_size = pil_frames[0].size
        pil_frames = [f.resize(base_size, PILImage.LANCZOS) for f in pil_frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,
            duration=duration_ms,
            loop=0,
        )
        print(f"[DynamicBridge] GIF 已保存：{gif_path}")
        return gif_path

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _make_visualizer(self):
        if not self.enable_vis:
            return None
        try:
            from dynamic_prediction.visualizer_4d import Visualizer4D
            return Visualizer4D(window_name="DROID-W Dynamic Prediction")
        except Exception as e:
            print(f"[DynamicBridge] 无法启动 Open3D 可视化：{e}")
            return None

    def _prediction_loop(self, frames, pids_list, predictor, visualizer) -> dict:
        print(f"[DynamicBridge] 开始预测循环 "
              f"（window={self.window_size}, steps={self.predict_steps}）...")

        for t, (pts, pids) in enumerate(zip(frames, pids_list)):
            t0 = time.time()
            predictor.add_frame(pts, pids)

            predictions = {}
            if predictor.ready():
                predictions = predictor.predict()

            if visualizer is not None:
                current_pts = predictor.get_current_points()
                history = predictor.get_history()
                alive = visualizer.update(current_pts, history, predictions)
                if not alive:
                    print(f"\n[DynamicBridge] 窗口关闭，停止于第 {t} 帧。")
                    break

            if self.fps > 0:
                elapsed = time.time() - t0
                wait = 1.0 / self.fps - elapsed
                if wait > 0:
                    time.sleep(wait)

            status = "▶ PRED" if predictor.ready() else f"  WARM {t+1}/{self.window_size}"
            print(
                f"\r  [{status}] 帧 {t+1:04d}/{len(frames)}"
                f" | 动态点: {len(pts):4d}"
                f" | tracked: {predictor.n_tracked_points}",
                end="", flush=True,
            )

        print(f"\n[DynamicBridge] 预测循环完成。")
        return {}


# ===========================================================================
# 在线预测器（嵌入 Tracker 逐帧循环）
# ===========================================================================

class OnlineDynamicPredictor:
    """
    嵌入 Tracker.run() 逐帧循环的在线动态预测器。

    每当产生新关键帧时调用 update()，从 DepthVideo 实时读取
    不确定性图 + 视差图 + 位姿，提取动态点并运行 Kalman 预测。
    SLAM 结束后调用 finalize() 合成 GIF。

    Parameters
    ----------
    cfg         : SLAM 配置字典
    save_dir    : 输出目录
    """

    def __init__(self, cfg: dict, save_dir: str) -> None:
        self.cfg       = cfg
        self.save_dir  = save_dir
        dp_cfg         = cfg.get("dynamic_prediction", {})

        self.uncer_thresh  = dp_cfg.get("online_uncer_thresh", dp_cfg.get("uncer_thresh", 0.5))
        self.window_size   = dp_cfg.get("window_size",    8)
        self.predict_steps = dp_cfg.get("predict_steps",  2)
        self.match_radius  = dp_cfg.get("match_radius",   0.3)
        self.max_history   = dp_cfg.get("max_history",    6)
        self.dot_radius    = dp_cfg.get("dot_radius",     3.0)
        self.duration_ms   = dp_cfg.get("duration_ms",    150)

        from dynamic_prediction.sliding_window import SlidingWindowPredictor
        self._predictor = SlidingWindowPredictor(
            window_size   = self.window_size,
            predict_steps = self.predict_steps,
        )

        # 跨帧匹配状态
        self._prev_pts: "np.ndarray | None" = None
        self._prev_ids: "np.ndarray | None" = None
        self._next_id: int = 0

        # 保存每帧渲染结果
        self._frames_dir = os.path.join(save_dir, "dynamic_pred_frames")
        os.makedirs(self._frames_dir, exist_ok=True)
        self._frame_paths: list = []
        self._kf_count: int = 0

        import matplotlib
        matplotlib.use("Agg")

    # ------------------------------------------------------------------

    def update(self, video, kf_idx: int) -> None:
        """
        在新关键帧产生后调用。
        video : DepthVideo 实例（含 poses / disps / uncertainties / images / intrinsics）
        kf_idx: 当前关键帧在 video buffer 中的索引
        """
        import numpy as np

        # ---- 从 DepthVideo 读取当前关键帧数据 ----
        with video.get_lock():
            pose_7   = video.poses[kf_idx].cpu().numpy()          # [7]
            disp     = video.disps[kf_idx].cpu().numpy()           # [H_l, W_l]
            intr     = video.intrinsics[kf_idx].cpu().numpy()      # [4]
            uncer    = video.uncertainties[kf_idx].cpu().numpy()   # [H_l, W_l]
            img_t    = video.images[kf_idx].cpu().numpy()          # [3, H, W]

        H_l, W_l = disp.shape
        _, H_img, W_img = img_t.shape
        scale_x = W_img / W_l
        scale_y = H_img / H_l

        # ---- 提取动态像素 ----
        depth = np.where(disp > 1e-6, 1.0 / disp, 0.0)
        dyn_mask = (uncer > self.uncer_thresh) & (depth > 0.0)

        if dyn_mask.sum() == 0:
            pts_world = np.zeros((0, 3))
        else:
            fx, fy, cx, cy = intr
            v_g, u_g = np.meshgrid(np.arange(H_l, dtype=float),
                                    np.arange(W_l, dtype=float), indexing="ij")
            u = u_g[dyn_mask]; v = v_g[dyn_mask]; d = depth[dyn_mask]
            pts_cam = np.stack([(u - cx)/fx*d, (v - cy)/fy*d, d], axis=-1)
            pts_world = self._cam_to_world(pts_cam, pose_7)

        # ---- 跨帧最近邻 ID 匹配 ----
        pids = self._assign_ids(pts_world)

        # ---- 喂给 Kalman 滑动窗口 ----
        self._predictor.add_frame(pts_world, pids)

        predictions = {}
        history     = {}
        if self._predictor.ready():
            predictions = self._predictor.predict()
            history     = self._predictor.get_history()

        # ---- 渲染并保存帧 ----
        self._render_frame(
            img_t, pose_7, intr, scale_x, scale_y,
            pts_world, pids, history, predictions,
            H_img, W_img,
        )
        self._kf_count += 1

    def finalize(self, gif_name: str = "dynamic_prediction_online.gif") -> str:
        """SLAM 结束后调用，合成 GIF 并生成关键帧图片。"""
        if not self._frame_paths:
            return ""

        gif_path = os.path.join(self.save_dir, gif_name)
        pil_frames = [PILImage.open(p).convert("RGBA") for p in self._frame_paths]
        base_size = pil_frames[0].size
        pil_frames = [f.resize(base_size, PILImage.LANCZOS) for f in pil_frames]
        pil_frames[0].save(
            gif_path, save_all=True,
            append_images=pil_frames[1:],
            optimize=False, duration=self.duration_ms, loop=0,
        )
        print(f"[OnlineDynPred] GIF saved: {gif_path}")

        # 关键帧单独保存（动态点最多的 5 帧 + 第一个 PRED 帧）
        import numpy as np
        sizes = [0] * len(self._frame_paths)
        for i, p in enumerate(self._frame_paths):
            sizes[i] = os.path.getsize(p)
        top5 = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)[:5]
        kf_set = set(top5) | {self.window_size - 1}
        kf_dir = os.path.join(self.save_dir, "dynamic_keyframes_online")
        os.makedirs(kf_dir, exist_ok=True)
        for idx in kf_set:
            if idx < len(self._frame_paths):
                src = self._frame_paths[idx]
                dst = os.path.join(kf_dir, f"keyframe_{idx:04d}.png")
                PILImage.open(src).save(dst)
        print(f"[OnlineDynPred] Keyframes saved to: {kf_dir}")
        return gif_path

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _assign_ids(self, pts_world: "np.ndarray") -> "np.ndarray":
        import numpy as np
        N = len(pts_world)
        if N == 0:
            return np.zeros(0, dtype=int)

        if self._prev_pts is None or len(self._prev_pts) == 0:
            ids = np.arange(self._next_id, self._next_id + N, dtype=int)
            self._next_id += N
        else:
            ids = np.full(N, -1, dtype=int)
            diff = pts_world[:, None, :] - self._prev_pts[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)           # [N, M]
            nearest_idx  = dist.argmin(axis=1)
            nearest_dist = dist[np.arange(N), nearest_idx]
            matched = nearest_dist < self.match_radius
            ids[matched] = self._prev_ids[nearest_idx[matched]]
            n_new = (~matched).sum()
            if n_new > 0:
                ids[~matched] = np.arange(self._next_id,
                                           self._next_id + n_new, dtype=int)
                self._next_id += n_new

        self._prev_pts = pts_world.copy()
        self._prev_ids = ids.copy()
        return ids

    @staticmethod
    def _cam_to_world(pts_cam: "np.ndarray", pose_7: "np.ndarray") -> "np.ndarray":
        import numpy as np
        tx, ty, tz, qx, qy, qz, qw = pose_7
        norm = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        R_cw = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])
        t_cw = np.array([tx, ty, tz])
        return (R_cw.T @ (pts_cam - t_cw[None, :]).T).T

    def _render_frame(
        self, img_t, pose_7, intr, scale_x, scale_y,
        pts_world, pids, history, predictions,
        H_img, W_img,
    ) -> None:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # 重建 [4,4] world-to-camera 矩阵
        tx, ty, tz, qx, qy, qz, qw = pose_7
        norm = (qx**2 + qy**2 + qz**2 + qw**2) ** 0.5
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        R_cw = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])
        t_cw = np.array([tx, ty, tz])
        pose_cw = np.eye(4)
        pose_cw[:3, :3] = R_cw
        pose_cw[:3, 3]  = t_cw

        fx = intr[0] * scale_x; fy = intr[1] * scale_y
        cx = intr[2] * scale_x; cy = intr[3] * scale_y

        def w2p(pts_w):
            pts_w = np.asarray(pts_w)
            if pts_w.ndim == 1: pts_w = pts_w[None, :]
            if len(pts_w) == 0:
                return np.array([]), np.array([]), np.array([], dtype=bool)
            pts_c = (R_cw @ pts_w.T).T + t_cw[None, :]
            z = pts_c[:, 2]
            valid = z > 0.01
            u = np.where(valid, fx * pts_c[:, 0] / (z+1e-8) + cx, -1.)
            v = np.where(valid, fy * pts_c[:, 1] / (z+1e-8) + cy, -1.)
            valid &= (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
            return u, v, valid

        img = np.clip(img_t.transpose(1, 2, 0), 0, 1)
        dpi = 100
        fig, ax = plt.subplots(1, 1,
                               figsize=(W_img/dpi, H_img/dpi), dpi=dpi)
        ax.imshow(img)
        ax.set_xlim(0, W_img); ax.set_ylim(H_img, 0)
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=0.93, bottom=0)

        n_dyn = 0

        # 🔵 历史轨迹
        for pid, traj in history.items():
            recent = traj[-self.max_history:]
            if len(recent) < 2: continue
            seg_u, seg_v = [], []
            for pt in recent:
                u, v, val = w2p(pt)
                if val[0]:
                    seg_u.append(float(u[0])); seg_v.append(float(v[0]))
                else:
                    if len(seg_u) >= 2:
                        ax.plot(seg_u, seg_v, color="#55AAFF",
                                lw=0.7, alpha=0.55, zorder=2)
                    seg_u, seg_v = [], []
            if len(seg_u) >= 2:
                ax.plot(seg_u, seg_v, color="#55AAFF",
                        lw=0.7, alpha=0.55, zorder=2)

        # 🔴 当前动态点
        if len(pts_world) > 0:
            u, v, valid = w2p(pts_world)
            n_dyn = int(valid.sum())
            if n_dyn > 0:
                ax.scatter(u[valid], v[valid],
                           s=self.dot_radius**2, c="#FF3333",
                           alpha=0.75, linewidths=0, zorder=3)

        # 🟡 预测落点 + 橙色连接线（仅当前帧存在的点）
        current_pids = set(pids.tolist()) if len(pids) > 0 else set()
        if predictions and current_pids:
            for pid, steps in predictions.items():
                if pid not in current_pids or not steps: continue
                hist_end = history.get(pid, [])
                if not hist_end: continue
                u0, v0, val0 = w2p(hist_end[-1])
                u1, v1, val1 = w2p(steps[0])
                if not (val0[0] and val1[0]): continue
                ax.plot([float(u0[0]), float(u1[0])],
                        [float(v0[0]), float(v1[0])],
                        color="#FFA500", lw=0.6, alpha=0.7,
                        linestyle="--", zorder=4)
                ax.scatter(float(u1[0]), float(v1[0]),
                           s=self.dot_radius**2,
                           facecolors="none", edgecolors="#FFEE00",
                           linewidths=0.7, alpha=0.9, zorder=5)

        # 图例 + 标题
        status = "PRED" if self._predictor.ready() else f"WARM {self._kf_count+1}/{self.window_size}"
        legend_elems = [
            mpatches.Patch(color="#FF3333", label=f"Dynamic pts ({n_dyn})"),
            mpatches.Patch(color="#55AAFF", label="History"),
            plt.Line2D([0],[0], color="#FFA500", lw=1.2,
                       linestyle="--", label="Pred link"),
            mpatches.Patch(facecolor="none", edgecolor="#FFEE00",
                           linewidth=1.2, label="Pred point"),
        ]
        ax.legend(handles=legend_elems, loc="upper right",
                  fontsize=6, framealpha=0.55, labelspacing=0.3)
        ax.set_title(f"KF {self._kf_count+1:03d}  [{status}]",
                     fontsize=8, color="white",
                     bbox=dict(boxstyle="round,pad=0.25",
                               fc="#000000", alpha=0.55))

        frame_path = os.path.join(self._frames_dir,
                                   f"online_{self._kf_count:04d}.png")
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        self._frame_paths.append(frame_path)
