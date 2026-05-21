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


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) + 1e-12
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
    ], dtype=float)


def _pose_to_w2c_matrix(pose: np.ndarray) -> np.ndarray:
    """Return a 4x4 world-to-camera matrix from DROID-W pose formats."""
    pose = np.asarray(pose)
    if pose.shape == (4, 4):
        # DepthVideo.save_video() writes camera-to-world matrices to "poses".
        return np.linalg.inv(pose)
    if pose.shape == (7,):
        tx, ty, tz, qx, qy, qz, qw = pose
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
        mat[:3, 3] = np.array([tx, ty, tz], dtype=float)
        return mat
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


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
        dp_cfg = cfg.get("dynamic_prediction", {})
        self.match_radius = dp_cfg.get("match_radius", 0.3)
        self.process_noise = dp_cfg.get("process_noise", 0.1)
        self.measurement_noise = dp_cfg.get("measurement_noise", 1.0)

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

        bridge = DROIDWBridge(
            uncer_thresh=self.uncer_thresh,
            match_radius=self.match_radius,
        )
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
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
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
        duration_ms: int = 100,
        max_history: int = 6,
        dot_radius: float = 3.0,
        dataset_dir: Optional[str] = None,
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
        pose_data  = np.asarray(npz["tum_poses"] if "tum_poses" in npz else npz["poses"])
        intrinsics = np.asarray(npz["intrinsics"])  # [T, 4] 低分辨率内参

        T, _, H_img, W_img = images.shape
        # 内参是在 1/8 分辨率下标定的，缩放到原图分辨率
        scale_x = W_img / (W_img // 8)  # = 8.0
        scale_y = H_img / (H_img // 8)  # = 8.0

        bridge = DROIDWBridge(
            uncer_thresh=self.uncer_thresh,
            match_radius=self.match_radius,
        )
        dw_data = bridge.load(video_npz_path)
        frames, pids_list = bridge.extract_dynamic_points(dw_data)

        predictor = SlidingWindowPredictor(
            window_size=self.window_size,
            predict_steps=self.predict_steps,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
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
            pose_cw = _pose_to_w2c_matrix(pose_data[t])

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

            # 🟡 预测落点 + 连接线（仅对当前帧存在的、且已有速度估计的点）
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

        # ---- 在关键帧之间插入原始 RGB 帧，消除跳帧感 ----
        final_frame_paths = frame_paths      # 默认只用关键帧
        final_durations   = [duration_ms] * len(frame_paths)

        if dataset_dir is not None:
            rgb_txt = os.path.join(dataset_dir, "rgb.txt")
            if os.path.exists(rgb_txt):
                # 解析 rgb.txt → {timestamp_int: abs_path}
                ts2path = {}
                with open(rgb_txt) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            ts_str, rel_path = parts[0], parts[1]
                            ts_int = int(float(ts_str) * 1000)   # ms
                            abs_path = os.path.join(dataset_dir, rel_path)
                            ts2path[ts_int] = abs_path

                kf_timestamps = [int(ts * 1000) for ts in npz["timestamps"]]
                all_raw_ts    = sorted(ts2path.keys())

                interp_dir = os.path.join(output_dir, "interp")
                os.makedirs(interp_dir, exist_ok=True)

                final_frame_paths = []
                final_durations   = []

                for ki, kf_ts in enumerate(kf_timestamps):
                    # 插入从上一关键帧到当前关键帧之间的原始帧
                    prev_ts = kf_timestamps[ki - 1] if ki > 0 else kf_ts
                    between = [t for t in all_raw_ts if prev_ts < t < kf_ts]
                    for raw_ts in between:
                        raw_src = ts2path[raw_ts]
                        raw_dst = os.path.join(
                            interp_dir, f"raw_{raw_ts:015d}.png")
                        if not os.path.exists(raw_dst):
                            # 直接复制原始帧（不带任何标注）
                            pil_raw = PILImage.open(raw_src).convert("RGBA")
                            pil_raw.save(raw_dst)
                        final_frame_paths.append(raw_dst)
                        final_durations.append(duration_ms)
                    # 当前关键帧（带预测标注）
                    final_frame_paths.append(frame_paths[ki])
                    final_durations.append(duration_ms * 2)  # 关键帧停留稍长

                print(f"[DynamicBridge] 插帧后总帧数: {len(final_frame_paths)} "
                      f"（关键帧 {len(frame_paths)}，插入原始帧 "
                      f"{len(final_frame_paths)-len(frame_paths)}）")

        gif_path = os.path.join(self.save_dir, gif_name)
        base_img  = PILImage.open(final_frame_paths[0]).convert("RGBA")
        base_size = base_img.size
        pil_frames = [
            PILImage.open(p).convert("RGBA").resize(base_size, PILImage.LANCZOS)
            for p in final_frame_paths
        ]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,
            duration=final_durations,
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

    def __init__(self, cfg: dict, save_dir: str, dataset_dir: str = "") -> None:
        self.cfg       = cfg
        self.save_dir  = save_dir
        self.dataset_dir = dataset_dir   # 原始数据集目录，用于插入原始帧
        dp_cfg         = cfg.get("dynamic_prediction", {})

        self.uncer_thresh  = dp_cfg.get("uncer_thresh",   0.8)
        self.window_size   = dp_cfg.get("window_size",    8)
        self.predict_steps = dp_cfg.get("predict_steps",  2)
        self.match_radius  = dp_cfg.get("match_radius",   0.3)
        self.max_history   = dp_cfg.get("max_history",    6)
        self.dot_radius    = dp_cfg.get("dot_radius",     3.0)
        self.duration_ms   = dp_cfg.get("duration_ms",    150)
        self.pred_uncer_value = dp_cfg.get("pred_uncer_value", 2.0)
        self.blend_alpha      = dp_cfg.get("blend_alpha",      0.7)
        self.splat_radius     = dp_cfg.get("splat_radius",     2)
        self.process_noise    = dp_cfg.get("process_noise",    0.1)
        self.measurement_noise = dp_cfg.get("measurement_noise", 1.0)
        self.online_render    = dp_cfg.get("online_render",    False)
        self.online_make_gif  = dp_cfg.get("online_make_gif",  False)
        self.online_insert_raw_frames = dp_cfg.get("online_insert_raw_frames", False)
        self.max_feedback_points = dp_cfg.get("max_feedback_points", 200)
        self.max_feedback_coverage = dp_cfg.get("max_feedback_coverage", 0.08)

        from dynamic_prediction.sliding_window import SlidingWindowPredictor
        self._predictor = SlidingWindowPredictor(
            window_size   = self.window_size,
            predict_steps = self.predict_steps,
            process_noise = self.process_noise,
            measurement_noise = self.measurement_noise,
        )

        # 跨帧匹配状态
        self._prev_pts: "np.ndarray | None" = None
        self._prev_ids: "np.ndarray | None" = None
        self._next_id: int = 0

        # 保存每帧渲染结果
        self._frames_dir = os.path.join(save_dir, "dynamic_pred_frames")
        os.makedirs(self._frames_dir, exist_ok=True)
        self._frame_paths: list = []
        self._kf_timestamps: list = []   # 每个关键帧对应的原始序列帧索引
        self._kf_count: int = 0
        self._next_kf_to_update: int = 0
        self._pending_predictions: dict = {}
        self._feedback_log_path = os.path.join(save_dir, "dynamic_feedback_stats.csv")
        if not os.path.exists(self._feedback_log_path):
            with open(self._feedback_log_path, "w", encoding="utf-8") as f:
                f.write(
                    "kf_idx,n_predictions,n_projected,mask_pixels,coverage,"
                    "uncer_before_max,uncer_after_max,applied\n"
                )

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
        if not hasattr(video, "uncertainties"):
            return

        with video.get_lock():
            pose_7   = video.poses[kf_idx].cpu().numpy()          # [7]
            disp     = video.disps[kf_idx].cpu().numpy()           # [H_l, W_l]
            intr     = video.intrinsics[kf_idx].cpu().numpy()      # [4]
            uncer    = video.uncertainties[kf_idx].cpu().numpy()   # [H_l, W_l]
            img_t    = video.images[kf_idx].cpu().numpy()          # [3, H, W]
            frame_ts = int(video.timestamp[kf_idx].item())         # 原始序列帧索引

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

        # 无论 WARM 期还是 PRED 期，对所有已有 Kalman 状态的点强制预测
        # 直接访问 _predictors，绕过 ready() 检查
        predictions = {
            pid: kf.predict(self.predict_steps)
            for pid, kf in self._predictor._predictors.items()
        }
        history = self._predictor.get_history()

        # 缓存给下一关键帧。下一帧刚 append 后、frontend BA 前写入，
        # 这样预测动态区域才会真正参与当前 SLAM 优化权重。
        self._pending_predictions = predictions

        if self.online_render:
            self._render_frame(
                img_t, pose_7, intr, scale_x, scale_y,
                pts_world, pids, history, predictions,
                H_img, W_img,
            )
        self._kf_timestamps.append(frame_ts)
        self._kf_count += 1

    def apply_pending_feedback(self, video, kf_idx: int) -> bool:
        """
        将上一关键帧预测出的动态落点写入当前关键帧 uncertainty。

        调用时机必须在当前关键帧 append 之后、frontend BA 之前。
        返回 True 表示有预测 mask 被写入。
        """
        if not hasattr(video, "uncertainties") or not self._pending_predictions:
            return False

        with video.get_lock():
            H_l, W_l = video.uncertainties[kf_idx].shape

        return self._feedback_to_video(
            video,
            kf_idx,
            self._pending_predictions,
            H_l,
            W_l,
            pred_uncer_value=self.pred_uncer_value,
            blend_alpha=self.blend_alpha,
            splat_radius=self.splat_radius,
            max_feedback_points=self.max_feedback_points,
            max_feedback_coverage=self.max_feedback_coverage,
        )

    def finalize(self, gif_name: str = "dynamic_prediction_online.gif") -> str:
        """SLAM 结束后调用，合成 GIF（插入原始帧使播放流畅）并生成关键帧图片。"""
        if not self.online_make_gif or not self._frame_paths:
            return ""

        # ---- 加载原始帧列表 ----
        # rgb.txt 格式：  timestamp  rgb/filename.png
        raw_frames = []   # [(frame_idx, abs_path), ...]
        if self.dataset_dir:
            rgb_txt = os.path.join(self.dataset_dir, "rgb.txt")
            if os.path.exists(rgb_txt):
                with open(rgb_txt) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        rel_path = parts[1]
                        abs_path = os.path.join(self.dataset_dir, rel_path)
                        raw_frames.append(abs_path)
                print(f"[OnlineDynPred] 原始帧共 {len(raw_frames)} 张")

        # ---- 拼合带插值的帧序列 ----
        base_size = PILImage.open(self._frame_paths[0]).size
        all_frames = []      # [(pil_image, duration_ms)]

        def load_raw(path):
            img = PILImage.open(path).convert("RGBA").resize(base_size, PILImage.LANCZOS)
            return img

        def load_kf(path):
            return PILImage.open(path).convert("RGBA").resize(base_size, PILImage.LANCZOS)

        for i, (kf_path, kf_ts) in enumerate(zip(self._frame_paths, self._kf_timestamps)):
            # 插入上一关键帧到本关键帧之间的原始帧
            if self.online_insert_raw_frames and raw_frames and i > 0:
                prev_ts = self._kf_timestamps[i - 1]
                # 上一关键帧之后、本关键帧之前的原始帧索引
                for raw_idx in range(prev_ts + 1, kf_ts):
                    if raw_idx < len(raw_frames) and os.path.exists(raw_frames[raw_idx]):
                        all_frames.append((load_raw(raw_frames[raw_idx]), self.duration_ms))
            # 关键帧本身（带预测标注，停留时间稍长）
            all_frames.append((load_kf(kf_path), int(self.duration_ms * 1.5)))

        if not all_frames:
            return ""

        gif_path = os.path.join(self.save_dir, gif_name)
        imgs   = [f[0] for f in all_frames]
        durs   = [f[1] for f in all_frames]
        imgs[0].save(
            gif_path, save_all=True,
            append_images=imgs[1:],
            optimize=False, duration=durs, loop=0,
        )
        print(f"[OnlineDynPred] GIF saved: {gif_path}  ({len(all_frames)} frames total, {len(self._frame_paths)} keyframes + {len(all_frames)-len(self._frame_paths)} raw)")

        # ---- 关键帧单独保存 ----
        sizes = [os.path.getsize(p) for p in self._frame_paths]
        top5  = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)[:5]
        kf_set = set(top5) | {self.window_size - 1}
        kf_dir = os.path.join(self.save_dir, "dynamic_keyframes_online")
        os.makedirs(kf_dir, exist_ok=True)
        for idx in kf_set:
            if idx < len(self._frame_paths):
                PILImage.open(self._frame_paths[idx]).save(
                    os.path.join(kf_dir, f"keyframe_{idx:04d}.png"))
        print(f"[OnlineDynPred] Keyframes saved to: {kf_dir}")
        return gif_path

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------


    def _feedback_to_video(
        self,
        video,
        kf_idx: int,
        predictions: dict,
        H_l: int,
        W_l: int,
        pred_uncer_value: float = 2.0,
        blend_alpha: float = 0.7,
        splat_radius: int = 2,
        max_feedback_points: int = 200,
        max_feedback_coverage: float = 0.08,
    ) -> bool:
        """
        把 Kalman 预测的动态点落点写回 video.uncertainties[kf_idx]。

        原理
        ----
        video.ba() 内部会把不确定性换算成 BA 权重：
            uncer_rescaled = clamp(45 * uncer - 35, min=0.1)
            weight_mask    = clamp(1 / uncer_rescaled, 0, 1)
        当 uncer >= 0.78 时 weight_mask <= 1.0，uncer=2.0 时 weight_mask≈0.011
        写入高不确定性值 → 该像素 BA 权重趋近 0 → 不参与位姿/深度优化

        Parameters
        ----------
        video            : DepthVideo 实例
        kf_idx           : 写入的当前关键帧索引
        predictions      : {pid: [pred_step0, pred_step1, ...]}  世界坐标
        H_l / W_l        : 低分辨率图像尺寸（1/8 原图）
        pred_uncer_value : 写入的不确定性值（>0.78 即开始压制，2.0 几乎完全压制）
        blend_alpha      : 预测掩码与原有不确定性的混合权重（1.0=完全覆盖）
        splat_radius     : 以预测像素为中心的扩散半径（像素，低分辨率下）
        max_feedback_points    : 最多反馈多少个预测点，防止 dense mask 过大
        max_feedback_coverage  : 预测 mask 允许覆盖的最大图像比例
        """
        import torch

        if kf_idx >= video.uncertainties.shape[0]:
            return False
        if not predictions:
            return False

        with video.get_lock():
            pose_7 = video.poses[kf_idx].cpu().numpy()
            intr   = video.intrinsics[kf_idx].cpu().numpy()

        import numpy as np
        tx, ty, tz, qx, qy, qz, qw = pose_7
        R_cw = _quat_to_rot(qx, qy, qz, qw)
        t_cw = np.array([tx, ty, tz])
        fx, fy, cx, cy = intr  # 低分辨率内参，直接对应 H_l x W_l

        # 收集所有预测落点（第 1 步）并投影到低分辨率像素坐标
        projected_centers = []
        pred_items = list(predictions.items())
        if max_feedback_points and len(pred_items) > max_feedback_points:
            step = max(1, len(pred_items) // max_feedback_points)
            pred_items = pred_items[::step][:max_feedback_points]

        pred_mask = np.zeros((H_l, W_l), dtype=bool)
        for pid, steps in pred_items:
            if not steps:
                continue
            pt_w = np.asarray(steps[0], dtype=float)
            if pt_w.ndim == 1:
                pt_w = pt_w[None, :]
            pt_c = (R_cw @ pt_w.T).T + t_cw[None, :]
            z = pt_c[0, 2]
            if z < 0.01:
                continue
            u = fx * pt_c[0, 0] / (z + 1e-8) + cx
            v = fy * pt_c[0, 1] / (z + 1e-8) + cy
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W_l and 0 <= vi < H_l):
                continue
            projected_centers.append((ui, vi))

        n_projected = len(projected_centers)
        for ui, vi in projected_centers:
            # splat：以预测落点为中心扩散 splat_radius 像素
            r = splat_radius
            u0 = max(ui - r, 0); u1 = min(ui + r + 1, W_l)
            v0 = max(vi - r, 0); v1 = min(vi + r + 1, H_l)
            if u1 > u0 and v1 > v0:
                pred_mask[v0:v1, u0:u1] = True

        if not pred_mask.any():
            self._write_feedback_stat(
                kf_idx, len(predictions), n_projected, 0, H_l * W_l,
                0.0, 0.0, False,
            )
            return False

        mask_pixels = int(pred_mask.sum())
        coverage = mask_pixels / max(H_l * W_l, 1)
        if max_feedback_coverage and coverage > max_feedback_coverage:
            self._write_feedback_stat(
                kf_idx, len(predictions), n_projected, mask_pixels, H_l * W_l,
                0.0, 0.0, False,
            )
            return False

        # 转为 torch tensor，写回 video.uncertainties[kf_idx]
        pred_mask_t = torch.from_numpy(pred_mask).to(video.uncertainties.device)
        with video.get_lock():
            orig = video.uncertainties[kf_idx]            # [H_l, W_l]
            before_max = float(orig.max().item())
            # 混合：预测动态区域 → 拉高不确定性
            high_val = torch.full_like(orig, pred_uncer_value)
            video.uncertainties[kf_idx] = torch.where(
                pred_mask_t,
                blend_alpha * high_val + (1.0 - blend_alpha) * orig,
                orig,
            )
            after_max = float(video.uncertainties[kf_idx].max().item())
        self._write_feedback_stat(
            kf_idx,
            len(predictions),
            n_projected,
            mask_pixels,
            H_l * W_l,
            before_max,
            after_max,
            True,
        )
        return True

    def _write_feedback_stat(
        self,
        kf_idx: int,
        n_predictions: int,
        n_projected: int,
        mask_pixels: int,
        total_pixels: int,
        uncer_before_max: float,
        uncer_after_max: float,
        applied: bool,
    ) -> None:
        coverage = mask_pixels / max(total_pixels, 1)
        with open(self._feedback_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{kf_idx},{n_predictions},{n_projected},{mask_pixels},"
                f"{coverage:.8f},{uncer_before_max:.6f},"
                f"{uncer_after_max:.6f},{int(applied)}\n"
            )

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
        R_cw = _quat_to_rot(qx, qy, qz, qw)
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
        R_cw = _quat_to_rot(qx, qy, qz, qw)
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

        # 🟡 预测落点 + 橙色连接线（仅当前帧存在的、且已有速度估计的点）
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
