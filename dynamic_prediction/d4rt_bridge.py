"""
d4rt_bridge.py
==============
数据加载接口，支持三种来源：

1. D4RTLoader      —— 加载 D4RT 输出的 .npz 文件
   格式：tracks [N,T,3]，visibility [N,T]，camera_poses [T,4,4]

2. DROIDWBridge    —— 从 DROID-W 保存的 video.npz 提取动态点轨迹
   需要 uncertainty maps（高不确定性像素 ≈ 动态点）

3. SyntheticDataGenerator —— 生成合成螺旋/随机轨迹用于 demo 模式
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. D4RT Loader（官方 .npz 格式）
# ---------------------------------------------------------------------------

class D4RTLoader:
    """
    加载 D4RT 输出的 .npz 文件。

    预期字段：
        tracks        : [N, T, 3]  3D 轨迹（必须）
        visibility    : [N, T]     可见性 mask（可选，默认全 True）
        camera_poses  : [T, 4, 4]  相机位姿（可选）
    """

    @staticmethod
    def load(path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"D4RT .npz 文件不存在：{path}")

        data = np.load(path, allow_pickle=True)

        if "tracks" not in data:
            raise KeyError(
                "D4RT .npz 文件中缺少 'tracks' 字段，"
                "期望形状 [N_points, N_frames, 3]"
            )

        tracks = np.asarray(data["tracks"], dtype=np.float32)   # [N, T, 3]
        assert tracks.ndim == 3 and tracks.shape[2] == 3, (
            f"tracks 形状应为 [N, T, 3]，实际 {tracks.shape}"
        )

        N, T, _ = tracks.shape

        # 可见性 mask
        if "visibility" in data:
            visibility = np.asarray(data["visibility"], dtype=bool)   # [N, T]
        else:
            visibility = np.ones((N, T), dtype=bool)

        # 相机位姿（可选）
        camera_poses = None
        if "camera_poses" in data:
            camera_poses = np.asarray(data["camera_poses"], dtype=np.float32)  # [T, 4, 4]

        return {
            "tracks": tracks,              # [N, T, 3]
            "visibility": visibility,      # [N, T]
            "camera_poses": camera_poses,  # [T, 4, 4] | None
        }

    @staticmethod
    def to_frame_sequence(
        d4rt_data: Dict,
        visible_only: bool = True,
    ) -> Tuple[list, list]:
        """
        将 D4RT 输出转换为 SlidingWindowPredictor.add_frame() 所需格式。

        Returns
        -------
        frames      : list of np.ndarray [N_t, 3]，每帧的 3D 点
        point_ids   : list of np.ndarray [N_t]，每帧的点 ID
        """
        tracks = d4rt_data["tracks"]          # [N, T, 3]
        vis = d4rt_data["visibility"]         # [N, T]
        N, T, _ = tracks.shape

        frames: list = []
        point_ids_list: list = []

        for t in range(T):
            if visible_only:
                mask = vis[:, t]              # [N] bool
            else:
                mask = np.ones(N, dtype=bool)

            pts = tracks[mask, t, :]          # [N_t, 3]
            pids = np.where(mask)[0]          # [N_t] — 原始点索引即为 ID

            frames.append(pts)
            point_ids_list.append(pids)

        return frames, point_ids_list


# ---------------------------------------------------------------------------
# 2. DROID-W Bridge（从 video.npz 提取动态点轨迹）
# ---------------------------------------------------------------------------

class DROIDWBridge:
    """
    从 DROID-W 保存的 video.npz 中提取动态点。

    原理：
        1. 加载 poses / disps / intrinsics / uncertainties
        2. 对每帧，将 uncertainty > uncer_thresh 的像素视为动态点
        3. 利用 inverse depth 反投影至相机坐标系
        4. 通过 SE3 逆变换转换至世界坐标系
        5. 输出 per-frame 点云（注意：跨帧点 ID 匹配需额外处理）

    注意
    ----
    此桥接方案给出的是 per-frame 独立点云（无跨帧 ID 对应）。
    若 D4RT 提供了带 ID 的轨迹，优先使用 D4RTLoader。
    """

    def __init__(self, uncer_thresh: float = 0.8, match_radius: float = 0.3) -> None:
        self.uncer_thresh = uncer_thresh
        self.match_radius = match_radius  # 跨帧匹配半径（米）

    def load(self, video_npz_path: str) -> Dict:
        """
        加载 video.npz，返回原始数据字典。
        兼容 DROID-W 实际输出格式：
          poses=[T,4,4] 旋转矩阵 或 [T,7] 四元数
          droid_disps=[T,H,W]，timestamps=[T]
        """
        if not os.path.exists(video_npz_path):
            raise FileNotFoundError(f"video.npz 不存在：{video_npz_path}")

        data = np.load(video_npz_path, allow_pickle=True)

        poses_raw = np.asarray(data["poses"])  # [T,4,4] 或 [T,7]

        # disps: 优先使用与 uncertainties 尺寸一致的低分辨率版本
        if "droid_disps" in data:
            disps = np.asarray(data["droid_disps"])
        elif "disps" in data:
            disps = np.asarray(data["disps"])
        else:
            raise KeyError("video.npz 中找不到视差图字段（droid_disps / disps）")

        # timestamps: 兼容两种字段名
        if "timestamps" in data:
            tstamps = np.asarray(data["timestamps"])
        elif "tstamps" in data:
            tstamps = np.asarray(data["tstamps"])
        else:
            tstamps = np.arange(poses_raw.shape[0], dtype=np.float32)

        intrinsics = np.asarray(data["intrinsics"])
        uncertainties = (
            np.asarray(data["uncertainties"])
            if "uncertainties" in data else None
        )

        return {
            "poses": poses_raw,
            "disps": disps,
            "tstamps": tstamps,
            "intrinsics": intrinsics,
            "uncertainties": uncertainties,
        }

    def extract_dynamic_points(self, droidw_data: Dict) -> Tuple[list, list]:
        """
        从 DROID-W 数据中提取每帧的动态点云，并进行跨帧最近邻 ID 匹配。

        Returns
        -------
        frames      : list of np.ndarray [N_t, 3]（世界坐标）
        point_ids   : list of np.ndarray [N_t]（跨帧一致的物理点 ID）
        """
        poses = droidw_data["poses"]
        disps = droidw_data["disps"]
        intrinsics = droidw_data["intrinsics"]
        uncertainties = droidw_data["uncertainties"]

        T, H, W = disps.shape
        raw_frames: list = []

        # --- 第一步：提取每帧原始点云 ---
        for t in range(T):
            fx, fy, cx, cy = intrinsics[t]

            v_grid, u_grid = np.meshgrid(
                np.arange(H, dtype=float),
                np.arange(W, dtype=float),
                indexing="ij",
            )

            disp_t = disps[t]
            depth_t = np.where(disp_t > 1e-6, 1.0 / disp_t, 0.0)

            if uncertainties is not None:
                dyn_mask = uncertainties[t] > self.uncer_thresh
            else:
                dyn_mask = depth_t > 0.0

            dyn_mask &= depth_t > 0.0

            if dyn_mask.sum() == 0:
                raw_frames.append(np.zeros((0, 3)))
                continue

            u = u_grid[dyn_mask]
            v = v_grid[dyn_mask]
            d = depth_t[dyn_mask]

            pts_cam = np.stack([
                (u - cx) / fx * d,
                (v - cy) / fy * d,
                d,
            ], axis=-1)

            pts_world = self._cam_to_world(pts_cam, poses[t])
            raw_frames.append(pts_world)

        # --- 第二步：跨帧最近邻 ID 匹配 ---
        return self._assign_consistent_ids(raw_frames)

    def _assign_consistent_ids(self, raw_frames: list) -> Tuple[list, list]:
        """
        对每帧点云用最近邻匹配赋予跨帧一致的全局 ID。

        策略：
            - 第 0 帧：所有点直接分配新 ID
            - 后续帧：每个点找上一帧最近的点
              · 距离 < match_radius  → 继承该点 ID（同一物理点）
              · 距离 >= match_radius → 分配新 ID（新出现的动态点）
        """
        point_ids_list: list = []
        next_id: int = 0
        prev_pts: np.ndarray = None
        prev_ids: np.ndarray = None

        for pts in raw_frames:
            N = pts.shape[0]
            if N == 0:
                point_ids_list.append(np.zeros(0, dtype=int))
                continue

            if prev_pts is None or prev_pts.shape[0] == 0:
                ids = np.arange(next_id, next_id + N, dtype=int)
                next_id += N
            else:
                ids = np.full(N, -1, dtype=int)

                # 距离矩阵 [N, M]
                diff = pts[:, None, :] - prev_pts[None, :, :]
                dist = np.linalg.norm(diff, axis=-1)

                nearest_idx = dist.argmin(axis=1)
                nearest_dist = dist[np.arange(N), nearest_idx]

                matched = nearest_dist < self.match_radius
                ids[matched] = prev_ids[nearest_idx[matched]]

                n_new = (~matched).sum()
                if n_new > 0:
                    ids[~matched] = np.arange(next_id, next_id + n_new, dtype=int)
                    next_id += n_new

            point_ids_list.append(ids)
            prev_pts = pts
            prev_ids = ids

        return raw_frames, point_ids_list

    @staticmethod
    def _cam_to_world(
        pts_cam: np.ndarray,
        pose_cw: np.ndarray,
    ) -> np.ndarray:
        """
        将相机坐标系下的点转换到世界坐标系。

        pose_cw : [4, 4] world-to-camera 变换矩阵
                  或 [7]  [tx, ty, tz, qx, qy, qz, qw]（world-to-camera）
        返回 [N, 3]
        """
        if pose_cw.shape == (4, 4):
            R_cw = pose_cw[:3, :3]
            t_cw = pose_cw[:3, 3]
        else:
            # [7]: tx, ty, tz, qx, qy, qz, qw
            t_cw = pose_cw[:3]
            qx, qy, qz, qw = pose_cw[3], pose_cw[4], pose_cw[5], pose_cw[6]
            R_cw = DROIDWBridge._quat_to_rot(qx, qy, qz, qw)

        # X_world = R_cw^T @ (X_cam - t_cw)
        pts_world = (R_cw.T @ (pts_cam - t_cw[None, :]).T).T
        return pts_world

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """四元数 → 3×3 旋转矩阵"""
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        R = np.array([
            [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [  2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
        ], dtype=float)
        return R


# ---------------------------------------------------------------------------
# 3. 合成数据生成器（Demo 模式）
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    生成用于演示的合成 3D 轨迹。

    生成模式：
        "spiral"   : 螺旋轨迹（方向随机，带随机偏移）
        "random"   : 带随机加速度的随机游走
        "mixed"    : 前半部分螺旋 + 后半部分随机游走
    """

    def __init__(
        self,
        n_points: int = 80,
        n_frames: int = 60,
        mode: str = "mixed",
        noise_std: float = 0.02,
        rng_seed: Optional[int] = 42,
    ) -> None:
        self.n_points = n_points
        self.n_frames = n_frames
        self.mode = mode
        self.noise_std = noise_std
        self._rng = np.random.RandomState(rng_seed)

    def generate(self) -> Dict:
        """
        生成合成数据，格式与 D4RTLoader 输出兼容。

        Returns
        -------
        {
            "tracks"      : [N, T, 3],
            "visibility"  : [N, T],
            "camera_poses": None,
        }
        """
        N, T = self.n_points, self.n_frames
        tracks = np.zeros((N, T, 3), dtype=np.float32)

        for i in range(N):
            tracks[i] = self._gen_single_track(i, T)

        return {
            "tracks": tracks,
            "visibility": np.ones((N, T), dtype=bool),
            "camera_poses": None,
        }

    def _gen_single_track(self, idx: int, T: int) -> np.ndarray:
        """为第 idx 个点生成一条轨迹 [T, 3]."""
        rng = self._rng

        if self.mode == "spiral":
            return self._spiral(idx, T, rng)
        elif self.mode == "random":
            return self._random_walk(T, rng)
        else:  # mixed
            if idx % 2 == 0:
                return self._spiral(idx, T, rng)
            else:
                return self._random_walk(T, rng)

    @staticmethod
    def _spiral(
        idx: int,
        T: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """螺旋形轨迹"""
        t_arr = np.linspace(0, 2 * np.pi, T)
        r0 = rng.uniform(0.5, 2.0)
        omega = rng.uniform(0.5, 2.0) * (1 if idx % 2 == 0 else -1)
        vz = rng.uniform(-0.05, 0.05)
        x0, y0, z0 = rng.randn(3)

        x = x0 + r0 * np.cos(omega * t_arr)
        y = y0 + r0 * np.sin(omega * t_arr)
        z = z0 + vz * t_arr

        noise = rng.randn(T, 3) * 0.02
        track = np.stack([x, y, z], axis=-1) + noise
        return track.astype(np.float32)

    @staticmethod
    def _random_walk(
        T: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """带随机加速度的随机游走"""
        pos = rng.randn(3) * 2.0
        vel = rng.randn(3) * 0.1
        track = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            track[t] = pos.copy()
            acc = rng.randn(3) * 0.02
            vel = vel + acc
            vel = np.clip(vel, -0.3, 0.3)
            pos = pos + vel
        return track
