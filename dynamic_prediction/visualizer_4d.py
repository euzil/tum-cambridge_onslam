"""
visualizer_4d.py
================
基于 Open3D 的 4D 动态点云实时可视化器

颜色约定：
  白色  —— 当前帧动态点
  蓝色  —— 历史轨迹线段
  红色  —— 预测轨迹线段

用法示例：
    vis = Visualizer4D()
    for frame in ...:
        vis.update(current_pts, history, predictions)
    vis.destroy()
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_lineset(
    trajectories: Dict[int, List[np.ndarray]],
    color: List[float],
    prefix_point: Dict[int, np.ndarray] | None = None,
) -> "o3d.geometry.LineSet":
    """
    将多条轨迹（{pid: [pt0, pt1, ...]}）构建为 Open3D LineSet。

    prefix_point: 若给定，每条轨迹在 trajectories[pid] 之前再加一个
                  起始点（用于将历史末尾连到预测头部）。
    """
    all_points: List[np.ndarray] = []
    all_lines: List[List[int]] = []
    all_colors: List[List[float]] = []
    offset = 0

    for pid, traj in trajectories.items():
        chain = traj
        if prefix_point is not None and pid in prefix_point:
            chain = [prefix_point[pid]] + list(traj)

        n = len(chain)
        if n < 2:
            continue

        for pt in chain:
            all_points.append(np.asarray(pt, dtype=float))
        for i in range(n - 1):
            all_lines.append([offset + i, offset + i + 1])
            all_colors.append(color)
        offset += n

    ls = o3d.geometry.LineSet()
    if all_points:
        ls.points = o3d.utility.Vector3dVector(np.array(all_points))
        ls.lines = o3d.utility.Vector2iVector(np.array(all_lines))
        ls.colors = o3d.utility.Vector3dVector(np.array(all_colors))
    return ls


# ---------------------------------------------------------------------------
# 主可视化类
# ---------------------------------------------------------------------------

class Visualizer4D:
    """
    Open3D 交互式 4D 可视化器。

    Parameters
    ----------
    window_name : 窗口标题
    width, height : 窗口分辨率
    point_size : 点的渲染大小
    bg_color : 背景颜色 RGB（0~1）
    """

    def __init__(
        self,
        window_name: str = "Dynamic Prediction 4D",
        width: int = 1280,
        height: int = 720,
        point_size: float = 3.0,
        bg_color: List[float] = None,
    ) -> None:
        if not _O3D_AVAILABLE:
            raise ImportError("open3d is required: pip install open3d")

        if bg_color is None:
            bg_color = [0.05, 0.05, 0.05]

        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name, width=width, height=height)

        opt = self._vis.get_render_option()
        opt.background_color = np.array(bg_color)
        opt.point_size = point_size
        opt.line_width = 2.0

        # 三组几何体：当前点云 / 历史轨迹 / 预测轨迹
        self._pcd_current = o3d.geometry.PointCloud()
        self._lines_history = o3d.geometry.LineSet()
        self._lines_pred = o3d.geometry.LineSet()

        self._vis.add_geometry(self._pcd_current)
        self._vis.add_geometry(self._lines_history)
        self._vis.add_geometry(self._lines_pred)

        self._first_frame = True

    # ------------------------------------------------------------------
    # 主更新接口
    # ------------------------------------------------------------------

    def update(
        self,
        current_points: np.ndarray,
        history: Dict[int, List[np.ndarray]],
        predictions: Dict[int, List[np.ndarray]],
    ) -> bool:
        """
        刷新可视化内容。

        Parameters
        ----------
        current_points : [N, 3] 当前帧动态点（白色）
        history        : {pid: [[x,y,z],...]} 历史轨迹（蓝色）
        predictions    : {pid: [[x,y,z],...]} 预测轨迹（红色）

        Returns
        -------
        bool: False 表示窗口已被用户关闭，调用方应退出循环
        """
        # ---------- 当前点云（白色）----------
        if current_points is not None and len(current_points) > 0:
            pts = np.asarray(current_points, dtype=float)
            self._pcd_current.points = o3d.utility.Vector3dVector(pts)
            self._pcd_current.paint_uniform_color([1.0, 1.0, 1.0])
        else:
            self._pcd_current.clear()

        # ---------- 历史轨迹（蓝色）----------
        new_hist = _build_lineset(history, color=[0.1, 0.5, 1.0])
        self._lines_history.points = new_hist.points
        self._lines_history.lines = new_hist.lines
        self._lines_history.colors = new_hist.colors

        # ---------- 预测轨迹（红色，从历史末尾出发）----------
        # 取每个点的历史末尾作为 prefix
        last_history: Dict[int, np.ndarray] = {
            pid: traj[-1] for pid, traj in history.items() if len(traj) > 0
        }
        new_pred = _build_lineset(
            predictions,
            color=[1.0, 0.2, 0.2],
            prefix_point=last_history,
        )
        self._lines_pred.points = new_pred.points
        self._lines_pred.lines = new_pred.lines
        self._lines_pred.colors = new_pred.colors

        # ---------- 推送到 GPU / 渲染 ----------
        self._vis.update_geometry(self._pcd_current)
        self._vis.update_geometry(self._lines_history)
        self._vis.update_geometry(self._lines_pred)

        if self._first_frame:
            self._vis.reset_view_point(True)
            self._first_frame = False

        # poll_events 返回 False 说明窗口被关闭
        if not self._vis.poll_events():
            return False
        self._vis.update_renderer()
        return True

    # ------------------------------------------------------------------
    # 截图 / 销毁
    # ------------------------------------------------------------------

    def capture_screenshot(self, path: str) -> None:
        """保存当前帧截图到 path（PNG）。"""
        self._vis.capture_screen_image(path)

    def destroy(self) -> None:
        """关闭并销毁 Open3D 窗口。"""
        self._vis.destroy_window()


# ---------------------------------------------------------------------------
# Matplotlib 后备实现（无 Open3D 时使用）
# ---------------------------------------------------------------------------

class Visualizer4DMatplotlib:
    """
    Matplotlib 后备可视化器（无 Open3D 时使用，功能有限）。
    每次调用 update() 会保存一张 PNG 到 output_dir。
    """

    def __init__(self, output_dir: str = "vis_frames") -> None:
        import os
        import matplotlib
        matplotlib.use("Agg")
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._frame_idx = 0

    def update(
        self,
        current_points: np.ndarray,
        history: Dict[int, List[np.ndarray]],
        predictions: Dict[int, List[np.ndarray]],
    ) -> bool:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.tick_params(colors="white")

        # 当前点（白色）
        if current_points is not None and len(current_points) > 0:
            pts = np.asarray(current_points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="white", s=3)

        # 历史轨迹（蓝色）
        for pid, traj in history.items():
            if len(traj) < 2:
                continue
            traj_arr = np.array(traj)
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
                    color=(0.1, 0.5, 1.0), linewidth=1)

        # 预测（红色）
        for pid, preds in predictions.items():
            if pid not in history or not history[pid]:
                continue
            chain = [history[pid][-1]] + list(preds)
            chain_arr = np.array(chain)
            ax.plot(chain_arr[:, 0], chain_arr[:, 1], chain_arr[:, 2],
                    color=(1.0, 0.2, 0.2), linewidth=1, linestyle="--")

        out_path = f"{self._output_dir}/frame_{self._frame_idx:05d}.png"
        plt.savefig(out_path, dpi=80, bbox_inches="tight",
                    facecolor="black")
        plt.close(fig)
        self._frame_idx += 1
        return True

    def capture_screenshot(self, path: str) -> None:
        pass  # 已在 update 中自动保存

    def destroy(self) -> None:
        pass
