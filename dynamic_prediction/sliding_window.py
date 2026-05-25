"""
sliding_window.py
=================
滑动窗口管理器（SlidingWindowPredictor）

负责：
1. 维护过去 window_size 帧的 3D 点云数据
2. 为每个 tracked 点维护一个 KalmanPointPredictor
3. 每帧更新后向外暴露：当前点云 / 历史轨迹 / 未来预测

运行流程（近实时）：
    t=1..N-1    : add_frame() 积累数据
    t>=N        : ready() → True，predict() 输出预测，滑动向前
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from dynamic_prediction.point_predictor import KalmanPointPredictor


class SlidingWindowPredictor:
    """
    逐点滑动窗口预测器。

    Parameters
    ----------
    window_size       : 使用多少帧历史来维护 Kalman 状态
    predict_steps     : 预测未来多少帧（通常 1-2）
    process_noise     : Kalman 过程噪声
    measurement_noise : Kalman 测量噪声
    """

    def __init__(
        self,
        window_size: int = 8,
        predict_steps: int = 2,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ) -> None:
        self.window_size = window_size
        self.predict_steps = predict_steps
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # point_id  →  KalmanPointPredictor
        self._predictors: Dict[int, KalmanPointPredictor] = {}

        # 滑动窗口：每个元素是 (frame_idx, {pid: np.ndarray[3]})
        self._window: deque = deque(maxlen=window_size)

        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def add_frame(
        self,
        points: np.ndarray,
        point_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        添加一帧 3D 动态点。

        Parameters
        ----------
        points    : [N, 3]  当前帧所有动态点的 3D 世界坐标
        point_ids : [N]     每个点的全局 ID（用于跨帧匹配）
                            若为 None，默认使用 0..N-1
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points 应为 [N, 3]，实际得到 {points.shape}")

        N = points.shape[0]
        if point_ids is None:
            point_ids = np.arange(N, dtype=int)
        else:
            point_ids = np.asarray(point_ids, dtype=int)

        frame_dict: Dict[int, np.ndarray] = {}
        for pid, pos in zip(point_ids, points):
            pid = int(pid)
            frame_dict[pid] = pos.copy()

            # 懒初始化每个点的 Kalman 预测器
            if pid not in self._predictors:
                self._predictors[pid] = KalmanPointPredictor(
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            self._predictors[pid].add(pos)

        self._window.append((self._frame_count, frame_dict))
        self._frame_count += 1

    def predict(self) -> Dict[int, List[np.ndarray]]:
        """
        对当前滑动窗口内仍活跃的点返回未来 predict_steps 帧的预测位置。

        注意：这里不再对 self._predictors 中的所有历史点预测。
        旧点如果已经离开滑动窗口，继续外推会把过期动态点写回
        后续帧，导致反馈 mask 过大、污染 BA。

        Returns
        -------
        {point_id: [pred_t+1, pred_t+2, ...]}
        每个 pred 是 np.ndarray shape [3]
        """
        if not self.ready():
            return {}

        active_ids = self.get_active_point_ids(current_only=True)
        return self.predict_for_ids(active_ids)

    def predict_for_ids(self, point_ids) -> Dict[int, List[np.ndarray]]:
        """只对指定 point ids 做未来预测。"""
        preds: Dict[int, List[np.ndarray]] = {}
        for pid in point_ids:
            pid = int(pid)
            kf = self._predictors.get(pid)
            if kf is None or kf.n_observations < 2:
                continue
            preds[pid] = kf.predict(self.predict_steps)
        return preds

    def get_history(self) -> Dict[int, List[np.ndarray]]:
        """
        返回当前滑动窗口内每个点的历史轨迹。

        Returns
        -------
        {point_id: [[x,y,z], [x,y,z], ...]}  （时间升序）
        """
        history: Dict[int, List[np.ndarray]] = {}
        for _, frame_dict in self._window:
            for pid, pos in frame_dict.items():
                history.setdefault(pid, []).append(pos.copy())
        return history

    def get_current_points(self) -> np.ndarray:
        """
        返回最新一帧的点云（[N, 3]）。
        """
        if len(self._window) == 0:
            return np.zeros((0, 3), dtype=float)
        _, frame_dict = self._window[-1]
        if not frame_dict:
            return np.zeros((0, 3), dtype=float)
        return np.array(list(frame_dict.values()), dtype=float)

    def get_current_point_ids(self) -> np.ndarray:
        """返回最新一帧中出现的 point ids。"""
        if len(self._window) == 0:
            return np.zeros((0,), dtype=int)
        _, frame_dict = self._window[-1]
        return np.array(list(frame_dict.keys()), dtype=int)

    def get_active_point_ids(self, current_only: bool = True) -> np.ndarray:
        """
        返回滑动窗口里的活跃 point ids。

        current_only=True 时只返回最新一帧出现的点，适合在线反馈；
        False 时返回整个窗口中出现过的点，适合画历史轨迹。
        """
        if current_only:
            return self.get_current_point_ids()

        ids = set()
        for _, frame_dict in self._window:
            ids.update(frame_dict.keys())
        return np.array(sorted(ids), dtype=int)

    def ready(self) -> bool:
        """当已积累 window_size 帧时返回 True，可以开始预测。"""
        return len(self._window) >= self.window_size

    def prune_inactive(self) -> None:
        """删除已经不在滑动窗口中的 Kalman 状态，避免历史点无限累积。"""
        active = set(self.get_active_point_ids(current_only=False).tolist())
        stale = [pid for pid in self._predictors if pid not in active]
        for pid in stale:
            del self._predictors[pid]

    def reset(self) -> None:
        """重置所有状态（用于重新运行）。"""
        self._predictors.clear()
        self._window.clear()
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def n_tracked_points(self) -> int:
        return len(self._predictors)
