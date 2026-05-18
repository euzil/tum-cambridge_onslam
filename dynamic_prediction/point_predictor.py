"""
point_predictor.py
==================
逐点运动预测器

提供两种预测器：
1. ConstantVelocityPredictor  —— 匀速模型（验证用）
2. KalmanPointPredictor       —— Kalman Filter（推荐 MVP）
   状态向量：[x, y, z, vx, vy, vz]
"""

from __future__ import annotations

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
    _FILTERPY_AVAILABLE = True
except ImportError:
    _FILTERPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 匀速模型
# ---------------------------------------------------------------------------

class ConstantVelocityPredictor:
    """
    基于历史位置均值速度的匀速外推预测器。

    适用场景：快速验证、直线匀速运动。
    限制：突然加速或转弯时预测失效。
    """

    def __init__(self) -> None:
        self._history: list[np.ndarray] = []

    def add(self, position: np.ndarray) -> None:
        """
        添加当前帧的 3D 位置（[3]）。
        """
        self._history.append(np.asarray(position, dtype=float).copy())

    def predict(self, n_steps: int = 1) -> list[np.ndarray]:
        """
        预测未来 n_steps 帧的位置。

        Returns
        -------
        list of np.ndarray shape [3]，长度为 n_steps
        """
        if len(self._history) == 0:
            return [np.zeros(3)] * n_steps
        if len(self._history) == 1:
            return [self._history[-1].copy()] * n_steps

        positions = np.stack(self._history, axis=0)          # [T, 3]
        velocities = np.diff(positions, axis=0)               # [T-1, 3]
        avg_vel = velocities.mean(axis=0)                     # [3]
        last = self._history[-1].copy()

        return [last + avg_vel * (i + 1) for i in range(n_steps)]

    def reset(self) -> None:
        self._history.clear()

    @property
    def n_observations(self) -> int:
        return len(self._history)


# ---------------------------------------------------------------------------
# Kalman Filter 预测器
# ---------------------------------------------------------------------------

class KalmanPointPredictor:
    """
    基于 filterpy 的 6-DOF Kalman Filter 单点预测器。

    状态：[x, y, z, vx, vy, vz]
    观测：[x, y, z]
    运动模型：匀速（Constant Velocity），可扩展为匀加速。

    Parameters
    ----------
    process_noise      : 过程噪声强度（Q），越大越相信测量值
    measurement_noise  : 测量噪声强度（R），越大越平滑但滞后
    dt                 : 帧间时间步长（默认 1.0，即按帧计）
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        if not _FILTERPY_AVAILABLE:
            raise ImportError(
                "filterpy is required: pip install filterpy"
            )

        self._dt = dt
        self._kf = KalmanFilter(dim_x=6, dim_z=3)

        # ---- 状态转移矩阵 F（匀速模型）----
        self._kf.F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ], dtype=float)

        # ---- 观测矩阵 H（只观测位置）----
        self._kf.H = np.eye(3, 6, dtype=float)

        # ---- 观测噪声 R ----
        self._kf.R = np.eye(3, dtype=float) * measurement_noise

        # ---- 过程噪声 Q ----
        q = process_noise
        self._kf.Q = np.diag([q * 0.5, q * 0.5, q * 0.5, q, q, q])

        # ---- 初始协方差 P ----
        self._kf.P = np.eye(6, dtype=float) * 100.0

        self._initialized: bool = False
        self._n_obs: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def add(self, position: np.ndarray) -> None:
        """
        添加一次观测值（[3]），更新 Kalman Filter。
        首次调用时初始化状态，后续执行 predict → update。
        """
        z = np.asarray(position, dtype=float).reshape(3, 1)

        if not self._initialized:
            # 以第一个观测值初始化状态；速度初始为 0
            self._kf.x = np.array(
                [z[0, 0], z[1, 0], z[2, 0], 0.0, 0.0, 0.0],
                dtype=float,
            ).reshape(6, 1)
            self._initialized = True
        else:
            self._kf.predict()
            self._kf.update(z)

        self._n_obs += 1

    def predict(self, n_steps: int = 1) -> list[np.ndarray]:
        """
        向前预测 n_steps 帧的位置，不修改内部状态。

        Returns
        -------
        list of np.ndarray shape [3]，长度为 n_steps
        """
        if not self._initialized:
            return [np.zeros(3)] * n_steps

        # 临时保存当前状态
        x_saved = self._kf.x.copy()
        P_saved = self._kf.P.copy()

        preds: list[np.ndarray] = []
        for _ in range(n_steps):
            self._kf.predict()
            preds.append(self._kf.x[:3].flatten().copy())

        # 恢复状态（不污染后续真实更新）
        self._kf.x = x_saved
        self._kf.P = P_saved

        return preds

    def reset(self) -> None:
        """重置 Kalman Filter 至未初始化状态。"""
        self._kf.P = np.eye(6, dtype=float) * 100.0
        self._initialized = False
        self._n_obs = 0

    @property
    def n_observations(self) -> int:
        return self._n_obs

    @property
    def state(self) -> np.ndarray:
        """当前状态估计 [x, y, z, vx, vy, vz]"""
        if not self._initialized:
            return np.zeros(6)
        return self._kf.x.flatten().copy()

    @property
    def velocity(self) -> np.ndarray:
        """当前速度估计 [vx, vy, vz]"""
        return self.state[3:]
