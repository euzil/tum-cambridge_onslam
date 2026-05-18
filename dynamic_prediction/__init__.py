"""
dynamic_prediction
==================
动态点运动预测模块（基于 D4RT + Kalman Filter）

核心模块：
  - point_predictor:   匀速模型 / Kalman Filter 逐点预测
  - sliding_window:    滑动窗口管理器（近实时预测流水线）
  - visualizer_4d:     Open3D 4D 可视化（当前/历史/预测）
  - d4rt_bridge:       D4RT / DROID-W 数据加载接口
  - evaluator:         ADE / FDE 评估工具
"""

from dynamic_prediction.point_predictor import ConstantVelocityPredictor, KalmanPointPredictor
from dynamic_prediction.sliding_window import SlidingWindowPredictor
from dynamic_prediction.evaluator import compute_ade, compute_fde

__all__ = [
    "ConstantVelocityPredictor",
    "KalmanPointPredictor",
    "SlidingWindowPredictor",
    "compute_ade",
    "compute_fde",
]
