"""
dynamic_bridge.py
=================
DROID-W 与 dynamic_prediction 模块之间的集成桥接层。

主要功能：
1. 从 DepthVideo（运行时）或 video.npz（后处理）提取动态点轨迹
2. 将数据转换为 SlidingWindowPredictor 所需格式
3. 运行预测循环并可视化

集成方式：
    在 SLAM.terminate() 结束后调用
    DynamicBridge(cfg, save_dir).run_from_video_npz(video_npz_path)

或通过命令行：
    python run_dynamic_prediction.py --mode droidw --video_npz /path/video.npz
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np


class DynamicBridge:
    """
    DROID-W 动态预测集成桥接器。

    在 SLAM 完成（terminate() 之后）运行，从保存的 video.npz 中
    提取动态轨迹并执行预测 + 可视化。

    Parameters
    ----------
    cfg             : SLAM 配置字典
    save_dir        : SLAM 输出目录（包含 video.npz）
    window_size     : 滑动窗口大小（帧数）
    predict_steps   : 预测未来帧数
    uncer_thresh    : 不确定性阈值（高于此值的像素视为动态）
    enable_vis      : 是否启用 Open3D 可视化
    fps             : 可视化帧率
    """

    def __init__(
        self,
        cfg: dict,
        save_dir: str,
        window_size: int = 8,
        predict_steps: int = 2,
        uncer_thresh: float = 2.0,
        enable_vis: bool = True,
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
        """
        从保存的 video.npz 提取动态点并运行预测。

        Parameters
        ----------
        video_npz_path : 若为 None，自动从 save_dir/video.npz 加载

        Returns
        -------
        {"ADE": float, "FDE": float} 若未评估则返回 {}
        """
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

        # 选择可视化器
        visualizer = self._make_visualizer()

        # 运行预测循环
        results = self._prediction_loop(frames, pids_list, predictor, visualizer)

        if visualizer is not None:
            visualizer.destroy()

        return results

    def run_from_d4rt(self, d4rt_path: str) -> dict:
        """
        从 D4RT 输出的 .npz 文件运行预测。

        Parameters
        ----------
        d4rt_path : D4RT 输出路径

        Returns
        -------
        {} （当前不含评估，可扩展）
        """
        from dynamic_prediction.d4rt_bridge import D4RTLoader
        from dynamic_prediction.sliding_window import SlidingWindowPredictor

        print(f"[DynamicBridge] 加载 D4RT 数据：{d4rt_path}")
        data = D4RTLoader.load(d4rt_path)
        frames, pids_list = D4RTLoader.to_frame_sequence(data)

        predictor = SlidingWindowPredictor(
            window_size=self.window_size,
            predict_steps=self.predict_steps,
        )
        visualizer = self._make_visualizer()

        results = self._prediction_loop(frames, pids_list, predictor, visualizer)

        if visualizer is not None:
            visualizer.destroy()
        return results

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _make_visualizer(self):
        """初始化可视化器（若 enable_vis=False 则返回 None）。"""
        if not self.enable_vis:
            return None
        try:
            from dynamic_prediction.visualizer_4d import Visualizer4D
            return Visualizer4D(window_name="DROID-W Dynamic Prediction")
        except Exception as e:
            print(f"[DynamicBridge] 无法启动 Open3D 可视化：{e}")
            return None

    def _prediction_loop(
        self,
        frames: list,
        pids_list: list,
        predictor,
        visualizer,
    ) -> dict:
        """核心预测 + 可视化循环。"""
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

            # 帧率控制
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
                end="",
                flush=True,
            )

        print(f"\n[DynamicBridge] 预测循环完成。")
        return {}
