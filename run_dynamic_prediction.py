"""
run_dynamic_prediction.py
=========================
动态点运动预测系统主运行脚本

支持两种模式：
  demo    —— 使用合成数据，无需任何外部文件，立即可运行
  real    —— 加载 D4RT .npz 输出文件进行真实预测
  droidw  —— 加载 DROID-W video.npz，从不确定性图提取动态点

用法示例：

  # 合成数据演示
  python run_dynamic_prediction.py --mode demo --n_points 80 --n_frames 60

  # D4RT 真实数据
  python run_dynamic_prediction.py --mode real --d4rt_path /path/to/d4rt_output.npz

  # DROID-W 后处理
  python run_dynamic_prediction.py --mode droidw --video_npz /path/to/video.npz

可选参数：
  --window_size      滑动窗口大小（默认 8）
  --predict_steps    预测未来帧数（默认 2）
  --fps              目标帧率（默认 10.0，0=不限速）
  --no_vis           禁用可视化（只评估）
  --save_frames DIR  将每帧截图保存到指定目录
  --eval             在 demo 模式下计算 ADE/FDE
"""

from __future__ import annotations

import argparse
import time
import os

import numpy as np


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_visualizer(args):
    """根据参数选择并实例化可视化器。"""
    if args.no_vis:
        return None

    try:
        from dynamic_prediction.visualizer_4d import Visualizer4D
        return Visualizer4D(
            window_name="Dynamic Prediction 4D",
            width=1280,
            height=720,
        )
    except ImportError:
        print("[WARN] open3d 未安装，回退到 Matplotlib 后备可视化器")
        from dynamic_prediction.visualizer_4d import Visualizer4DMatplotlib
        save_dir = args.save_frames or "vis_frames"
        return Visualizer4DMatplotlib(output_dir=save_dir)


def _frame_delay(fps: float, t_start: float) -> None:
    """按指定帧率限速。"""
    if fps <= 0:
        return
    elapsed = time.time() - t_start
    wait = 1.0 / fps - elapsed
    if wait > 0:
        time.sleep(wait)


# ---------------------------------------------------------------------------
# 主循环（所有模式共用）
# ---------------------------------------------------------------------------

def run_prediction_loop(
    frames: list,
    point_ids_list: list,
    gt_frames: list | None,
    predictor,
    visualizer,
    args,
) -> dict:
    """
    主预测循环。

    Parameters
    ----------
    frames          : list[np.ndarray [N_t, 3]]
    point_ids_list  : list[np.ndarray [N_t]]
    gt_frames       : list[np.ndarray [N_t, 3]] 或 None（用于评估）
    predictor       : SlidingWindowPredictor
    visualizer      : Visualizer4D 或 None
    args            : argparse.Namespace

    Returns
    -------
    evaluation_results: dict（若 args.eval 为 True）
    """
    all_preds: list = []
    all_gts: list = []

    for t, (pts, pids) in enumerate(zip(frames, point_ids_list)):
        t_start = time.time()

        # 1. 喂给滑动窗口
        predictor.add_frame(pts, pids)

        # 2. 若窗口已满，进行预测
        predictions: dict = {}
        if predictor.ready():
            predictions = predictor.predict()

            # 收集用于评估
            if args.eval and gt_frames is not None:
                future_idx = t + 1
                if future_idx < len(gt_frames):
                    gt_pts = gt_frames[future_idx]   # 下一帧真值
                    gt_pids = point_ids_list[future_idx] if future_idx < len(point_ids_list) else np.arange(len(gt_pts))
                    gt_traj = {int(pid): [pos] for pid, pos in zip(gt_pids, gt_pts)}
                    # 仅取第 1 步预测比较
                    pred_traj_step1 = {
                        pid: [p[0]] for pid, p in predictions.items() if len(p) > 0
                    }
                    all_preds.append(pred_traj_step1)
                    all_gts.append(gt_traj)

        # 3. 可视化
        if visualizer is not None:
            current_pts = predictor.get_current_points()
            history = predictor.get_history()
            alive = visualizer.update(current_pts, history, predictions)
            if not alive:
                print(f"\n[INFO] 窗口已关闭，停止于第 {t} 帧")
                break

            # 可选截图
            if args.save_frames:
                os.makedirs(args.save_frames, exist_ok=True)
                vis_path = os.path.join(args.save_frames, f"frame_{t:05d}.png")
                try:
                    visualizer.capture_screenshot(vis_path)
                except Exception:
                    pass

        # 4. 帧率控制
        _frame_delay(args.fps, t_start)

        # 进度打印
        status = "PRED" if predictor.ready() else f"WARM({t+1}/{args.window_size})"
        n_pts = pts.shape[0] if pts is not None else 0
        print(
            f"\r  帧 {t+1:04d}/{len(frames)} | {status}"
            f" | 动态点: {n_pts:4d}"
            f" | tracked: {predictor.n_tracked_points:4d}",
            end="",
            flush=True,
        )

    print()  # 换行

    # 5. 评估
    eval_results = {}
    if args.eval and all_preds:
        from dynamic_prediction.evaluator import compute_ade, compute_fde, format_eval_report
        # 合并所有帧的预测/真值
        merged_pred: dict = {}
        merged_gt: dict = {}
        for pred_t, gt_t in zip(all_preds, all_gts):
            merged_pred.update(pred_t)
            merged_gt.update(gt_t)
        eval_results["ADE"] = compute_ade(merged_pred, merged_gt)
        eval_results["FDE"] = compute_fde(merged_pred, merged_gt)
        print(format_eval_report(eval_results))

    return eval_results


# ---------------------------------------------------------------------------
# Demo 模式
# ---------------------------------------------------------------------------

def run_demo(args) -> None:
    from dynamic_prediction.d4rt_bridge import SyntheticDataGenerator, D4RTLoader
    from dynamic_prediction.sliding_window import SlidingWindowPredictor

    print(f"[Demo] 生成 {args.n_points} 个点 × {args.n_frames} 帧合成数据 ...")
    gen = SyntheticDataGenerator(
        n_points=args.n_points,
        n_frames=args.n_frames,
        mode="mixed",
    )
    data = gen.generate()

    # 将 [N, T, 3] 转为 frame 序列
    frames, pids_list = D4RTLoader.to_frame_sequence(data)

    # 后半段作为"真值"用于评估（如果需要）
    gt_frames = frames  # demo 下 gt = tracks 本身（不适合真实评估，仅演示）

    predictor = SlidingWindowPredictor(
        window_size=args.window_size,
        predict_steps=args.predict_steps,
    )
    visualizer = _make_visualizer(args)

    print(f"[Demo] 开始运行，窗口大小={args.window_size}，预测步数={args.predict_steps}")
    run_prediction_loop(frames, pids_list, gt_frames, predictor, visualizer, args)

    if visualizer is not None:
        visualizer.destroy()
    print("[Demo] 完成。")


# ---------------------------------------------------------------------------
# Real 模式（D4RT .npz）
# ---------------------------------------------------------------------------

def run_real(args) -> None:
    from dynamic_prediction.d4rt_bridge import D4RTLoader
    from dynamic_prediction.sliding_window import SlidingWindowPredictor

    print(f"[Real] 加载 D4RT 数据：{args.d4rt_path}")
    data = D4RTLoader.load(args.d4rt_path)
    frames, pids_list = D4RTLoader.to_frame_sequence(data, visible_only=True)

    print(f"[Real] 共 {len(frames)} 帧，每帧约 {np.mean([len(f) for f in frames]):.0f} 个可见点")

    predictor = SlidingWindowPredictor(
        window_size=args.window_size,
        predict_steps=args.predict_steps,
    )
    visualizer = _make_visualizer(args)

    run_prediction_loop(frames, pids_list, None, predictor, visualizer, args)

    if visualizer is not None:
        visualizer.destroy()
    print("[Real] 完成。")


# ---------------------------------------------------------------------------
# DROID-W 模式（video.npz）
# ---------------------------------------------------------------------------

def run_droidw(args) -> None:
    from dynamic_prediction.d4rt_bridge import DROIDWBridge
    from dynamic_prediction.sliding_window import SlidingWindowPredictor

    print(f"[DROID-W] 加载 video.npz：{args.video_npz}")
    bridge = DROIDWBridge(uncer_thresh=args.uncer_thresh)
    dw_data = bridge.load(args.video_npz)
    frames, pids_list = bridge.extract_dynamic_points(dw_data)

    print(f"[DROID-W] 共 {len(frames)} 个关键帧")

    predictor = SlidingWindowPredictor(
        window_size=args.window_size,
        predict_steps=args.predict_steps,
    )
    visualizer = _make_visualizer(args)

    run_prediction_loop(frames, pids_list, None, predictor, visualizer, args)

    if visualizer is not None:
        visualizer.destroy()
    print("[DROID-W] 完成。")


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="动态点运动预测系统（D4RT + Kalman Filter）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["demo", "real", "droidw"], default="demo",
        help="运行模式"
    )

    # Demo 参数
    demo_group = parser.add_argument_group("Demo 模式参数")
    demo_group.add_argument("--n_points", type=int, default=80,
                            help="合成点数量")
    demo_group.add_argument("--n_frames", type=int, default=60,
                            help="合成帧数")

    # Real 参数
    real_group = parser.add_argument_group("Real 模式参数（D4RT）")
    real_group.add_argument("--d4rt_path", type=str, default=None,
                            help="D4RT 输出 .npz 文件路径")

    # DROID-W 参数
    dw_group = parser.add_argument_group("DROID-W 模式参数")
    dw_group.add_argument("--video_npz", type=str, default=None,
                          help="DROID-W 保存的 video.npz 路径")
    dw_group.add_argument("--uncer_thresh", type=float, default=2.0,
                          help="不确定性阈值（高于此值视为动态点）")

    # 通用参数
    parser.add_argument("--window_size", type=int, default=8,
                        help="滑动窗口大小（帧数）")
    parser.add_argument("--predict_steps", type=int, default=2,
                        help="预测未来帧数")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="目标帧率（0=不限速）")
    parser.add_argument("--no_vis", action="store_true",
                        help="禁用可视化")
    parser.add_argument("--save_frames", type=str, default=None,
                        help="截图保存目录（None=不保存）")
    parser.add_argument("--eval", action="store_true",
                        help="计算 ADE/FDE（demo 模式下自动使用合成真值）")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 50)
    print("  动态点预测系统  (D4RT + Kalman Filter)")
    print("=" * 50)
    print(f"  模式         : {args.mode}")
    print(f"  窗口大小     : {args.window_size}")
    print(f"  预测步数     : {args.predict_steps}")
    print(f"  帧率上限     : {args.fps if args.fps > 0 else '不限速'}")
    print("=" * 50)

    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "real":
        if args.d4rt_path is None:
            raise ValueError("--mode real 需要提供 --d4rt_path")
        run_real(args)
    elif args.mode == "droidw":
        if args.video_npz is None:
            raise ValueError("--mode droidw 需要提供 --video_npz")
        run_droidw(args)
