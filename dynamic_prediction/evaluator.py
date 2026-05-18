"""
evaluator.py
============
ADE / FDE 评估工具

ADE (Average Displacement Error):
    预测轨迹上每一步与真值的平均 L2 距离，再对所有点取均值

FDE (Final Displacement Error):
    仅看预测末尾帧与真值末尾帧的 L2 距离，对所有点取均值
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_ade(
    pred_trajs: Dict[int, List[np.ndarray]],
    gt_trajs: Dict[int, List[np.ndarray]],
) -> float:
    """
    计算 Average Displacement Error（ADE）。

    Parameters
    ----------
    pred_trajs : {pid: [pred_t+1, pred_t+2, ...]}  每个 pred 为 [3]
    gt_trajs   : {pid: [gt_t+1,  gt_t+2, ...]}     同上

    Returns
    -------
    float : ADE（单位与坐标单位相同，通常为米）
    """
    errors: List[float] = []
    for pid in pred_trajs:
        if pid not in gt_trajs:
            continue
        preds = np.array(pred_trajs[pid])   # [K, 3]
        gts = np.array(gt_trajs[pid])       # [K, 3]
        T = min(len(preds), len(gts))
        if T == 0:
            continue
        per_step = np.linalg.norm(preds[:T] - gts[:T], axis=1)  # [T]
        errors.append(float(per_step.mean()))

    return float(np.mean(errors)) if errors else 0.0


def compute_fde(
    pred_trajs: Dict[int, List[np.ndarray]],
    gt_trajs: Dict[int, List[np.ndarray]],
) -> float:
    """
    计算 Final Displacement Error（FDE）。

    Parameters
    ----------
    pred_trajs, gt_trajs : 同 compute_ade

    Returns
    -------
    float : FDE
    """
    errors: List[float] = []
    for pid in pred_trajs:
        if pid not in gt_trajs:
            continue
        preds = pred_trajs[pid]
        gts = gt_trajs[pid]
        T = min(len(preds), len(gts))
        if T == 0:
            continue
        err = np.linalg.norm(
            np.array(preds[T - 1]) - np.array(gts[T - 1])
        )
        errors.append(float(err))

    return float(np.mean(errors)) if errors else 0.0


def evaluate_sliding_window(
    pred_trajs: Dict[int, List[np.ndarray]],
    gt_trajs: Dict[int, List[np.ndarray]],
    predict_steps: int = 2,
) -> Dict[str, float]:
    """
    同时计算 ADE 和 FDE，并按预测步骤细分。

    Parameters
    ----------
    pred_trajs    : {pid: [pred_t+1, ..., pred_t+K]}
    gt_trajs      : {pid: [gt_t+1,  ..., gt_t+K]}
    predict_steps : K

    Returns
    -------
    {
        "ADE":          float,
        "FDE":          float,
        "step_ADE_1":   float,   # 第 1 步平均误差
        "step_ADE_2":   float,   # 第 2 步平均误差（如有）
        ...
    }
    """
    results = {
        "ADE": compute_ade(pred_trajs, gt_trajs),
        "FDE": compute_fde(pred_trajs, gt_trajs),
    }

    # 逐步误差
    for step in range(1, predict_steps + 1):
        step_pred = {
            pid: [traj[step - 1]] for pid, traj in pred_trajs.items()
            if len(traj) >= step
        }
        step_gt = {
            pid: [traj[step - 1]] for pid, traj in gt_trajs.items()
            if len(traj) >= step
        }
        results[f"step_ADE_{step}"] = compute_ade(step_pred, step_gt)

    return results


def format_eval_report(results: Dict[str, float]) -> str:
    """格式化评估结果为可读字符串。"""
    lines = ["=" * 40, "  动态预测评估结果", "=" * 40]
    for key, val in sorted(results.items()):
        lines.append(f"  {key:<20s}: {val:.4f} m")
    lines.append("=" * 40)
    return "\n".join(lines)
