from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def read_tum_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    positions: list[list[float]] = []
    quats_xyzw: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            timestamps.append(float(parts[0]))
            positions.append([float(v) for v in parts[1:4]])
            quats_xyzw.append([float(v) for v in parts[4:8]])
    if not timestamps:
        raise ValueError(f"No TUM poses found in {path}")
    return np.asarray(timestamps), np.asarray(positions), normalize_quat(np.asarray(quats_xyzw))


def read_npz_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    if "timestamps" in data.files:
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    else:
        pose_key = "poses" if "poses" in data.files else "traj"
        timestamps = np.arange(len(data[pose_key]), dtype=np.float64)

    if "positions" in data.files:
        positions = np.asarray(data["positions"], dtype=np.float64)
        if "quats_xyzw" in data.files:
            quats = np.asarray(data["quats_xyzw"], dtype=np.float64)
        elif "orientations_quat_xyzw" in data.files:
            quats = np.asarray(data["orientations_quat_xyzw"], dtype=np.float64)
        elif "orientations_quat_wxyz" in data.files:
            q = np.asarray(data["orientations_quat_wxyz"], dtype=np.float64)
            quats = q[:, [1, 2, 3, 0]]
        else:
            raise ValueError(f"{path} has positions but no quaternion array")
        return timestamps, positions, normalize_quat(quats)

    pose_key = "poses" if "poses" in data.files else "traj"
    poses = np.asarray(data[pose_key], dtype=np.float64)
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        positions = poses[:, :3, 3]
        quats = rotmat_to_quat_xyzw(poses[:, :3, :3])
        return timestamps, positions, quats
    if poses.ndim == 2 and poses.shape[1] >= 7:
        positions = poses[:, :3]
        quats = poses[:, 3:7]
        return timestamps, positions, normalize_quat(quats)
    raise ValueError(f"Unsupported npz trajectory format in {path}; expected poses [N,4,4] or [N,7]")


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        return read_npz_trajectory(path)
    return read_tum_trajectory(path)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    r[:, 0, 1] = 2.0 * (xy - wz)
    r[:, 0, 2] = 2.0 * (xz + wy)
    r[:, 1, 0] = 2.0 * (xy + wz)
    r[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    r[:, 1, 2] = 2.0 * (yz - wx)
    r[:, 2, 0] = 2.0 * (xz - wy)
    r[:, 2, 1] = 2.0 * (yz + wx)
    r[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return r


def rotmat_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    q = np.empty((r.shape[0], 4), dtype=np.float64)
    for i, m in enumerate(r):
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            q[i] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        else:
            axis = int(np.argmax(np.diag(m)))
            if axis == 0:
                s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                q[i] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
            elif axis == 1:
                s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                q[i] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
            else:
                s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                q[i] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return normalize_quat(q)


def associate_by_timestamp(
    est_t: np.ndarray,
    gt_t: np.ndarray,
    max_diff: float,
) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[int, int]] = []
    j = 0
    for i, t in enumerate(est_t):
        while j + 1 < len(gt_t) and abs(gt_t[j + 1] - t) <= abs(gt_t[j] - t):
            j += 1
        if abs(gt_t[j] - t) <= max_diff:
            pairs.append((i, j))
    if len(pairs) < 2:
        raise ValueError(f"Only {len(pairs)} associated poses; increase --max-time-diff or check timestamps")
    est_idx, gt_idx = zip(*pairs)
    return np.asarray(est_idx), np.asarray(gt_idx)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    sfix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sfix[-1, -1] = -1.0
    rot = u @ sfix @ vt
    scale = 1.0
    if with_scale:
        var = np.mean(np.sum(src_c * src_c, axis=1))
        scale = float(np.trace(np.diag(d) @ sfix) / max(var, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def make_poses(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions
    return poses


def rotation_angle_deg(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def summarize_errors(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_rmse": float(math.sqrt(np.mean(values * values))),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def relative_pose_errors(est_poses: np.ndarray, gt_poses: np.ndarray, delta: int) -> tuple[np.ndarray, np.ndarray]:
    trans_errors: list[float] = []
    rot_errors: list[float] = []
    for i in range(0, len(est_poses) - delta):
        est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i + delta]
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        err = np.linalg.inv(gt_rel) @ est_rel
        trans_errors.append(float(np.linalg.norm(err[:3, 3])))
        rot_errors.append(float(rotation_angle_deg(err[:3, :3][None])[0]))
    return np.asarray(trans_errors), np.asarray(rot_errors)


def pose_auc(errors: np.ndarray, threshold: float) -> float:
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    errors = errors[np.isfinite(errors)]
    errors = errors[errors <= threshold]
    if errors.size == 0:
        return 0.0
    recall = np.arange(1, errors.size + 1, dtype=np.float64) / max(len(errors), 1)
    x = np.concatenate([[0.0], errors, [threshold]])
    y = np.concatenate([[0.0], recall, [recall[-1]]])
    return float(np.trapz(y, x) / threshold * 100.0)


def translation_direction_error_deg(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    est_norm = np.linalg.norm(est, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)
    valid = (est_norm > 1e-12) & (gt_norm > 1e-12)
    out = np.zeros(len(est), dtype=np.float64)
    cos = np.sum(est[valid] * gt[valid], axis=1) / (est_norm[valid] * gt_norm[valid])
    out[valid] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def evaluate_pose_metrics(
    est_path: Path,
    gt_path: Path,
    max_time_diff: float,
    rpe_delta: int,
    normalize_gt_length: bool,
    correct_scale: bool,
    associate_by_index: bool,
) -> dict[str, float | int | str]:
    est_t, est_p, est_q = read_trajectory(est_path)
    gt_t, gt_p, gt_q = read_trajectory(gt_path)
    if associate_by_index:
        n = min(len(est_t), len(gt_t))
        if n < 2:
            raise ValueError("Need at least two poses for index-based association")
        est_idx = np.arange(n)
        gt_idx = np.arange(n)
    else:
        est_idx, gt_idx = associate_by_timestamp(est_t, gt_t, max_time_diff)
    est_p, est_q, est_t = est_p[est_idx], est_q[est_idx], est_t[est_idx]
    gt_p, gt_q, gt_t = gt_p[gt_idx], gt_q[gt_idx], gt_t[gt_idx]

    if normalize_gt_length:
        length = float(np.sum(np.linalg.norm(np.diff(gt_p, axis=0), axis=1)))
        if length > 1e-12:
            gt_p = gt_p / length
            est_p = est_p / length

    scale, align_r, align_t = umeyama_alignment(est_p, gt_p, with_scale=correct_scale)
    est_p_aligned = scale * (est_p @ align_r.T) + align_t
    est_r_aligned = align_r[None] @ quat_to_rotmat(est_q)
    gt_r = quat_to_rotmat(gt_q)

    ape = np.linalg.norm(est_p_aligned - gt_p, axis=1)
    est_poses = make_poses(est_p_aligned, est_r_aligned)
    gt_poses = make_poses(gt_p, gt_r)
    rpe_t, rpe_r = relative_pose_errors(est_poses, gt_poses, rpe_delta)

    rel_est_t = est_poses[1:, :3, 3] - est_poses[:-1, :3, 3]
    rel_gt_t = gt_poses[1:, :3, 3] - gt_poses[:-1, :3, 3]
    rel_r_err = rotation_angle_deg(np.swapaxes(gt_r[:-1], -1, -2) @ gt_r[1:] @ np.swapaxes(est_r_aligned[1:], -1, -2) @ est_r_aligned[:-1])
    rel_t_err = translation_direction_error_deg(rel_est_t, rel_gt_t)
    auc_pose_error = np.maximum(rel_r_err, rel_t_err)

    metrics: dict[str, float | int | str] = {
        "est_path": str(est_path),
        "gt_path": str(gt_path),
        "num_est_poses": int(len(est_t)),
        "num_gt_poses": int(len(gt_t)),
        "num_associated_poses": int(len(ape)),
        "sim3_scale": float(scale),
        "sim3_correct_scale": int(correct_scale),
        "normalized_gt_length": int(normalize_gt_length),
        "associate_by_index": int(associate_by_index),
        "rpe_delta": int(rpe_delta),
    }
    metrics.update(summarize_errors(ape, "ate"))
    metrics.update(summarize_errors(rpe_t, "rpe_t"))
    metrics.update(summarize_errors(rpe_r, "rpe_r_deg"))
    for threshold in [5.0, 10.0, 20.0, 30.0]:
        metrics[f"pose_auc_at_{int(threshold)}"] = pose_auc(auc_pose_error, threshold)
    return metrics


def write_outputs(metrics: dict[str, float | int | str], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with out_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])
    report = f"""# Paper Pose Metrics

- Estimated trajectory: `{metrics['est_path']}`
- Ground-truth trajectory: `{metrics['gt_path']}`
- Associated poses: {metrics['num_associated_poses']}
- Sim(3) scale: {float(metrics['sim3_scale']):.8f}

| Metric | Value |
|---|---:|
| ATE RMSE | {float(metrics['ate_rmse']):.6f} |
| RPE-T RMSE | {float(metrics['rpe_t_rmse']):.6f} |
| RPE-R RMSE deg | {float(metrics['rpe_r_deg_rmse']):.6f} |
| Pose AUC@30 | {float(metrics['pose_auc_at_30']):.3f} |
"""
    out_prefix.with_suffix(".report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DROID-W/D4RT-style pose metrics: Sim(3)-aligned ATE, RPE-T, RPE-R, and Pose AUC@30."
    )
    parser.add_argument("--est", required=True, help="Estimated trajectory in TUM txt or npz format.")
    parser.add_argument("--gt", required=True, help="Ground-truth trajectory in TUM txt or npz format.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for .json/.csv/.report.md.")
    parser.add_argument("--max-time-diff", type=float, default=0.02, help="Maximum timestamp association difference.")
    parser.add_argument("--rpe-delta", type=int, default=1, help="Frame delta for relative pose error.")
    parser.add_argument("--no-scale", action="store_true", help="Use SE(3) alignment instead of Sim(3).")
    parser.add_argument("--normalize-gt-length", action="store_true", help="DROID-W/DyCheck-style trajectory length normalization.")
    parser.add_argument("--associate-by-index", action="store_true", help="Pair poses by order instead of timestamp.")
    args = parser.parse_args()

    metrics = evaluate_pose_metrics(
        est_path=Path(args.est),
        gt_path=Path(args.gt),
        max_time_diff=args.max_time_diff,
        rpe_delta=args.rpe_delta,
        normalize_gt_length=args.normalize_gt_length,
        correct_scale=not args.no_scale,
        associate_by_index=args.associate_by_index,
    )
    write_outputs(metrics, Path(args.out_prefix))
    print(f"Saved pose metrics to {Path(args.out_prefix).with_suffix('.json')}")


if __name__ == "__main__":
    main()
