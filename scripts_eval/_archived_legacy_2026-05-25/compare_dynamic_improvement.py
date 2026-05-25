from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rgb_timestamps(dataset_dir: Path) -> list[float]:
    path = dataset_dir / "rgb.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing rgb.txt: {path}")

    timestamps: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            timestamps.append(float(parts[0]))
    return timestamps


def read_groundtruth(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_dir / "groundtruth.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing groundtruth.txt: {path}")

    ts, xyz = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts.append(float(parts[0]))
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(ts, dtype=float), np.asarray(xyz, dtype=float)


def load_kf_trajectory(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    video_path = run_dir / "video.npz"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video.npz: {video_path}")

    data = np.load(video_path)
    poses = np.asarray(data["poses"], dtype=float)
    timestamps = np.asarray(data["timestamps"], dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses [T,4,4] in {video_path}, got {poses.shape}")
    return timestamps, poses[:, :3, 3]


def match_gt_for_kfs(
    kf_indices: np.ndarray,
    rgb_ts: list[float],
    gt_ts: np.ndarray,
    gt_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matched_ts, matched_xyz = [], []
    for idx in kf_indices.astype(int):
        if idx < 0 or idx >= len(rgb_ts):
            continue
        t = rgb_ts[idx]
        j = int(np.argmin(np.abs(gt_ts - t)))
        matched_ts.append(float(idx))
        matched_xyz.append(gt_xyz[j])
    return np.asarray(matched_ts, dtype=float), np.asarray(matched_xyz, dtype=float)


def intersect_by_timestamp(
    base_ts: np.ndarray,
    base_xyz: np.ndarray,
    dyn_ts: np.ndarray,
    dyn_xyz: np.ndarray,
    rgb_ts: list[float],
    gt_ts: np.ndarray,
    gt_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_map = {int(t): p for t, p in zip(base_ts.astype(int), base_xyz)}
    dyn_map = {int(t): p for t, p in zip(dyn_ts.astype(int), dyn_xyz)}
    common = np.asarray(sorted(set(base_map) & set(dyn_map)), dtype=int)
    if len(common) == 0:
        raise ValueError("No common keyframe timestamps between baseline and dynamic runs")

    gt_common_ts, gt_common_xyz = match_gt_for_kfs(common, rgb_ts, gt_ts, gt_xyz)
    valid_common = gt_common_ts.astype(int)
    base_common = np.asarray([base_map[int(t)] for t in valid_common], dtype=float)
    dyn_common = np.asarray([dyn_map[int(t)] for t in valid_common], dtype=float)
    return valid_common.astype(float), base_common, dyn_common, gt_common_xyz


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Align src to dst with similarity transform: dst ~= scale * R @ src + t."""
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("Need at least 3 paired points for Sim(3) alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(s * d) / max(var_src, 1e-12))
    t = mu_dst - scale * (r @ mu_src)
    return r, scale, t


def apply_alignment(xyz: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return (scale * (r @ xyz.T)).T + t[None, :]


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors * errors)))


def read_metric_rmse(run_dir: Path) -> float | None:
    path = run_dir / "traj" / "metrics_kf_traj.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"'rmse':\s*([0-9.eE+-]+)", text)
    return float(m.group(1)) if m else None


def read_feedback(run_dir: Path) -> dict[str, np.ndarray] | None:
    path = run_dir / "dynamic_feedback_stats.csv"
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    return {
        "kf_idx": np.asarray([float(r["kf_idx"]) for r in rows]),
        "coverage": np.asarray([float(r["coverage"]) for r in rows]),
        "applied": np.asarray([float(r["applied"]) for r in rows]),
    }


def save_error_csv(
    path: Path,
    frame_idx: np.ndarray,
    base_err: np.ndarray,
    dyn_err: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "baseline_error_m", "dynamic_error_m", "improvement_m"])
        for t, b, d in zip(frame_idx, base_err, dyn_err):
            writer.writerow([int(t), f"{b:.8f}", f"{d:.8f}", f"{(b - d):.8f}"])


def make_plot(
    out_png: Path,
    frame_idx: np.ndarray,
    gt: np.ndarray,
    base_aligned: np.ndarray,
    dyn_aligned: np.ndarray,
    base_err: np.ndarray,
    dyn_err: np.ndarray,
    feedback: dict[str, np.ndarray] | None,
    base_metric_rmse: float | None,
    dyn_metric_rmse: float | None,
) -> None:
    improvement = base_err - dyn_err
    positive = improvement > 0
    frac_better = float(np.mean(positive)) if len(positive) else 0.0
    mean_gain = float(np.mean(improvement))
    rmse_base = rmse(base_err)
    rmse_dyn = rmse(dyn_err)
    rel_gain = (rmse_base - rmse_dyn) / max(rmse_base, 1e-12) * 100.0

    fig = plt.figure(figsize=(15, 10), dpi=140)
    gs = fig.add_gridspec(2, 3)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1:])
    ax_gain = fig.add_subplot(gs[1, 0:2])
    ax_bar = fig.add_subplot(gs[1, 2])

    ax_traj.plot(gt[:, 0], gt[:, 1], color="black", lw=2, label="GT")
    ax_traj.plot(base_aligned[:, 0], base_aligned[:, 1], color="#d95f02", lw=1.4, label="Baseline")
    ax_traj.plot(dyn_aligned[:, 0], dyn_aligned[:, 1], color="#1b9e77", lw=1.4, label="Dynamic")
    ax_traj.set_title("Aligned Keyframe Trajectory (XY)")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(alpha=0.25)
    ax_traj.legend()

    ax_err.plot(frame_idx, base_err, color="#d95f02", lw=1.2, label=f"Baseline RMSE {rmse_base:.4f} m")
    ax_err.plot(frame_idx, dyn_err, color="#1b9e77", lw=1.2, label=f"Dynamic RMSE {rmse_dyn:.4f} m")
    ax_err.fill_between(frame_idx, dyn_err, base_err, where=positive, color="#1b9e77", alpha=0.18, label="Dynamic better")
    ax_err.fill_between(frame_idx, dyn_err, base_err, where=~positive, color="#d95f02", alpha=0.12, label="Dynamic worse")
    ax_err.set_title("Per-keyframe Translation Error")
    ax_err.set_xlabel("dataset frame index")
    ax_err.set_ylabel("ATE after Sim(3) alignment [m]")
    ax_err.grid(alpha=0.25)
    ax_err.legend(loc="upper right")

    ax_gain.axhline(0.0, color="black", lw=0.8)
    colors = np.where(positive, "#1b9e77", "#d95f02")
    ax_gain.bar(frame_idx, improvement, width=2.0, color=colors, alpha=0.85)
    ax_gain.set_title("Improvement per Keyframe: baseline error - dynamic error")
    ax_gain.set_xlabel("dataset frame index")
    ax_gain.set_ylabel("positive is better [m]")
    ax_gain.grid(alpha=0.25, axis="y")

    if feedback is not None:
        ax_cov = ax_gain.twinx()
        ax_cov.plot(feedback["kf_idx"], feedback["coverage"], color="#7570b3", lw=1.0, alpha=0.65, label="feedback coverage")
        ax_cov.set_ylabel("feedback coverage")
        ax_cov.set_ylim(0.0, max(0.12, float(feedback["coverage"].max()) * 1.15))
        ax_cov.legend(loc="upper right")

    bars = ax_bar.bar(["Baseline", "Dynamic"], [rmse_base, rmse_dyn], color=["#d95f02", "#1b9e77"])
    ax_bar.set_title("RMSE Summary")
    ax_bar.set_ylabel("RMSE [m]")
    ax_bar.grid(alpha=0.25, axis="y")
    for bar, val in zip(bars, [rmse_base, rmse_dyn]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom")

    text = [
        f"relative gain: {rel_gain:+.2f}%",
        f"mean per-KF gain: {mean_gain:+.4f} m",
        f"frames better: {frac_better * 100:.1f}%",
    ]
    if base_metric_rmse is not None and dyn_metric_rmse is not None:
        text.append(f"reported kf RMSE: {base_metric_rmse:.4f} -> {dyn_metric_rmse:.4f}")
    ax_bar.text(0.02, 0.98, "\n".join(text), transform=ax_bar.transAxes, va="top", ha="left")

    fig.suptitle("Dynamic Prediction SLAM Improvement Check", fontsize=15)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual comparison for baseline SLAM vs dynamic-prediction SLAM outputs."
    )
    parser.add_argument("--baseline-dir", required=True, help="Output directory from baseline run")
    parser.add_argument("--dynamic-dir", required=True, help="Output directory from dynamic run")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory containing rgb.txt and groundtruth.txt")
    parser.add_argument("--out", default=None, help="Output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    dynamic_dir = Path(args.dynamic_dir)
    dataset_dir = Path(args.dataset_dir)
    out_png = Path(args.out) if args.out else dynamic_dir / "dynamic_improvement_summary.png"

    rgb_ts = read_rgb_timestamps(dataset_dir)
    gt_ts, gt_xyz = read_groundtruth(dataset_dir)
    base_ts, base_xyz = load_kf_trajectory(baseline_dir)
    dyn_ts, dyn_xyz = load_kf_trajectory(dynamic_dir)
    frame_idx, base_xyz, dyn_xyz, gt_common = intersect_by_timestamp(
        base_ts, base_xyz, dyn_ts, dyn_xyz, rgb_ts, gt_ts, gt_xyz
    )

    r_b, s_b, t_b = umeyama_align(base_xyz, gt_common)
    r_d, s_d, t_d = umeyama_align(dyn_xyz, gt_common)
    base_aligned = apply_alignment(base_xyz, r_b, s_b, t_b)
    dyn_aligned = apply_alignment(dyn_xyz, r_d, s_d, t_d)
    base_err = np.linalg.norm(base_aligned - gt_common, axis=1)
    dyn_err = np.linalg.norm(dyn_aligned - gt_common, axis=1)

    feedback = read_feedback(dynamic_dir)
    base_metric_rmse = read_metric_rmse(baseline_dir)
    dyn_metric_rmse = read_metric_rmse(dynamic_dir)

    make_plot(
        out_png,
        frame_idx,
        gt_common,
        base_aligned,
        dyn_aligned,
        base_err,
        dyn_err,
        feedback,
        base_metric_rmse,
        dyn_metric_rmse,
    )
    save_error_csv(out_png.with_suffix(".csv"), frame_idx, base_err, dyn_err)

    print(f"Saved plot: {out_png}")
    print(f"Saved csv : {out_png.with_suffix('.csv')}")
    print(f"RMSE baseline: {rmse(base_err):.6f} m")
    print(f"RMSE dynamic : {rmse(dyn_err):.6f} m")
    print(f"Relative gain: {(rmse(base_err) - rmse(dyn_err)) / max(rmse(base_err), 1e-12) * 100.0:+.2f}%")


if __name__ == "__main__":
    main()
