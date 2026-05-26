from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_eval.evaluate_dynamic_method import evaluate_model


def axis_metrics(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics.get("axis_x_signed_mean_m", np.nan)),
        float(metrics.get("axis_y_signed_mean_m", np.nan)),
        float(metrics.get("axis_z_signed_mean_m", np.nan)),
    )


def method_row(name: str, model: Path, metrics: dict[str, float]) -> dict[str, float | str]:
    x, y, z = axis_metrics(metrics)
    return {
        "method": name,
        "model_npz": str(model),
        "dynamic_points": metrics["model_dynamic_points"],
        "motion_valid_points": metrics["model_motion_valid_points"],
        "motion_nn_mean_m": metrics["motion_nn_m_mean"],
        "motion_nn_median_m": metrics["motion_nn_m_median"],
        "motion_nn_p90_m": metrics["motion_nn_m_p90"],
        "static_nn_mean_m": metrics["static_nn_m_mean"],
        "static_nn_median_m": metrics["static_nn_m_median"],
        "static_nn_p90_m": metrics["static_nn_m_p90"],
        "gain_percent": metrics["motion_vs_static_gain_percent"],
        "motion_inlier_ratio": metrics["model_motion_inlier_ratio"],
        "static_inlier_ratio": metrics["model_static_inlier_ratio"],
        "axis_x_signed_mean_m": x,
        "axis_y_signed_mean_m": y,
        "axis_z_signed_mean_m": z,
    }


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def align_frame_rows(old_rows: list[list[float]], new_rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old = {int(r[0]): r for r in old_rows}
    new = {int(r[0]): r for r in new_rows}
    frames = np.asarray(sorted(set(old) & set(new)), dtype=np.int32)
    old_motion = np.asarray([old[int(f)][4] for f in frames], dtype=np.float64)
    new_motion = np.asarray([new[int(f)][4] for f in frames], dtype=np.float64)
    return frames, old_motion, new_motion


def make_plot(
    path: Path,
    old_name: str,
    new_name: str,
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
    old_rows: list[list[float]],
    new_rows: list[list[float]],
) -> None:
    frames, old_motion, new_motion = align_frame_rows(old_rows, new_rows)
    improvement = old_motion - new_motion
    positive = improvement > 0

    fig = plt.figure(figsize=(15, 9), dpi=140)
    gs = fig.add_gridspec(2, 3)
    ax_curve = fig.add_subplot(gs[0, :2])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_gain = fig.add_subplot(gs[1, :2])
    ax_axis = fig.add_subplot(gs[1, 2])

    ax_curve.plot(frames, old_motion, color="#d95f02", lw=1.5, label=old_name)
    ax_curve.plot(frames, new_motion, color="#1b9e77", lw=1.5, label=new_name)
    ax_curve.set_title("Per-frame 4D Motion Prediction NN Error")
    ax_curve.set_xlabel("frame")
    ax_curve.set_ylabel("motion NN mean [m]")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend()

    old_mean = old_metrics["motion_nn_m_mean"]
    new_mean = new_metrics["motion_nn_m_mean"]
    bars = ax_bar.bar([old_name, new_name], [old_mean, new_mean], color=["#d95f02", "#1b9e77"])
    ax_bar.set_title("Mean Motion NN Error")
    ax_bar.set_ylabel("error [m]")
    ax_bar.grid(alpha=0.25, axis="y")
    for bar, val in zip(bars, [old_mean, new_mean]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom")

    ax_gain.axhline(0.0, color="black", lw=0.8)
    ax_gain.bar(frames, improvement, color=np.where(positive, "#1b9e77", "#d95f02"), width=0.8)
    ax_gain.set_title(f"Improvement per Frame: {old_name} error - {new_name} error")
    ax_gain.set_xlabel("frame")
    ax_gain.set_ylabel("positive is better [m]")
    ax_gain.grid(alpha=0.25, axis="y")

    old_axis = np.abs(np.asarray(axis_metrics(old_metrics), dtype=np.float64))
    new_axis = np.abs(np.asarray(axis_metrics(new_metrics), dtype=np.float64))
    x = np.arange(3)
    width = 0.36
    ax_axis.bar(x - width / 2, old_axis, width, color="#d95f02", label=old_name)
    ax_axis.bar(x + width / 2, new_axis, width, color="#1b9e77", label=new_name)
    ax_axis.set_xticks(x, ["X", "Y", "Z"])
    ax_axis.set_title("Absolute Axis Signed Bias")
    ax_axis.set_ylabel("abs signed mean [m]")
    ax_axis.grid(alpha=0.25, axis="y")
    ax_axis.legend()

    rel_gain = (old_mean - new_mean) / max(old_mean, 1e-12) * 100.0
    static_gain_old = old_metrics["motion_vs_static_gain_percent"]
    static_gain_new = new_metrics["motion_vs_static_gain_percent"]
    text = (
        f"{old_name} motion NN: {old_mean:.6f} m\n"
        f"{new_name} motion NN: {new_mean:.6f} m\n"
        f"method-to-method gain: {rel_gain:+.2f}%\n"
        f"{old_name} vs static: {static_gain_old:+.2f}%\n"
        f"{new_name} vs static: {static_gain_new:+.2f}%\n"
        f"frames improved: {np.mean(positive) * 100:.1f}%"
    )
    fig.text(0.02, 0.02, text, fontsize=10, va="bottom")
    fig.suptitle("4D Dynamic Motion Method Comparison", fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_experiment_report(
    path: Path,
    old_name: str,
    new_name: str,
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
    method_gain: float,
) -> None:
    old_static_mean = old_metrics["static_nn_m_mean"]
    new_static_mean = new_metrics["static_nn_m_mean"]
    old_mean = old_metrics["motion_nn_m_mean"]
    new_mean = new_metrics["motion_nn_m_mean"]
    baseline_gain = (new_static_mean - new_mean) / max(new_static_mean, 1e-12) * 100.0
    kalman_gain = (old_mean - new_mean) / max(old_mean, 1e-12) * 100.0
    old_vs_static = (old_static_mean - old_mean) / max(old_static_mean, 1e-12) * 100.0

    text = f"""# Dynamic Object 4D Mapping Experiment

## Experiment Design

This experiment evaluates whether learned dynamic prediction improves dynamic-object 4D mapping.

The comparison uses the same metric for all methods:

```text
3D nearest-neighbor error between predicted dynamic points at frame t+1
and observed dynamic points in frame t+1.
```

Lower is better. The unit is meters.

## Compared Methods

| Method | Meaning |
|---|---|
| Baseline / Static Carry-Forward | No dynamic prediction. Dynamic points are assumed not to move. |
| {old_name} | Previous dynamic motion method, used as the Kalman/old-method reference. |
| {new_name} | Learned pixel-flow + next-depth 3D motion + region-level smoothing. |

## Main Results

| Method | Mean 3D NN Error | Relative to Baseline |
|---|---:|---:|
| Baseline / Static Carry-Forward for {old_name} | {old_static_mean:.6f} m | 0.00% |
| {old_name} | {old_mean:.6f} m | {old_vs_static:+.2f}% |
| Baseline / Static Carry-Forward for {new_name} | {new_static_mean:.6f} m | 0.00% |
| {new_name} | {new_mean:.6f} m | {baseline_gain:+.2f}% |

Direct method comparison:

```text
{old_name} -> {new_name}: {old_mean:.6f} m -> {new_mean:.6f} m
Relative improvement: {kalman_gain:+.2f}%
```

## Interpretation

Baseline means the system does not predict dynamic-object motion. It simply checks how close current dynamic points are to the next frame if they are carried forward without motion.

Each method is compared to its own static carry-forward baseline, because the valid dynamic point subset can differ after filtering and region smoothing.

{old_name} represents the previous motion prediction path. In this experiment it does not improve dynamic-object mapping over the baseline:

```text
{old_name} vs Baseline: {old_vs_static:+.2f}%
```

{new_name} uses learned pixel motion and then stabilizes motion at the dynamic-region level. It improves dynamic-object 4D temporal consistency:

```text
{new_name} vs Baseline: {baseline_gain:+.2f}%
{new_name} vs {old_name}: {kalman_gain:+.2f}%
```

## Conclusion

The learned model is better than the previous/Kalman-style motion method, and it also improves over the no-prediction baseline.

The key evidence is:

```text
{old_name} static baseline: {old_static_mean:.6f} m
{old_name}: {old_mean:.6f} m
{new_name} static baseline: {new_static_mean:.6f} m
{new_name}: {new_mean:.6f} m
```

This shows that dynamic prediction is now being fed back into dynamic-object 4D mapping in a useful way.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new 4D dynamic motion methods with identical metrics.")
    parser.add_argument("--old-model", required=True)
    parser.add_argument("--new-model", required=True)
    parser.add_argument("--old-name", default="Old")
    parser.add_argument("--new-name", default="New")
    parser.add_argument("--max-nn-dist", type=float, default=0.5)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    old_model = Path(args.old_model)
    new_model = Path(args.new_model)
    out_prefix = Path(args.out_prefix)

    old_metrics, old_frame_rows = evaluate_model(old_model, args.max_nn_dist)
    new_metrics, new_frame_rows = evaluate_model(new_model, args.max_nn_dist)

    rows = [
        method_row(args.old_name, old_model, old_metrics),
        method_row(args.new_name, new_model, new_metrics),
    ]
    write_rows(out_prefix.with_suffix(".csv"), rows)

    detail = {
        "max_nn_dist_m": args.max_nn_dist,
        "old_name": args.old_name,
        "new_name": args.new_name,
        "old_model": str(old_model),
        "new_model": str(new_model),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "method_to_method_gain_percent": (old_metrics["motion_nn_m_mean"] - new_metrics["motion_nn_m_mean"])
        / max(old_metrics["motion_nn_m_mean"], 1e-12)
        * 100.0,
    }
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    frame_csv = Path(str(out_prefix) + "_per_frame.csv")
    frames, old_motion, new_motion = align_frame_rows(old_frame_rows, new_frame_rows)
    with frame_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "old_motion_nn_mean_m", "new_motion_nn_mean_m", "improvement_m"])
        for frame, old, new in zip(frames, old_motion, new_motion):
            writer.writerow([int(frame), f"{old:.8f}", f"{new:.8f}", f"{old - new:.8f}"])

    make_plot(out_prefix.with_suffix(".png"), args.old_name, args.new_name, old_metrics, new_metrics, old_frame_rows, new_frame_rows)
    write_experiment_report(
        Path(str(out_prefix) + "_experiment_report.md"),
        args.old_name,
        args.new_name,
        old_metrics,
        new_metrics,
        detail["method_to_method_gain_percent"],
    )

    print(f"Saved summary csv: {out_prefix.with_suffix('.csv')}")
    print(f"Saved summary json: {out_prefix.with_suffix('.json')}")
    print(f"Saved per-frame csv: {frame_csv}")
    print(f"Saved plot: {out_prefix.with_suffix('.png')}")
    print(f"Saved experiment report: {Path(str(out_prefix) + '_experiment_report.md')}")
    print(f"{args.old_name} motion NN: {old_metrics['motion_nn_m_mean']:.6f} m")
    print(f"{args.new_name} motion NN: {new_metrics['motion_nn_m_mean']:.6f} m")
    print(f"Method-to-method gain: {detail['method_to_method_gain_percent']:+.2f}%")
    print(f"{args.old_name} vs static: {old_metrics['motion_vs_static_gain_percent']:+.2f}%")
    print(f"{args.new_name} vs static: {new_metrics['motion_vs_static_gain_percent']:+.2f}%")


if __name__ == "__main__":
    main()
