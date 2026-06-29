from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_eval.evaluate_dynamic_method import evaluate_model


def axis_bias_l2(metrics: dict[str, float]) -> float:
    x = float(metrics.get("axis_x_signed_mean_m", 0.0))
    y = float(metrics.get("axis_y_signed_mean_m", 0.0))
    z = float(metrics.get("axis_z_signed_mean_m", 0.0))
    return float(math.sqrt(x * x + y * y + z * z))


def rel_gain(lower_better_baseline: float, lower_better_new: float) -> float:
    return (lower_better_baseline - lower_better_new) / max(lower_better_baseline, 1e-12) * 100.0


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs learned method for dynamic-object 4D modeling quality."
    )
    parser.add_argument("--baseline-model", required=True, help="Path to baseline model_4d.npz")
    parser.add_argument("--learned-model", required=True, help="Path to learned model_4d.npz")
    parser.add_argument("--baseline-name", default="Baseline")
    parser.add_argument("--learned-name", default="Learned")
    parser.add_argument("--max-nn-dist", type=float, default=0.5)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    baseline_model = Path(args.baseline_model)
    learned_model = Path(args.learned_model)
    out_prefix = Path(args.out_prefix)

    b, _ = evaluate_model(baseline_model, args.max_nn_dist)
    l, _ = evaluate_model(learned_model, args.max_nn_dist)

    b_axis_l2 = axis_bias_l2(b)
    l_axis_l2 = axis_bias_l2(l)

    rows = [
        {
            "method": args.baseline_name,
            "model_npz": str(baseline_model),
            "motion_nn_mean_m": float(b["motion_nn_m_mean"]),
            "motion_nn_median_m": float(b["motion_nn_m_median"]),
            "motion_vs_static_gain_percent": float(b["motion_vs_static_gain_percent"]),
            "motion_inlier_ratio": float(b["model_motion_inlier_ratio"]),
            "axis_x_signed_mean_m": float(b.get("axis_x_signed_mean_m", np.nan)),
            "axis_y_signed_mean_m": float(b.get("axis_y_signed_mean_m", np.nan)),
            "axis_z_signed_mean_m": float(b.get("axis_z_signed_mean_m", np.nan)),
            "axis_bias_l2_m": b_axis_l2,
        },
        {
            "method": args.learned_name,
            "model_npz": str(learned_model),
            "motion_nn_mean_m": float(l["motion_nn_m_mean"]),
            "motion_nn_median_m": float(l["motion_nn_m_median"]),
            "motion_vs_static_gain_percent": float(l["motion_vs_static_gain_percent"]),
            "motion_inlier_ratio": float(l["model_motion_inlier_ratio"]),
            "axis_x_signed_mean_m": float(l.get("axis_x_signed_mean_m", np.nan)),
            "axis_y_signed_mean_m": float(l.get("axis_y_signed_mean_m", np.nan)),
            "axis_z_signed_mean_m": float(l.get("axis_z_signed_mean_m", np.nan)),
            "axis_bias_l2_m": l_axis_l2,
        },
    ]

    comparison = {
        "baseline_name": args.baseline_name,
        "learned_name": args.learned_name,
        "baseline_model": str(baseline_model),
        "learned_model": str(learned_model),
        "max_nn_dist_m": float(args.max_nn_dist),
        "metrics": {
            "motion_nn_mean_gain_percent": rel_gain(float(b["motion_nn_m_mean"]), float(l["motion_nn_m_mean"])),
            "motion_nn_median_gain_percent": rel_gain(float(b["motion_nn_m_median"]), float(l["motion_nn_m_median"])),
            "motion_vs_static_gain_delta_percent": float(l["motion_vs_static_gain_percent"] - b["motion_vs_static_gain_percent"]),
            "motion_inlier_ratio_delta": float(l["model_motion_inlier_ratio"] - b["model_motion_inlier_ratio"]),
            "axis_bias_l2_gain_percent": rel_gain(b_axis_l2, l_axis_l2),
        },
    }

    write_csv(out_prefix.with_suffix(".csv"), rows)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    report = f"""# Dynamic 4D Modeling Comparison

## Methods
- {args.baseline_name}: `{baseline_model}`
- {args.learned_name}: `{learned_model}`

## Key Metrics (Your Target)
- `motion_nn_mean_m` (lower is better): {b['motion_nn_m_mean']:.6f} -> {l['motion_nn_m_mean']:.6f} m
- `motion_nn_median_m` (lower is better): {b['motion_nn_m_median']:.6f} -> {l['motion_nn_m_median']:.6f} m
- `motion_vs_static_gain_percent` (higher is better): {b['motion_vs_static_gain_percent']:+.2f}% -> {l['motion_vs_static_gain_percent']:+.2f}%
- `motion_inlier_ratio` (higher is better): {b['model_motion_inlier_ratio']:.4f} -> {l['model_motion_inlier_ratio']:.4f}
- `axis_bias_l2_m` (lower is better): {b_axis_l2:.6f} -> {l_axis_l2:.6f} m

## Relative Change (Baseline -> Learned)
- motion_nn_mean gain: {comparison['metrics']['motion_nn_mean_gain_percent']:+.2f}%
- motion_nn_median gain: {comparison['metrics']['motion_nn_median_gain_percent']:+.2f}%
- motion_vs_static_gain delta: {comparison['metrics']['motion_vs_static_gain_delta_percent']:+.2f}%
- motion_inlier_ratio delta: {comparison['metrics']['motion_inlier_ratio_delta']:+.4f}
- axis_bias_l2 gain: {comparison['metrics']['axis_bias_l2_gain_percent']:+.2f}%
"""
    out_prefix.with_suffix(".report.md").write_text(report, encoding="utf-8")

    print(f"Saved metrics CSV: {out_prefix.with_suffix('.csv')}")
    print(f"Saved comparison JSON: {out_prefix.with_suffix('.json')}")
    print(f"Saved report MD: {out_prefix.with_suffix('.report.md')}")


if __name__ == "__main__":
    main()
