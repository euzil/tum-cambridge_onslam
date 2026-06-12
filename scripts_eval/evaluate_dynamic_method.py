from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


def summarize(values: Iterable[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
        }
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
    }


def read_numeric_csv(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}

    cols: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key is None or value is None or value == "":
                    continue
                try:
                    cols.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return cols


def summarize_logs(output_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}

    id_cols = read_numeric_csv(output_dir / "dynamic_id_match_stats.csv")
    if id_cols:
        out.update(summarize(id_cols.get("n_points", []), "id_points"))
        out.update(summarize(id_cols.get("n_matched", []), "id_matched"))
        out.update(summarize(id_cols.get("match_ratio", []), "id_match_ratio"))
        out.update(summarize(id_cols.get("mean_dist", []), "id_match_dist"))

    fb_cols = read_numeric_csv(output_dir / "dynamic_feedback_stats.csv")
    if fb_cols:
        out.update(summarize(fb_cols.get("n_predictions", []), "feedback_predictions"))
        out.update(summarize(fb_cols.get("coverage", []), "feedback_coverage"))
        applied = np.asarray(fb_cols.get("applied", []), dtype=np.float64)
        if applied.size:
            out["feedback_applied_ratio"] = float(applied.mean())

    motion_cols = read_numeric_csv(output_dir / "dynamic_motion_feedback_stats.csv")
    if motion_cols:
        out.update(summarize(motion_cols.get("n_written", []), "motion_written"))
        out.update(summarize(motion_cols.get("coverage", []), "motion_coverage"))
        out.update(summarize(motion_cols.get("prior_mean_m", []), "motion_prior_mean_m"))
        applied = np.asarray(motion_cols.get("applied", []), dtype=np.float64)
        if applied.size:
            out["motion_applied_ratio"] = float(applied.mean())

    ba_cols = read_numeric_csv(output_dir / "dynamic_ba_edge_stats.csv")
    if ba_cols:
        out.update(summarize(ba_cols.get("n_edges", []), "ba_edges"))
        out.update(summarize(ba_cols.get("n_adjacent_with_mask", []), "ba_adjacent_with_mask"))
        out.update(summarize(ba_cols.get("mean_mask_pixels", []), "ba_mean_mask_pixels"))
        adj = np.asarray(ba_cols.get("n_adjacent_forward", []), dtype=np.float64)
        used = np.asarray(ba_cols.get("n_adjacent_with_mask", []), dtype=np.float64)
        if adj.size and used.size:
            valid = adj > 0
            out["ba_adjacent_mask_ratio"] = float((used[valid] / adj[valid]).mean()) if np.any(valid) else 0.0

    return out


def evaluate_model(model_npz: Path, max_nn_dist: float) -> tuple[dict[str, float], list[list[float]]]:
    data = np.load(model_npz)
    required = ["points", "dynamic", "motion_valid", "predicted_next_points", "frame_offsets"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{model_npz} is missing required fields: {missing}")

    points = np.asarray(data["points"], dtype=np.float32)
    predicted = np.asarray(data["predicted_next_points"], dtype=np.float32)
    dynamic = np.asarray(data["dynamic"]).astype(bool)
    motion_valid = np.asarray(data["motion_valid"]).astype(bool)
    offsets = np.asarray(data["frame_offsets"], dtype=np.int64)

    static_dists: list[float] = []
    motion_dists: list[float] = []
    motion_inlier_dists: list[float] = []
    axis_errors: list[np.ndarray] = []
    rows: list[list[float]] = []
    total_motion_queries = 0
    total_motion_inliers = 0
    total_static_inliers = 0

    for t in range(len(offsets) - 2):
        lo, hi = int(offsets[t]), int(offsets[t + 1])
        nlo, nhi = int(offsets[t + 1]), int(offsets[t + 2])

        src_mask = dynamic[lo:hi] & motion_valid[lo:hi]
        tgt_mask = dynamic[nlo:nhi]
        if not np.any(src_mask) or not np.any(tgt_mask):
            continue

        src_pts = points[lo:hi][src_mask]
        pred_pts = predicted[lo:hi][src_mask]
        tgt_pts = points[nlo:nhi][tgt_mask]

        tree = cKDTree(tgt_pts)
        static_dist, _ = tree.query(src_pts, k=1)
        motion_dist, idx = tree.query(pred_pts, k=1)

        compare_keep = (static_dist <= max_nn_dist) | (motion_dist <= max_nn_dist)
        if not np.any(compare_keep):
            continue

        static_cmp = static_dist[compare_keep]
        motion_cmp = motion_dist[compare_keep]
        static_dists.extend(static_cmp.tolist())
        motion_dists.extend(motion_cmp.tolist())

        motion_keep = motion_dist <= max_nn_dist
        static_keep = static_dist <= max_nn_dist
        total_motion_queries += int(len(motion_dist))
        total_motion_inliers += int(motion_keep.sum())
        total_static_inliers += int(static_keep.sum())

        if np.any(motion_keep):
            matched = tgt_pts[idx[motion_keep]]
            err = pred_pts[motion_keep] - matched
            axis_errors.append(err.astype(np.float32))
            motion_inlier_dists.extend(motion_dist[motion_keep].tolist())

        rows.append(
            [
                float(t),
                float(len(src_pts)),
                float(static_cmp.mean()),
                float(np.median(static_cmp)),
                float(motion_cmp.mean()),
                float(np.median(motion_cmp)),
                float((static_cmp - motion_cmp).mean()),
                float(motion_keep.mean()),
            ]
        )

    metrics: dict[str, float] = {
        "model_frames": int(len(offsets) - 1),
        "model_dynamic_points": int(dynamic.sum()),
        "model_motion_valid_points": int((dynamic & motion_valid).sum()),
        "model_motion_query_points": int(total_motion_queries),
        "model_motion_inlier_points": int(total_motion_inliers),
        "model_static_inlier_points": int(total_static_inliers),
        "model_motion_inlier_ratio": float(total_motion_inliers / max(total_motion_queries, 1)),
        "model_static_inlier_ratio": float(total_static_inliers / max(total_motion_queries, 1)),
    }
    metrics.update(summarize(static_dists, "static_nn_m"))
    metrics.update(summarize(motion_dists, "motion_nn_m"))
    metrics.update(summarize(motion_inlier_dists, "motion_inlier_nn_m"))

    static_mean = metrics.get("static_nn_m_mean", float("nan"))
    motion_mean = metrics.get("motion_nn_m_mean", float("nan"))
    metrics["motion_vs_static_gain_percent"] = float(
        (static_mean - motion_mean) / max(static_mean, 1e-8) * 100.0
    )

    if axis_errors:
        err = np.concatenate(axis_errors, axis=0)
        abs_err = np.abs(err)
        for i, axis in enumerate(["x", "y", "z"]):
            metrics[f"axis_{axis}_signed_mean_m"] = float(err[:, i].mean())
            metrics[f"axis_{axis}_signed_median_m"] = float(np.median(err[:, i]))
            metrics[f"axis_{axis}_abs_mean_m"] = float(abs_err[:, i].mean())
            metrics[f"axis_{axis}_abs_p90_m"] = float(np.percentile(abs_err[:, i], 90))

    return metrics, rows


def write_summary_csv(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


def write_frame_csv(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "n_motion_points",
                "static_nn_mean_m",
                "static_nn_median_m",
                "motion_nn_mean_m",
                "motion_nn_median_m",
                "mean_gain_m",
                "motion_inlier_ratio",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize numeric evaluation for online dynamic 4D SLAM.")
    parser.add_argument("--model-npz", required=True, help="Path to exported model_4d.npz")
    parser.add_argument("--output-dir", default="", help="SLAM output dir containing dynamic_*.csv logs")
    parser.add_argument("--max-nn-dist", type=float, default=0.5)
    parser.add_argument("--out-prefix", default="", help="Output prefix for summary files")
    args = parser.parse_args()

    model_path = Path(args.model_npz)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent.parent
    out_prefix = Path(args.out_prefix) if args.out_prefix else model_path.parent / "dynamic_method_eval"

    model_metrics, frame_rows = evaluate_model(model_path, args.max_nn_dist)
    log_metrics = summarize_logs(output_dir)
    metrics = {
        "model_npz": str(model_path),
        "output_dir": str(output_dir),
        "max_nn_dist_m": float(args.max_nn_dist),
        **log_metrics,
        **model_metrics,
    }

    write_summary_csv(out_prefix.with_suffix(".csv"), metrics)
    write_frame_csv(Path(str(out_prefix) + "_per_frame.csv"), frame_rows)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"model: {model_path}")
    print(f"output logs: {output_dir}")
    print(f"summary csv: {out_prefix.with_suffix('.csv')}")
    print(f"summary json: {out_prefix.with_suffix('.json')}")
    print(f"per-frame csv: {Path(str(out_prefix) + '_per_frame.csv')}")
    print("")
    print(f"online predictions mean: {metrics.get('feedback_predictions_mean', float('nan')):.3f}")
    print(f"feedback applied ratio: {metrics.get('feedback_applied_ratio', float('nan')):.3f}")
    print(f"BA adjacent mask ratio: {metrics.get('ba_adjacent_mask_ratio', float('nan')):.3f}")
    print(f"motion valid dynamic points: {metrics['model_motion_valid_points']}")
    print(
        "4D NN error motion/static: "
        f"{metrics['motion_nn_m_mean']:.6f} / {metrics['static_nn_m_mean']:.6f} m"
    )
    print(f"4D temporal consistency gain: {metrics['motion_vs_static_gain_percent']:+.2f}%")
    print(
        "axis signed mean [m]: "
        f"x={metrics.get('axis_x_signed_mean_m', float('nan')):+.6f}, "
        f"y={metrics.get('axis_y_signed_mean_m', float('nan')):+.6f}, "
        f"z={metrics.get('axis_z_signed_mean_m', float('nan')):+.6f}"
    )


if __name__ == "__main__":
    main()
