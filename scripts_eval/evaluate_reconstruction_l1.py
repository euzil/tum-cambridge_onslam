from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_points(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        for key in keys:
            if key in data.files:
                pts = np.asarray(data[key], dtype=np.float64)
                break
        else:
            raise KeyError(f"{path} does not contain any of: {', '.join(keys)}")
    elif path.suffix == ".npy":
        pts = np.asarray(np.load(path), dtype=np.float64)
    else:
        pts = np.loadtxt(path, dtype=np.float64)
    pts = pts.reshape(-1, pts.shape[-1])
    if pts.shape[1] < 3:
        raise ValueError(f"Expected points with at least 3 coordinates, got {pts.shape}")
    pts = pts[:, :3]
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def nearest_neighbor_l1(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("--nearest-neighbor requires scipy") from exc
    tree = cKDTree(gt)
    _, idx = tree.query(pred, k=1)
    return np.abs(pred - gt[idx]).sum(axis=1)


def evaluate(pred: np.ndarray, gt: np.ndarray, nearest_neighbor: bool) -> dict[str, float | int]:
    pred_aligned = pred - pred.mean(axis=0, keepdims=True) + gt.mean(axis=0, keepdims=True)
    if nearest_neighbor:
        l1 = nearest_neighbor_l1(pred_aligned, gt)
    else:
        if pred_aligned.shape != gt.shape:
            raise ValueError("Pred/GT point clouds must have the same shape unless --nearest-neighbor is enabled")
        l1 = np.abs(pred_aligned - gt).sum(axis=1)
    return {
        "num_pred_points": int(len(pred)),
        "num_gt_points": int(len(gt)),
        "mean_l1": float(l1.mean()),
        "median_l1": float(np.median(l1)),
        "p90_l1": float(np.percentile(l1, 90)),
        "nearest_neighbor": int(nearest_neighbor),
    }


def write_outputs(metrics: dict[str, float | int | str], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with out_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])
    report = f"""# D4RT Point Cloud Reconstruction Metric

| Metric | Value |
|---|---:|
| Mean L1 | {float(metrics['mean_l1']):.6f} |
| Median L1 | {float(metrics['median_l1']):.6f} |
| P90 L1 | {float(metrics['p90_l1']):.6f} |
"""
    out_prefix.with_suffix(".report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D4RT-style point cloud reconstruction: mean-shift alignment + mean L1.")
    parser.add_argument("--pred", required=True, help="Predicted points: npz/npy/txt.")
    parser.add_argument("--gt", required=True, help="Ground-truth points: npz/npy/txt.")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--nearest-neighbor", action="store_true", help="Use nearest GT point if point-wise correspondence is unavailable.")
    args = parser.parse_args()

    pred = load_points(Path(args.pred), ("pred_points", "points", "pred"))
    gt = load_points(Path(args.gt), ("gt_points", "points", "gt"))
    metrics = evaluate(pred, gt, args.nearest_neighbor)
    metrics["pred_path"] = str(Path(args.pred))
    metrics["gt_path"] = str(Path(args.gt))
    write_outputs(metrics, Path(args.out_prefix))
    print(f"Saved reconstruction metrics to {Path(args.out_prefix).with_suffix('.json')}")


if __name__ == "__main__":
    main()
