from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_array(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        for key in keys:
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64)
        raise KeyError(f"{path} does not contain any of: {', '.join(keys)}")
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    return np.loadtxt(path, dtype=np.float64)


def valid_depth_mask(pred: np.ndarray, gt: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return np.isfinite(pred) & np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)


def align_scale_only(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    denom = float(np.sum(pred[mask] * pred[mask]))
    scale = float(np.sum(pred[mask] * gt[mask]) / max(denom, 1e-12))
    return pred * scale, scale


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    a = np.stack([pred[mask], np.ones(mask.sum(), dtype=np.float64)], axis=1)
    scale, shift = np.linalg.lstsq(a, gt[mask], rcond=None)[0]
    return pred * scale + shift, float(scale), float(shift)


def absrel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / np.maximum(gt[mask], 1e-12)))


def evaluate(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, min_depth: float, max_depth: float) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Pred/GT depth shapes differ: {pred.shape} vs {gt.shape}")
    valid = valid_depth_mask(pred, gt, min_depth, max_depth)
    if mask.size:
        if mask.shape != pred.shape:
            raise ValueError(f"Mask shape differs from depth shape: {mask.shape} vs {pred.shape}")
        valid &= mask.astype(bool)
    if not np.any(valid):
        raise ValueError("No valid depth pixels for evaluation")

    pred_s, scale_s = align_scale_only(pred, gt, valid)
    pred_ss, scale_ss, shift_ss = align_scale_shift(pred, gt, valid)
    return {
        "num_valid_pixels": int(valid.sum()),
        "absrel_s": absrel(pred_s, gt, valid),
        "absrel_ss": absrel(pred_ss, gt, valid),
        "scale_s": float(scale_s),
        "scale_ss": float(scale_ss),
        "shift_ss": float(shift_ss),
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
    report = f"""# D4RT Depth Metrics

| Metric | Value |
|---|---:|
| AbsRel(S) | {float(metrics['absrel_s']):.6f} |
| AbsRel(SS) | {float(metrics['absrel_ss']):.6f} |
| Valid pixels | {int(metrics['num_valid_pixels'])} |
"""
    out_prefix.with_suffix(".report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate D4RT-style depth metrics: AbsRel after scale-only and scale-shift alignment.")
    parser.add_argument("--pred", required=True, help="Predicted depth: npz/npy/txt.")
    parser.add_argument("--gt", required=True, help="Ground-truth depth: npz/npy/txt.")
    parser.add_argument("--mask", default="", help="Optional valid mask: npz/npy/txt.")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=float("inf"))
    args = parser.parse_args()

    pred = load_array(Path(args.pred), ("pred_depth", "depth", "pred"))
    gt = load_array(Path(args.gt), ("gt_depth", "depth", "gt"))
    mask = load_array(Path(args.mask), ("mask", "valid")) if args.mask else np.asarray([])
    metrics = evaluate(pred, gt, mask, args.min_depth, args.max_depth)
    metrics["pred_path"] = str(Path(args.pred))
    metrics["gt_path"] = str(Path(args.gt))
    write_outputs(metrics, Path(args.out_prefix))
    print(f"Saved depth metrics to {Path(args.out_prefix).with_suffix('.json')}")


if __name__ == "__main__":
    main()
