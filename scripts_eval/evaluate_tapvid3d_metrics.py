from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_PIXEL_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_ABSOLUTE_THRESHOLDS = (0.01, 0.02, 0.04, 0.08, 0.16)


def load_first_available(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    raise KeyError(f"Missing required array; tried {', '.join(names)}")


def load_tracks_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    pred = load_first_available(data, ("pred_tracks", "pred_points", "pred", "prediction")).astype(np.float64)
    gt = load_first_available(data, ("gt_tracks", "gt_points", "gt", "target")).astype(np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected pred/gt tracks with identical shape [N,T,3], got {pred.shape} and {gt.shape}")

    if "visible" in data.files:
        visible = np.asarray(data["visible"]).astype(bool)
    elif "gt_visible" in data.files:
        visible = np.asarray(data["gt_visible"]).astype(bool)
    elif "occluded" in data.files:
        visible = ~np.asarray(data["occluded"]).astype(bool)
    elif "gt_occluded" in data.files:
        visible = ~np.asarray(data["gt_occluded"]).astype(bool)
    else:
        visible = np.ones(pred.shape[:2], dtype=bool)

    if "pred_visible" in data.files:
        pred_visible = np.asarray(data["pred_visible"]).astype(bool)
    elif "pred_occluded" in data.files:
        pred_visible = ~np.asarray(data["pred_occluded"]).astype(bool)
    else:
        pred_visible = np.ones(pred.shape[:2], dtype=bool)

    if visible.shape != pred.shape[:2] or pred_visible.shape != pred.shape[:2]:
        raise ValueError("Visibility arrays must have shape [N,T]")

    query_points = None
    for key in ("query_points", "queries", "query_indices"):
        if key in data.files:
            query_points = np.asarray(data[key])
            break
    return {"pred": pred, "gt": gt, "visible": visible, "pred_visible": pred_visible, "query_points": query_points}


def parse_thresholds(text: str) -> tuple[float, ...]:
    vals = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if not vals:
        raise ValueError("At least one threshold is required")
    return vals


def scale_tracks(pred: np.ndarray, gt: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return pred, gt
    if mode == "global_median":
        pred_norm = np.median(np.linalg.norm(pred.reshape(-1, 3), axis=1))
        gt_norm = np.median(np.linalg.norm(gt.reshape(-1, 3), axis=1))
        scale = gt_norm / max(pred_norm, 1e-12)
        return pred * scale, gt
    if mode == "per_track_displacement":
        pred_disp = np.linalg.norm(pred - pred[:, :1], axis=2)
        gt_disp = np.linalg.norm(gt - gt[:, :1], axis=2)
        pred_med = np.median(pred_disp, axis=1)
        gt_med = np.median(gt_disp, axis=1)
        scale = gt_med / np.maximum(pred_med, 1e-12)
        return pred * scale[:, None, None], gt
    raise ValueError(f"Unknown scale mode: {mode}")


def average_jaccard(correct: np.ndarray, pred_visible: np.ndarray, gt_visible: np.ndarray) -> float:
    true_positive = correct & pred_visible & gt_visible
    false_positive = (~gt_visible) & pred_visible
    false_negative_or_bad_match = gt_visible & pred_visible & (~correct)
    denom = int(gt_visible.sum() + false_positive.sum() + false_negative_or_bad_match.sum())
    if denom == 0:
        return 0.0
    return float(true_positive.sum() / denom)


def occlusion_accuracy(pred_visible: np.ndarray, gt_visible: np.ndarray) -> float:
    return float((pred_visible == gt_visible).mean()) if gt_visible.size else 0.0


def evaluate_tapvid3d(
    pred: np.ndarray,
    gt: np.ndarray,
    visible: np.ndarray,
    pred_visible: np.ndarray,
    thresholds: tuple[float, ...],
    scale_mode: str,
    focal_length: float,
    depth_axis: int,
) -> dict[str, float | int | str]:
    pred, gt = scale_tracks(pred, gt, scale_mode)
    dist = np.linalg.norm(pred - gt, axis=2)
    valid_dist = dist[visible]
    if valid_dist.size == 0:
        raise ValueError("No visible GT track points available for evaluation")

    if focal_length > 0.0:
        depth = np.abs(gt[:, :, depth_axis])
        threshold_maps = [depth * threshold / focal_length for threshold in thresholds]
        threshold_kind = "depth_adaptive_pixel"
    else:
        threshold_maps = [np.full(dist.shape, threshold, dtype=np.float64) for threshold in thresholds]
        threshold_kind = "absolute_3d"

    metrics: dict[str, float | int | str] = {
        "num_tracks": int(pred.shape[0]),
        "num_frames": int(pred.shape[1]),
        "num_visible_points": int(visible.sum()),
        "scale_mode": scale_mode,
        "thresholds": ",".join(str(t) for t in thresholds),
        "threshold_kind": threshold_kind,
        "focal_length": float(focal_length),
        "depth_axis": int(depth_axis),
        "world_l1": float(np.abs(pred[visible] - gt[visible]).sum(axis=1).mean()),
        "world_l2": float(valid_dist.mean()),
        "oa": occlusion_accuracy(pred_visible, visible) * 100.0,
    }

    apd_values: list[float] = []
    aj_values: list[float] = []
    for threshold, threshold_map in zip(thresholds, threshold_maps):
        correct = dist <= threshold_map
        apd = float(correct[visible].mean()) if visible.any() else 0.0
        aj = average_jaccard(correct, pred_visible, visible)
        suffix = f"{threshold:g}px" if focal_length > 0.0 else f"{threshold:g}"
        metrics[f"apd3d_at_{suffix}"] = apd * 100.0
        metrics[f"aj_at_{suffix}"] = aj * 100.0
        apd_values.append(apd)
        aj_values.append(aj)

    metrics["apd3d"] = float(np.mean(apd_values) * 100.0)
    metrics["aj"] = float(np.mean(aj_values) * 100.0)
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
    report = f"""# TAPVid-3D Metrics

| Metric | Value |
|---|---:|
| APD3D | {float(metrics['apd3d']):.3f} |
| OA | {float(metrics['oa']):.3f} |
| AJ | {float(metrics['aj']):.3f} |
| World L1 | {float(metrics['world_l1']):.6f} |
| World L2 | {float(metrics['world_l2']):.6f} |
"""
    out_prefix.with_suffix(".report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate D4RT/TAPVid-3D-style APD3D, OA, AJ, and world-coordinate L1 for 3D tracks."
    )
    parser.add_argument("--tracks", required=True, help="NPZ with pred_tracks, gt_tracks, and optional visible/pred_visible arrays.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for .json/.csv/.report.md.")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(v) for v in DEFAULT_PIXEL_THRESHOLDS),
        help="Comma-separated pixel thresholds when --focal-length is set; otherwise absolute 3D thresholds.",
    )
    parser.add_argument(
        "--absolute-thresholds",
        default="",
        help="Comma-separated absolute 3D thresholds. Overrides --thresholds and disables depth-adaptive thresholds.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=0.0,
        help="Focal length in pixels for TAPVid-3D depth-adaptive thresholds: delta_3d = Z * delta_px / f.",
    )
    parser.add_argument("--depth-axis", type=int, default=2, choices=(0, 1, 2), help="Axis used as depth Z.")
    parser.add_argument(
        "--scale-mode",
        default="none",
        choices=("none", "global_median", "per_track_displacement"),
        help="Optional scale normalization before metric computation.",
    )
    args = parser.parse_args()

    arrays = load_tracks_npz(Path(args.tracks))
    metrics = evaluate_tapvid3d(
        pred=arrays["pred"],
        gt=arrays["gt"],
        visible=arrays["visible"],
        pred_visible=arrays["pred_visible"],
        thresholds=parse_thresholds(args.absolute_thresholds) if args.absolute_thresholds else parse_thresholds(args.thresholds),
        scale_mode=args.scale_mode,
        focal_length=0.0 if args.absolute_thresholds else args.focal_length,
        depth_axis=args.depth_axis,
    )
    metrics["tracks_path"] = str(Path(args.tracks))
    write_outputs(metrics, Path(args.out_prefix))
    print(f"Saved TAPVid-3D metrics to {Path(args.out_prefix).with_suffix('.json')}")


if __name__ == "__main__":
    main()
