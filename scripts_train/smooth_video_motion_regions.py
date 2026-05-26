from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def robust_component_motion(values: np.ndarray, mode: str, trim_quantile: float) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((3,), dtype=np.float32)
    if len(values) < 4 or trim_quantile <= 0:
        return np.median(values, axis=0).astype(np.float32) if mode == "median" else values.mean(axis=0).astype(np.float32)

    mag = np.linalg.norm(values, axis=1)
    hi = np.quantile(mag, min(max(trim_quantile, 0.0), 1.0))
    keep = mag <= hi
    kept = values[keep] if np.any(keep) else values
    return np.median(kept, axis=0).astype(np.float32) if mode == "median" else kept.mean(axis=0).astype(np.float32)


def smooth_frame(
    motion: np.ndarray,
    mask: np.ndarray,
    min_component_pixels: int,
    dilate_iters: int,
    mode: str,
    trim_quantile: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    base_mask = mask > 0
    if dilate_iters > 0:
        label_mask = ndi.binary_dilation(base_mask, iterations=dilate_iters)
    else:
        label_mask = base_mask

    labels, n_labels = ndi.label(label_mask)
    out_motion = np.zeros_like(motion, dtype=np.float32)
    out_mask = np.zeros(mask.shape, dtype=np.float32)
    kept_components = 0

    for lab in range(1, n_labels + 1):
        comp = labels == lab
        valid = comp & base_mask
        count = int(valid.sum())
        if count < min_component_pixels:
            continue
        comp_motion = robust_component_motion(motion[valid], mode, trim_quantile)
        target = comp & base_mask
        if blend < 1.0:
            out_motion[target] = (1.0 - blend) * motion[target] + blend * comp_motion[None, :]
        else:
            out_motion[target] = comp_motion[None, :]
        out_mask[target] = 1.0
        kept_components += 1

    return out_motion, out_mask, kept_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth dynamic motion by connected regions/object-like components.")
    parser.add_argument("--video-npz", required=True)
    parser.add_argument("--out-video", required=True)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--dilate-iters", type=int, default=1)
    parser.add_argument("--mode", choices=["median", "mean"], default="median")
    parser.add_argument("--trim-quantile", type=float, default=0.9)
    parser.add_argument("--blend", type=float, default=0.75)
    args = parser.parse_args()

    in_path = Path(args.video_npz)
    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path)
    required = ["dynamic_motions", "dynamic_motion_masks"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{in_path} is missing required fields: {missing}")

    motions = np.asarray(data["dynamic_motions"], dtype=np.float32)
    masks = np.asarray(data["dynamic_motion_masks"], dtype=np.float32)
    smooth_motions = np.zeros_like(motions, dtype=np.float32)
    smooth_masks = np.zeros_like(masks, dtype=np.float32)
    component_counts = []

    for t in range(len(motions)):
        sm, ma, n_comp = smooth_frame(
            motions[t],
            masks[t],
            args.min_component_pixels,
            args.dilate_iters,
            args.mode,
            args.trim_quantile,
            args.blend,
        )
        smooth_motions[t] = sm
        smooth_masks[t] = ma
        component_counts.append(n_comp)
        print(f"\r[region-motion] frame {t + 1:04d}/{len(motions)} components={n_comp}", end="", flush=True)
    print()

    out = {key: data[key] for key in data.files}
    out["dynamic_motions_before_region_smooth"] = motions
    out["dynamic_motion_masks_before_region_smooth"] = masks
    out["dynamic_motions"] = smooth_motions
    out["dynamic_motion_priors"] = smooth_motions
    out["dynamic_motion_masks"] = smooth_masks
    np.savez_compressed(out_path, **out)

    active = smooth_masks > 0
    mag = np.linalg.norm(smooth_motions[active], axis=-1) if np.any(active) else np.zeros((0,), dtype=np.float32)
    meta = {
        "source_video": str(in_path),
        "out_video": str(out_path),
        "frames": int(len(motions)),
        "active_pixels": int(active.sum()),
        "active_ratio": float(active.mean()),
        "component_mean": float(np.mean(component_counts)) if component_counts else 0.0,
        "component_max": int(np.max(component_counts)) if component_counts else 0,
        "motion_mean_m": float(mag.mean()) if mag.size else 0.0,
        "motion_p90_m": float(np.percentile(mag, 90)) if mag.size else 0.0,
        "min_component_pixels": args.min_component_pixels,
        "dilate_iters": args.dilate_iters,
        "mode": args.mode,
        "trim_quantile": args.trim_quantile,
        "blend": args.blend,
    }
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"saved region-smoothed video: {out_path}")
    print(f"saved metadata: {out_path.with_suffix('.json')}")
    print(
        f"active={meta['active_pixels']} ({meta['active_ratio']:.4%}) "
        f"components_mean={meta['component_mean']:.2f} motion_mean={meta['motion_mean_m']:.6f}m"
    )


if __name__ == "__main__":
    main()
