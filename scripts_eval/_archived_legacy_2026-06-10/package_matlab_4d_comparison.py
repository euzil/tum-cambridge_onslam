from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.io import savemat


MODEL_KEYS = [
    "points",
    "colors",
    "dynamic",
    "uncertainty",
    "time",
    "timestamps",
    "frame_offsets",
    "frame_counts",
    "frame_dynamic_counts",
    "poses",
    "intrinsics",
    "static_map_points",
    "static_map_colors",
    "static_map_obs",
]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    missing = [key for key in ["points", "colors", "dynamic", "frame_offsets"] if key not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")
    return {key: np.asarray(data[key]) for key in data.files}


def frame_ranges(frame_offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts_1 = frame_offsets[:-1].astype(np.int64) + 1
    ends_1 = frame_offsets[1:].astype(np.int64)
    return starts_1.reshape(-1, 1), ends_1.reshape(-1, 1)


def matlab_dict(data: dict[str, np.ndarray], mode: str, source_model: Path) -> dict[str, np.ndarray | str]:
    points = np.asarray(data["points"], dtype=np.float32)
    dynamic = np.asarray(data["dynamic"]).astype(bool)
    frame_offsets = np.asarray(data["frame_offsets"], dtype=np.int64)
    starts_1, ends_1 = frame_ranges(frame_offsets)

    if mode == "baseline":
        motion_world = np.zeros_like(points, dtype=np.float32)
        predicted_next_points = points.astype(np.float32, copy=True)
        motion_valid = dynamic.copy()
        method_name = "baseline_static_carry_forward"
    else:
        motion_world = np.asarray(data.get("motion_world", np.zeros_like(points)), dtype=np.float32)
        predicted_next_points = np.asarray(data.get("predicted_next_points", points), dtype=np.float32)
        motion_valid = np.asarray(data.get("motion_valid", dynamic)).astype(bool)
        method_name = mode

    out: dict[str, np.ndarray | str] = {
        "points": points,
        "colors": np.asarray(data["colors"], dtype=np.uint8),
        "dynamic": dynamic.reshape(-1, 1),
        "uncertainty": np.asarray(data.get("uncertainty", np.zeros(len(points))), dtype=np.float32).reshape(-1, 1),
        "time": np.asarray(data.get("time", np.zeros(len(points))), dtype=np.float32).reshape(-1, 1),
        "timestamps": np.asarray(data.get("timestamps", np.arange(len(frame_offsets) - 1)), dtype=np.float32).reshape(-1, 1),
        "frame_offsets0": frame_offsets.reshape(-1, 1),
        "frame_start_1": starts_1,
        "frame_end_1": ends_1,
        "frame_counts": np.asarray(data.get("frame_counts", np.diff(frame_offsets)), dtype=np.int64).reshape(-1, 1),
        "frame_dynamic_counts": np.asarray(
            data.get("frame_dynamic_counts", np.zeros(len(frame_offsets) - 1)), dtype=np.int64
        ).reshape(-1, 1),
        "poses_c2w": np.asarray(data.get("poses", np.zeros((0, 4, 4))), dtype=np.float32),
        "intrinsics": np.asarray(data.get("intrinsics", np.zeros((0, 4))), dtype=np.float32),
        "motion_world": motion_world,
        "predicted_next_points": predicted_next_points,
        "motion_valid": motion_valid.reshape(-1, 1),
        "method_name": method_name,
        "source_model": str(source_model),
    }
    if "static_map_points" in data:
        out["static_map_points"] = np.asarray(data["static_map_points"], dtype=np.float32)
        out["static_map_colors"] = np.asarray(data.get("static_map_colors", np.zeros((0, 3))), dtype=np.uint8)
        out["static_map_obs"] = np.asarray(data.get("static_map_obs", np.zeros((0,))), dtype=np.int32).reshape(-1, 1)
    return out


def save_method(
    out_root: Path,
    method_dir: str,
    data: dict[str, np.ndarray],
    mode: str,
    source_model: Path,
    write_npz: bool,
) -> dict[str, int | float | str]:
    out_dir = out_root / method_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mdict = matlab_dict(data, mode=mode, source_model=source_model)
    savemat(out_dir / "model_4d.mat", mdict, do_compression=True)

    if write_npz:
        np.savez_compressed(
            out_dir / "model_4d.npz",
            points=mdict["points"],
            colors=mdict["colors"],
            dynamic=np.asarray(mdict["dynamic"]).reshape(-1).astype(bool),
            uncertainty=np.asarray(mdict["uncertainty"]).reshape(-1).astype(np.float32),
            motion_world=mdict["motion_world"],
            predicted_next_points=mdict["predicted_next_points"],
            motion_valid=np.asarray(mdict["motion_valid"]).reshape(-1).astype(bool),
            time=np.asarray(mdict["time"]).reshape(-1).astype(np.float32),
            timestamps=np.asarray(mdict["timestamps"]).reshape(-1).astype(np.float32),
            frame_offsets=np.asarray(mdict["frame_offsets0"]).reshape(-1).astype(np.int64),
            frame_counts=np.asarray(mdict["frame_counts"]).reshape(-1).astype(np.int64),
            frame_dynamic_counts=np.asarray(mdict["frame_dynamic_counts"]).reshape(-1).astype(np.int64),
            poses=np.asarray(mdict["poses_c2w"]).astype(np.float32),
            intrinsics=np.asarray(mdict["intrinsics"]).astype(np.float32),
            static_map_points=np.asarray(mdict.get("static_map_points", np.zeros((0, 3))), dtype=np.float32),
            static_map_colors=np.asarray(mdict.get("static_map_colors", np.zeros((0, 3))), dtype=np.uint8),
            static_map_obs=np.asarray(mdict.get("static_map_obs", np.zeros((0,))), dtype=np.int32).reshape(-1),
        )

    motion_valid = np.asarray(mdict["motion_valid"]).reshape(-1).astype(bool)
    dynamic = np.asarray(mdict["dynamic"]).reshape(-1).astype(bool)
    motion = np.asarray(mdict["motion_world"], dtype=np.float32)
    active = dynamic & motion_valid
    mag = np.linalg.norm(motion[active], axis=1) if np.any(active) else np.zeros((0,), dtype=np.float32)
    meta = {
        "method_dir": method_dir,
        "mode": mode,
        "source_model": str(source_model),
        "model_mat": str(out_dir / "model_4d.mat"),
        "model_npz": str(out_dir / "model_4d.npz") if write_npz else "",
        "n_points": int(len(mdict["points"])),
        "n_frames": int(len(np.asarray(mdict["frame_counts"]).reshape(-1))),
        "n_dynamic_points": int(dynamic.sum()),
        "n_motion_valid_points": int(active.sum()),
        "motion_mean_m": float(mag.mean()) if mag.size else 0.0,
        "motion_p90_m": float(np.percentile(mag, 90)) if mag.size else 0.0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def write_guides(out_root: Path, metas: list[dict[str, int | float | str]], repo_root: Path) -> None:
    rows = "\n".join(
        [
            f"| `{m['method_dir']}` | `{m['model_mat']}` | {m['n_dynamic_points']} | "
            f"{m['n_motion_valid_points']} | {float(m['motion_mean_m']):.6f} |"
            for m in metas
        ]
    )
    text = f"""# MATLAB 4D Model Comparison

This folder contains three MATLAB-ready 4D world models for the same Bonn balloon sequence.

| Method | MATLAB file | Dynamic points | Motion-valid points | Mean motion |
|---|---|---:|---:|---:|
{rows}

## What The Three Models Mean

- `baseline_static`: no dynamic prediction. Dynamic points are displayed at their current-frame positions, and predicted next positions are identical to current positions.
- `kalman_previous`: the previous point-level/Kalman-style dynamic motion result.
- `learned_region`: the learned pixel-motion model with dynamic-region smoothing. This is the current best dynamic-object 4D mapping output.

## MATLAB Display

From the repository root in MATLAB:

```matlab
addpath('scripts_eval');
view_4d_model_matlab('Outputs/Bonn/bonn_balloon/matlab_4d_comparison/baseline_static/model_4d.mat', 8, true, 'none', true);
view_4d_model_matlab('Outputs/Bonn/bonn_balloon/matlab_4d_comparison/kalman_previous/model_4d.mat', 8, true, 'none', true);
view_4d_model_matlab('Outputs/Bonn/bonn_balloon/matlab_4d_comparison/learned_region/model_4d.mat', 8, true, 'none', true);
```

For synchronized side-by-side playback:

```matlab
addpath('scripts_eval');
view_4d_comparison_matlab('Outputs/Bonn/bonn_balloon/matlab_4d_comparison', 8, 'none', true);
```

Color convention in the viewers:

- normal RGB: reconstructed points
- red: dynamic points
- cyan: predicted next-frame dynamic positions
- green: observed dynamic points in the next frame

## How To Generalize To Another Dataset

1. Run SLAM and keep the output `video.npz`.
2. Export the previous/Kalman-style 4D model with `scripts_eval/export_4d_model.py`.
3. Apply the trained pixel-motion model to the same `video.npz` with `scripts_train/apply_pixel_motion_to_video.py`.
4. Optionally smooth learned motion regions with `scripts_train/smooth_video_motion_regions.py`.
5. Export the learned 4D model with `scripts_eval/export_4d_model.py`.
6. Package MATLAB comparison files with `scripts_eval/package_matlab_4d_comparison.py`.
7. Run `scripts_eval/compare_4d_methods.py` to produce CSV/JSON/PNG metrics.

Template commands:

```bash
python scripts_eval/export_4d_model.py \\
  --video-npz Outputs/<DATASET>/<SEQ>/video.npz \\
  --output-dir Outputs/<DATASET>/<SEQ>/4d_model_previous_motion \\
  --disp-source up --structure-mode --no-ply --no-tracks

python scripts_train/apply_pixel_motion_to_video.py \\
  --video-npz Outputs/<DATASET>/<SEQ>/video.npz \\
  --checkpoint checkpoints/pixel_motion_bonn_dynamic/best.pt \\
  --out-video Outputs/<DATASET>/<SEQ>/video_learned_motion.npz \\
  --device cpu

python scripts_train/smooth_video_motion_regions.py \\
  --video-npz Outputs/<DATASET>/<SEQ>/video_learned_motion.npz \\
  --out-video Outputs/<DATASET>/<SEQ>/video_learned_motion_region.npz

python scripts_eval/export_4d_model.py \\
  --video-npz Outputs/<DATASET>/<SEQ>/video_learned_motion_region.npz \\
  --output-dir Outputs/<DATASET>/<SEQ>/4d_model_learned_motion_region \\
  --disp-source up --structure-mode --no-ply --no-tracks

python scripts_eval/package_matlab_4d_comparison.py \\
  --kalman-model Outputs/<DATASET>/<SEQ>/4d_model_previous_motion/model_4d.npz \\
  --learned-model Outputs/<DATASET>/<SEQ>/4d_model_learned_motion_region/model_4d.npz \\
  --output-dir Outputs/<DATASET>/<SEQ>/matlab_4d_comparison

python scripts_eval/compare_4d_methods.py \\
  --old-model Outputs/<DATASET>/<SEQ>/4d_model_previous_motion/model_4d.npz \\
  --new-model Outputs/<DATASET>/<SEQ>/4d_model_learned_motion_region/model_4d.npz \\
  --old-name KalmanPrevious \\
  --new-name LearnedRegion \\
  --out-prefix Outputs/<DATASET>/<SEQ>/4d_model_learned_motion_region/experiment_baseline_kalman_learned
```

## Notes

The static scene is already strong in the current pipeline. The purpose of these files is to isolate the dynamic-object part:

- baseline tests what happens if dynamic points are not predicted;
- Kalman tests the older point-level prediction;
- learned-region tests whether the learned model gives better dynamic-object temporal consistency.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")

    viewer_src = repo_root / "scripts_eval" / "view_4d_comparison_matlab.m"
    if viewer_src.exists():
        shutil.copy2(viewer_src, out_root / "view_4d_comparison_matlab.m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package baseline, Kalman, and learned 4D models for MATLAB.")
    parser.add_argument("--kalman-model", required=True, help="Previous/Kalman-style model_4d.npz")
    parser.add_argument("--learned-model", required=True, help="Learned-region model_4d.npz")
    parser.add_argument("--output-dir", required=True, help="Output folder for MATLAB comparison package")
    parser.add_argument("--write-npz", action="store_true", help="Also write baseline/converted npz files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    kalman_model = Path(args.kalman_model)
    learned_model = Path(args.learned_model)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    kalman_data = load_npz(kalman_model)
    learned_data = load_npz(learned_model)

    metas = [
        save_method(out_root, "baseline_static", kalman_data, "baseline", kalman_model, args.write_npz),
        save_method(out_root, "kalman_previous", kalman_data, "kalman_previous", kalman_model, args.write_npz),
        save_method(out_root, "learned_region", learned_data, "learned_region", learned_model, args.write_npz),
    ]
    (out_root / "manifest.json").write_text(json.dumps({"methods": metas}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_guides(out_root, metas, repo_root)

    print(f"Saved MATLAB comparison package: {out_root}")
    for meta in metas:
        print(
            f"{meta['method_dir']}: mat={meta['model_mat']} "
            f"dynamic={meta['n_dynamic_points']} motion_valid={meta['n_motion_valid_points']}"
        )


if __name__ == "__main__":
    main()
