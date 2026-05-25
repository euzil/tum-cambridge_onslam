from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import whosmat


def main() -> None:
    parser = argparse.ArgumentParser(description="Check consistency between video.npz and exported 4D model files.")
    parser.add_argument("--video-npz", required=True)
    parser.add_argument("--model-npz", required=True)
    parser.add_argument("--model-mat", default="")
    parser.add_argument("--metadata", default="")
    args = parser.parse_args()

    video_path = Path(args.video_npz)
    model_path = Path(args.model_npz)
    mat_path = Path(args.model_mat) if args.model_mat else model_path.with_suffix(".mat")
    meta_path = Path(args.metadata) if args.metadata else model_path.parent / "metadata.json"

    video = np.load(video_path)
    model = np.load(model_path)

    print(f"video : {video_path}")
    print(f"model : {model_path}")
    print(f"mat   : {mat_path if mat_path.exists() else 'missing'}")
    print(f"meta  : {meta_path if meta_path.exists() else 'missing'}")
    print()

    print("video fields:")
    for key in video.files:
        arr = video[key]
        print(f"  {key:24s} {str(arr.shape):18s} {arr.dtype}")
    print()

    print("model fields:")
    for key in model.files:
        arr = model[key]
        print(f"  {key:24s} {str(arr.shape):18s} {arr.dtype}")
    print()

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("metadata source_video:", meta.get("source_video"))
        print("metadata disp_source :", meta.get("disp_source"))
        print("metadata n_frames    :", meta.get("n_frames"))
        print("metadata n_points    :", meta.get("n_points_total"))
        print("metadata n_motion    :", meta.get("n_motion_valid_points_total"))
        print()

    if mat_path.exists():
        print("MAT variables:")
        for name, shape, dtype in whosmat(mat_path):
            print(f"  {name:24s} {str(shape):18s} {dtype}")
        print()

    checks = []
    if "timestamps" in video.files and "timestamps" in model.files:
        checks.append(("timestamps equal", np.array_equal(video["timestamps"], model["timestamps"])))
    if "poses" in video.files and "poses" in model.files:
        checks.append(("poses equal", np.allclose(video["poses"], model["poses"], atol=1e-6)))
    if "frame_offsets" in model.files and "points" in model.files:
        offsets = model["frame_offsets"]
        checks.append(("frame_offsets length ok", len(offsets) == len(model["timestamps"]) + 1))
        checks.append(("last offset == n_points", int(offsets[-1]) == len(model["points"])))
    if {"points", "motion_world", "predicted_next_points"}.issubset(model.files):
        diff = model["predicted_next_points"] - (model["points"] + model["motion_world"])
        checks.append(("predicted = points + motion", float(np.max(np.abs(diff))) < 1e-5))
        print("predicted consistency max_abs_diff:", float(np.max(np.abs(diff))))
    if {"motion_valid", "dynamic"}.issubset(model.files):
        mv = model["motion_valid"].astype(bool)
        dyn = model["dynamic"].astype(bool)
        print("motion_valid count:", int(mv.sum()))
        print("dynamic count     :", int(dyn.sum()))
        print("motion_valid but not dynamic:", int((mv & ~dyn).sum()))

    print()
    print("checks:")
    for name, ok in checks:
        print(f"  {name:32s}: {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
