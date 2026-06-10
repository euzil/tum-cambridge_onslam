from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def check_npz(model_npz: Path) -> dict[str, np.ndarray]:
    data = np.load(model_npz)
    print(f"[npz] {model_npz}")
    print("[npz] keys:", ", ".join(data.files))

    required = ["points", "colors", "dynamic", "motion_world", "predicted_next_points", "motion_valid", "frame_offsets"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing npz fields: {missing}")

    n = len(data["points"])
    print(f"[npz] points={data['points'].shape}, colors={data['colors'].shape}, frames={len(data['frame_offsets']) - 1}")
    for key in ["dynamic", "motion_valid", "uncertainty", "time", "frame_ids"]:
        if key in data.files:
            print(f"[npz] {key}: shape={data[key].shape}")
            if len(data[key]) != n:
                raise ValueError(f"{key} length {len(data[key])} != points length {n}")

    offsets = np.asarray(data["frame_offsets"], dtype=np.int64)
    if offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0):
        raise ValueError("frame_offsets are invalid")

    pred_expected = data["points"] + data["motion_world"]
    pred_err = np.linalg.norm(data["predicted_next_points"] - pred_expected, axis=1)
    print(f"[npz] predicted_next = points + motion_world max_err={pred_err.max():.9f}")
    print(f"[npz] dynamic count={int(np.asarray(data['dynamic']).astype(bool).sum())}")
    print(f"[npz] motion_valid count={int(np.asarray(data['motion_valid']).astype(bool).sum())}")
    return {k: data[k] for k in data.files}


def check_mat(model_mat: Path, npz_data: dict[str, np.ndarray] | None) -> None:
    if not model_mat.exists():
        print(f"[mat] missing: {model_mat}")
        return
    data = loadmat(model_mat)
    keys = sorted(k for k in data.keys() if not k.startswith("__"))
    print(f"[mat] {model_mat}")
    print("[mat] keys:", ", ".join(keys))

    for key in ["points", "colors", "dynamic", "motion_world", "predicted_next_points", "motion_valid", "frame_offsets0"]:
        if key in data:
            print(f"[mat] {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"[mat] missing {key}")

    if npz_data is not None and "points" in data:
        n = min(len(npz_data["points"]), len(data["points"]))
        point_diff = np.linalg.norm(npz_data["points"][:n] - data["points"][:n], axis=1)
        print(f"[mat-vs-npz] first {n} point max_diff={point_diff.max():.9f}")
        if "predicted_next_points" in data and "predicted_next_points" in npz_data:
            pred_diff = np.linalg.norm(npz_data["predicted_next_points"][:n] - data["predicted_next_points"][:n], axis=1)
            print(f"[mat-vs-npz] first {n} predicted max_diff={pred_diff.max():.9f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model_4d.npz/.mat consistency and prediction fields.")
    parser.add_argument("--model-npz", required=True)
    parser.add_argument("--model-mat", default="")
    args = parser.parse_args()

    npz_path = Path(args.model_npz)
    mat_path = Path(args.model_mat) if args.model_mat else npz_path.with_suffix(".mat")

    npz_data = check_npz(npz_path)
    check_mat(mat_path, npz_data)


if __name__ == "__main__":
    main()
