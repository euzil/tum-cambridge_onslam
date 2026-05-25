from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def flip_z_xyz(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr).copy()
    if out.ndim >= 2 and out.shape[-1] == 3:
        out[..., 2] *= -1
    return out


def transform_pose_c2w_flip_z(poses: np.ndarray) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return poses
    F = np.diag([1.0, 1.0, -1.0, 1.0]).astype(poses.dtype)
    return np.einsum("ab,tbc,cd->tad", F, poses, F)


def transform_npz(src: Path, dst: Path) -> None:
    data = dict(np.load(src, allow_pickle=True))
    for k in ["points", "predicted_next_points", "motion_world", "static_map_points"]:
        if k in data:
            data[k] = flip_z_xyz(data[k])
    if "poses" in data:
        data["poses"] = transform_pose_c2w_flip_z(np.asarray(data["poses"]))
    np.savez_compressed(dst, **data)


def transform_mat(src: Path, dst: Path) -> None:
    S = loadmat(src)
    for k in ["points", "predicted_next_points", "motion_world", "static_map_points"]:
        if k in S:
            S[k] = flip_z_xyz(S[k])
    if "poses_c2w" in S:
        S["poses_c2w"] = transform_pose_c2w_flip_z(np.asarray(S["poses_c2w"]))
    savemat(dst, {k: v for k, v in S.items() if not k.startswith("__")}, do_compression=True)


def transform_tracks_npz(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    data = dict(np.load(src, allow_pickle=True))
    if "tracks" in data and data["tracks"].ndim == 3 and data["tracks"].shape[-1] == 3:
        data["tracks"] = flip_z_xyz(data["tracks"])
    np.savez_compressed(dst, **data)


def transform_tracks_mat(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    S = loadmat(src)
    if "tracks" in S:
        S["tracks"] = flip_z_xyz(S["tracks"])
    savemat(dst, {k: v for k, v in S.items() if not k.startswith("__")}, do_compression=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flip Z axis for exported 4D model files.")
    ap.add_argument("--input-dir", required=True, help="Folder containing model_4d.mat/.npz")
    ap.add_argument("--output-dir", required=True, help="Output folder")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_npz(in_dir / "model_4d.npz", out_dir / "model_4d.npz")
    transform_mat(in_dir / "model_4d.mat", out_dir / "model_4d.mat")
    transform_tracks_npz(in_dir / "dynamic_tracks.npz", out_dir / "dynamic_tracks.npz")
    transform_tracks_mat(in_dir / "dynamic_tracks.mat", out_dir / "dynamic_tracks.mat")

    meta_in = in_dir / "metadata.json"
    if meta_in.exists():
        meta = json.loads(meta_in.read_text(encoding="utf-8"))
        meta["axis_transform"] = "flip_z"
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[axis] wrote transformed model to: {out_dir}")


if __name__ == "__main__":
    main()
