from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose per-axis 4D prediction error in model_4d.npz.")
    parser.add_argument("--model-npz", required=True)
    parser.add_argument("--max-nn-dist", type=float, default=0.5)
    args = parser.parse_args()

    data = np.load(Path(args.model_npz))
    required = ["points", "dynamic", "motion_valid", "predicted_next_points", "frame_offsets"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"{args.model_npz} is missing {missing}")

    points = np.asarray(data["points"], dtype=np.float32)
    predicted = np.asarray(data["predicted_next_points"], dtype=np.float32)
    dynamic = np.asarray(data["dynamic"]).astype(bool)
    motion_valid = np.asarray(data["motion_valid"]).astype(bool)
    offsets = np.asarray(data["frame_offsets"], dtype=np.int64)

    signed_errors = []
    dists = []
    for t in range(len(offsets) - 2):
        lo, hi = offsets[t], offsets[t + 1]
        nlo, nhi = offsets[t + 1], offsets[t + 2]
        src_mask = dynamic[lo:hi] & motion_valid[lo:hi]
        tgt_mask = dynamic[nlo:nhi]
        if not np.any(src_mask) or not np.any(tgt_mask):
            continue
        pred_pts = predicted[lo:hi][src_mask]
        tgt_pts = points[nlo:nhi][tgt_mask]
        tree = cKDTree(tgt_pts)
        dist, idx = tree.query(pred_pts, k=1)
        keep = dist <= args.max_nn_dist
        if not np.any(keep):
            continue
        matched = tgt_pts[idx[keep]]
        err = pred_pts[keep] - matched
        signed_errors.append(err)
        dists.append(dist[keep])

    if not signed_errors:
        print("No matched predicted dynamic points found.")
        return

    err = np.concatenate(signed_errors, axis=0)
    dist = np.concatenate(dists, axis=0)
    abs_err = np.abs(err)
    axis_names = ["X", "Y", "Z"]

    print(f"model: {args.model_npz}")
    print(f"matched predicted points: {len(err)}")
    print(f"NN distance mean={dist.mean():.6f} median={np.median(dist):.6f} p90={np.percentile(dist, 90):.6f} m")
    for i, name in enumerate(axis_names):
        print(
            f"{name}: signed_mean={err[:, i].mean():+.6f} "
            f"signed_median={np.median(err[:, i]):+.6f} "
            f"abs_mean={abs_err[:, i].mean():.6f} "
            f"abs_p90={np.percentile(abs_err[:, i], 90):.6f} m"
        )


if __name__ == "__main__":
    main()
