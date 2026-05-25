from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_video(video_npz: Path) -> dict[str, np.ndarray]:
    data = np.load(video_npz)
    required = [
        "poses",
        "droid_disps",
        "intrinsics",
        "uncertainties",
        "dynamic_motions",
        "dynamic_motion_masks",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{video_npz} is missing {missing}. Rerun SLAM with dynamic motion export enabled.")
    return {key: np.asarray(data[key]) for key in required}


def frame_points(
    pose_c2w: np.ndarray,
    disp: np.ndarray,
    intr: np.ndarray,
    uncertainty: np.ndarray,
    motion_mask: np.ndarray,
    uncer_thresh: float,
    max_depth: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = disp.shape
    ys, xs = np.meshgrid(
        np.arange(0, h, stride, dtype=np.int32),
        np.arange(0, w, stride, dtype=np.int32),
        indexing="ij",
    )
    ys = ys.reshape(-1)
    xs = xs.reshape(-1)

    disp_v = disp[ys, xs].astype(np.float32)
    depth = np.zeros_like(disp_v)
    valid = disp_v > 1e-6
    depth[valid] = 1.0 / disp_v[valid]
    valid &= depth > 0
    if max_depth > 0:
        valid &= depth <= max_depth

    dyn = (uncertainty[ys, xs] > uncer_thresh) | (motion_mask[ys, xs] > 0)
    valid &= dyn

    ys = ys[valid]
    xs = xs[valid]
    depth = depth[valid]
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), ys, xs

    fx, fy, cx, cy = intr
    pts_cam = np.stack(
        [
            (xs.astype(np.float32) - cx) / fx * depth,
            (ys.astype(np.float32) - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )
    pts_world = (pose_c2w[:3, :3] @ pts_cam.T).T + pose_c2w[:3, 3][None, :]
    return pts_world.astype(np.float32), ys, xs


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic 4D temporal consistency using CUDA BA motion variables."
    )
    parser.add_argument("--video-npz", required=True)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--uncer-thresh", type=float, default=0.9)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-nn-dist", type=float, default=0.5)
    args = parser.parse_args()

    video = load_video(Path(args.video_npz))
    poses = video["poses"].astype(np.float32)
    disps = video["droid_disps"].astype(np.float32)
    intrinsics = video["intrinsics"].astype(np.float32)
    uncertainties = video["uncertainties"].astype(np.float32)
    motions = video["dynamic_motions"].astype(np.float32)
    motion_masks = video["dynamic_motion_masks"].astype(np.float32)

    rows = []
    static_all: list[float] = []
    motion_all: list[float] = []
    reverse_all: list[float] = []

    for t in range(len(disps) - 1):
        src_pts, ys, xs = frame_points(
            poses[t],
            disps[t],
            intrinsics[t],
            uncertainties[t],
            motion_masks[t],
            args.uncer_thresh,
            args.max_depth,
            args.stride,
        )
        tgt_pts, _, _ = frame_points(
            poses[t + 1],
            disps[t + 1],
            intrinsics[t + 1],
            uncertainties[t + 1],
            motion_masks[t + 1],
            args.uncer_thresh,
            args.max_depth,
            args.stride,
        )
        if len(src_pts) == 0 or len(tgt_pts) == 0:
            continue

        motion_cam = motions[t, ys, xs]
        motion_world = (poses[t + 1][:3, :3] @ motion_cam.T).T
        pred_pts = src_pts + motion_world
        reverse_pts = src_pts - motion_world

        tree = cKDTree(tgt_pts)
        static_dist, _ = tree.query(src_pts, k=1)
        motion_dist, _ = tree.query(pred_pts, k=1)
        reverse_dist, _ = tree.query(reverse_pts, k=1)
        keep = (
            (static_dist <= args.max_nn_dist)
            | (motion_dist <= args.max_nn_dist)
            | (reverse_dist <= args.max_nn_dist)
        )
        if not np.any(keep):
            continue

        static_dist = static_dist[keep]
        motion_dist = motion_dist[keep]
        reverse_dist = reverse_dist[keep]
        static_all.extend(static_dist.tolist())
        motion_all.extend(motion_dist.tolist())
        reverse_all.extend(reverse_dist.tolist())

        rows.append(
            [
                t,
                len(static_dist),
                float(static_dist.mean()),
                float(np.median(static_dist)),
                float(motion_dist.mean()),
                float(np.median(motion_dist)),
                float(reverse_dist.mean()),
                float(np.median(reverse_dist)),
                float((static_dist - motion_dist).mean()),
            ]
        )

    static_s = summarize(static_all)
    motion_s = summarize(motion_all)
    reverse_s = summarize(reverse_all)
    gain = (static_s["mean"] - motion_s["mean"]) / max(static_s["mean"], 1e-8) * 100.0
    reverse_gain = (static_s["mean"] - reverse_s["mean"]) / max(static_s["mean"], 1e-8) * 100.0

    print(f"video: {args.video_npz}")
    print(f"pairs evaluated: {len(rows)}")
    print(
        "static carry-forward NN error: "
        f"mean={static_s['mean']:.6f} median={static_s['median']:.6f} p90={static_s['p90']:.6f} m"
    )
    print(
        "motion-compensated NN error: "
        f"mean={motion_s['mean']:.6f} median={motion_s['median']:.6f} p90={motion_s['p90']:.6f} m"
    )
    print(
        "reverse-motion NN error: "
        f"mean={reverse_s['mean']:.6f} median={reverse_s['median']:.6f} p90={reverse_s['p90']:.6f} m"
    )
    print(f"relative dynamic 4D consistency gain: {gain:+.2f}%")
    print(f"reverse-motion diagnostic gain: {reverse_gain:+.2f}%")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "frame,n_points,static_mean,static_median,motion_mean,motion_median,"
            "reverse_mean,reverse_median,mean_gain_m"
        )
        np.savetxt(out, np.asarray(rows, dtype=np.float32), delimiter=",", header=header, comments="")
        print(f"saved csv: {out}")


if __name__ == "__main__":
    main()
