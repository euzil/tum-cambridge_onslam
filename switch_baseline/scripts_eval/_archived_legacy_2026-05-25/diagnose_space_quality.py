from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_video(video_npz: Path) -> dict[str, np.ndarray]:
    data = np.load(video_npz, allow_pickle=True)
    return {
        "poses": np.asarray(data["poses"], dtype=np.float32),
        "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32),
        "disps_low": np.asarray(data["droid_disps"], dtype=np.float32) if "droid_disps" in data else None,
        "disps_up": np.asarray(data["droid_disps_up"], dtype=np.float32) if "droid_disps_up" in data else None,
        "mono_disps": np.asarray(data["mono_disps"], dtype=np.float32) if "mono_disps" in data else None,
        "uncertainties": np.asarray(data["uncertainties"], dtype=np.float32) if "uncertainties" in data else None,
    }


def make_points_world(
    disp: np.ndarray,
    pose_c2w: np.ndarray,
    intr: np.ndarray,
    uncertainty: np.ndarray | None,
    stride: int,
    max_depth: float,
    uncer_q: float,
) -> np.ndarray:
    h, w = disp.shape
    ys, xs = np.meshgrid(
        np.arange(0, h, stride, dtype=np.float32),
        np.arange(0, w, stride, dtype=np.float32),
        indexing="ij",
    )
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    d = disp[ys.astype(int), xs.astype(int)]
    valid = d > 1e-6
    depth = np.zeros_like(d, dtype=np.float32)
    depth[valid] = 1.0 / d[valid]
    valid &= depth > 0
    if max_depth > 0:
        valid &= depth <= max_depth

    if uncertainty is not None and uncertainty.size > 0 and uncer_q > 0:
        hu, wu = uncertainty.shape
        yu = np.clip(np.round(ys * (hu / h)).astype(int), 0, hu - 1)
        xu = np.clip(np.round(xs * (wu / w)).astype(int), 0, wu - 1)
        u = uncertainty[yu, xu]
        thr = np.quantile(u[valid], min(max(uncer_q, 0.0), 1.0)) if np.any(valid) else np.inf
        valid &= u <= thr

    xs = xs[valid]
    ys = ys[valid]
    depth = depth[valid]

    if len(depth) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy, cx, cy = intr
    # intrinsics are usually in low-res coordinates (e.g. 64x48), rescale to disp size
    h_ref, w_ref = (48.0, 64.0)
    sx = w / w_ref
    sy = h / h_ref
    fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy

    pts_cam = np.stack(
        [
            (xs - cx) / fx * depth,
            (ys - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    ).astype(np.float32)

    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t[None, :]
    return pts_world.astype(np.float32)


def nn_error(a: np.ndarray, b: np.ndarray, max_samples: int = 5000) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) > max_samples:
        idx = np.random.choice(len(a), max_samples, replace=False)
        a = a[idx]
    if len(b) > max_samples:
        idx = np.random.choice(len(b), max_samples, replace=False)
        b = b[idx]
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return float(np.mean(d))


def evaluate_source(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    disps: np.ndarray,
    uncertainties: np.ndarray | None,
    stride: int,
    max_depth: float,
    uncer_q: float,
) -> dict[str, float]:
    n = len(disps)
    frame_pts: list[np.ndarray] = []
    for t in range(n):
        u = uncertainties[t] if uncertainties is not None else None
        pts = make_points_world(disps[t], poses[t], intrinsics[t], u, stride, max_depth, uncer_q)
        frame_pts.append(pts)

    errs = []
    for t in range(n - 1):
        e1 = nn_error(frame_pts[t], frame_pts[t + 1])
        e2 = nn_error(frame_pts[t + 1], frame_pts[t])
        if np.isfinite(e1) and np.isfinite(e2):
            errs.append(0.5 * (e1 + e2))

    counts = np.array([len(p) for p in frame_pts], dtype=np.float64)
    return {
        "mean_points_per_frame": float(np.mean(counts)) if len(counts) else 0.0,
        "median_points_per_frame": float(np.median(counts)) if len(counts) else 0.0,
        "temporal_nn_error_m": float(np.mean(errs)) if len(errs) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose whether video->4D geometry is space-like or collapsed.")
    ap.add_argument("--video-npz", required=True)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--uncer-quantile", type=float, default=0.9)
    args = ap.parse_args()

    data = load_video(Path(args.video_npz))
    poses = data["poses"]
    intr = data["intrinsics"]
    T = poses[:, :3, 3]
    span = T.max(axis=0) - T.min(axis=0)
    print("== Pose span (meters) ==")
    print(f"x={span[0]:.4f}, y={span[1]:.4f}, z={span[2]:.4f}, norm={np.linalg.norm(span):.4f}")

    candidates = {
        "droid_disps_up": data["disps_up"],
        "droid_disps_low": data["disps_low"],
        "mono_disps": data["mono_disps"],
    }
    print("\n== Source quality ==")
    for name, disps in candidates.items():
        if disps is None:
            continue
        stat = evaluate_source(
            poses=poses,
            intrinsics=intr,
            disps=disps,
            uncertainties=data["uncertainties"],
            stride=max(args.stride, 1),
            max_depth=args.max_depth,
            uncer_q=args.uncer_quantile,
        )
        print(
            f"{name:16s} | points/frame mean={stat['mean_points_per_frame']:.1f} "
            f"median={stat['median_points_per_frame']:.1f} | temporal_nn_error={stat['temporal_nn_error_m']:.4f} m"
        )

    print("\nLower temporal_nn_error usually means better cross-frame geometry consistency.")


if __name__ == "__main__":
    main()
