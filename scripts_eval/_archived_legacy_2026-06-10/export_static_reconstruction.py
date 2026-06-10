from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

try:
    import cv2
except ImportError:  # pragma: no cover - scipy fallback is used on minimal installs
    cv2 = None
    from scipy.ndimage import zoom


def resize_to(arr: np.ndarray, shape: tuple[int, int], linear: bool = True) -> np.ndarray:
    """Resize a 2D array to (height, width)."""
    if arr.shape == shape:
        return arr
    if cv2 is not None:
        interp = cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST
        return cv2.resize(arr, (shape[1], shape[0]), interpolation=interp)

    scale_y = shape[0] / arr.shape[0]
    scale_x = shape[1] / arr.shape[1]
    order = 1 if linear else 0
    return zoom(arr, (scale_y, scale_x), order=order)


def load_video(video_npz: Path) -> dict[str, np.ndarray]:
    if not video_npz.exists():
        raise FileNotFoundError(f"Missing video.npz: {video_npz}")
    data = np.load(video_npz, allow_pickle=True)
    required = ["images", "poses", "intrinsics", "uncertainties"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{video_npz} is missing required fields: {missing}")

    if "droid_disps_up" in data:
        disps = np.asarray(data["droid_disps_up"], dtype=np.float32)
        disp_source = "droid_disps_up"
    elif "droid_disps" in data:
        disps = np.asarray(data["droid_disps"], dtype=np.float32)
        disp_source = "droid_disps"
    elif "disps" in data:
        disps = np.asarray(data["disps"], dtype=np.float32)
        disp_source = "disps"
    else:
        raise KeyError(f"{video_npz} does not contain droid_disps_up/droid_disps/disps")

    motion_masks = (
        np.asarray(data["dynamic_motion_masks"], dtype=np.float32)
        if "dynamic_motion_masks" in data
        else None
    )
    timestamps = np.asarray(data["timestamps"] if "timestamps" in data else np.arange(len(disps)))
    return {
        "images": np.asarray(data["images"], dtype=np.float32),
        "poses": np.asarray(data["poses"], dtype=np.float32),
        "disps": disps,
        "disp_source": disp_source,
        "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32),
        "uncertainties": np.asarray(data["uncertainties"], dtype=np.float32),
        "dynamic_motion_masks": motion_masks,
        "timestamps": timestamps,
    }


def scaled_intrinsics(intr: np.ndarray, from_shape: tuple[int, int], to_shape: tuple[int, int]) -> np.ndarray:
    fx, fy, cx, cy = intr.astype(np.float32)
    sx = to_shape[1] / from_shape[1]
    sy = to_shape[0] / from_shape[0]
    return np.array([fx * sx, fy * sy, cx * sx, cy * sy], dtype=np.float32)


def frame_static_points(
    image: np.ndarray,
    c2w: np.ndarray,
    disp: np.ndarray,
    intr: np.ndarray,
    uncertainty: np.ndarray,
    motion_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one frame's static pixels into world coordinates."""
    _, img_h, img_w = image.shape
    disp_h, disp_w = disp.shape
    out_shape = (disp_h, disp_w)

    if image.shape[-2:] != out_shape:
        image_for_depth = np.empty((3, disp_h, disp_w), dtype=np.float32)
        for c in range(3):
            image_for_depth[c] = resize_to(image[c], out_shape, linear=True)
    else:
        image_for_depth = image

    intr_d = intr
    if image.shape[-2:] != out_shape:
        intr_d = scaled_intrinsics(intr, image.shape[-2:], out_shape)

    uncer = resize_to(uncertainty, out_shape, linear=True)
    dyn_from_motion = np.zeros(out_shape, dtype=bool)
    if motion_mask is not None:
        dyn_from_motion = resize_to(motion_mask, out_shape, linear=False) > args.motion_mask_thresh

    valid = disp > 1e-6
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[valid] = 1.0 / disp[valid]
    valid &= depth > args.min_depth
    if args.max_depth > 0:
        valid &= depth <= args.max_depth
    if args.uncer_thresh > 0:
        valid &= uncer <= args.uncer_thresh
    valid &= ~dyn_from_motion

    if args.pixel_stride > 1:
        sample = np.zeros_like(valid)
        sample[:: args.pixel_stride, :: args.pixel_stride] = True
        valid &= sample

    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    z = depth[ys, xs]
    fx, fy, cx, cy = intr_d
    pts_cam = np.stack(
        [
            (xs.astype(np.float32) - cx) / fx * z,
            (ys.astype(np.float32) - cy) / fy * z,
            z,
        ],
        axis=-1,
    )
    pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3][None, :]
    colors = np.clip(image_for_depth.transpose(1, 2, 0)[ys, xs], 0.0, 1.0)

    # Lower uncertainty and closer surfaces get slightly stronger votes.
    weights = 1.0 / (1.0 + np.maximum(uncer[ys, xs], 0.0))
    weights *= 1.0 / np.maximum(z, 0.25)
    return pts_world.astype(np.float32), colors.astype(np.float32), weights.astype(np.float32)


def voxel_fuse(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0 or voxel_size <= 0:
        return points, colors, weights

    ijk = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(ijk, axis=0, return_inverse=True)
    n = int(inverse.max()) + 1
    wsum = np.bincount(inverse, weights=weights, minlength=n).astype(np.float64)
    wsum_safe = np.maximum(wsum, 1e-12)

    pts_out = np.empty((n, 3), dtype=np.float32)
    rgb_out = np.empty((n, 3), dtype=np.float32)
    for d in range(3):
        pts_out[:, d] = np.bincount(inverse, weights=points[:, d] * weights, minlength=n) / wsum_safe
        rgb_out[:, d] = np.bincount(inverse, weights=colors[:, d] * weights, minlength=n) / wsum_safe
    return pts_out, rgb_out, wsum.astype(np.float32)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def export_static_reconstruction(args: argparse.Namespace) -> None:
    video_npz = Path(args.video_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video = load_video(video_npz)
    images = video["images"]
    poses = video["poses"]
    disps = video["disps"]
    intrinsics = video["intrinsics"]
    uncertainties = video["uncertainties"]
    motion_masks = video["dynamic_motion_masks"]

    frame_points = []
    frame_colors = []
    frame_weights = []
    used_frames = 0
    raw_points = 0

    n_frames = min(len(disps), len(images), len(poses))
    for t in range(0, n_frames, args.frame_stride):
        motion_mask = motion_masks[t] if motion_masks is not None and t < len(motion_masks) else None
        pts, rgb, weights = frame_static_points(
            images[t],
            poses[t],
            disps[t],
            intrinsics[t],
            uncertainties[t],
            motion_mask,
            args,
        )
        raw_points += len(pts)
        if len(pts) > 0:
            pts, rgb, weights = voxel_fuse(pts, rgb, weights, args.per_frame_voxel)
            frame_points.append(pts)
            frame_colors.append(rgb)
            frame_weights.append(weights)
            used_frames += 1

        print(
            f"\r[static] frame {t + 1:04d}/{n_frames} "
            f"raw={raw_points:9d} kept_frame={len(pts):7d}",
            end="",
            flush=True,
        )
    print()

    if frame_points:
        points = np.concatenate(frame_points, axis=0)
        colors = np.concatenate(frame_colors, axis=0)
        weights = np.concatenate(frame_weights, axis=0)
        points, colors, weights = voxel_fuse(points, colors, weights, args.voxel_size)
        if args.min_voxel_weight > 0:
            keep = weights >= args.min_voxel_weight
            points, colors, weights = points[keep], colors[keep], weights[keep]
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)
        weights = np.zeros((0,), dtype=np.float32)

    ply_path = out_dir / "static_reconstruction.ply"
    mat_path = out_dir / "static_reconstruction.mat"
    write_ply(ply_path, points, colors)
    savemat(
        mat_path,
        {
            "points": points.astype(np.float32),
            "colors": np.clip(colors * 255.0, 0, 255).astype(np.uint8),
            "colors01": colors.astype(np.float32),
            "voxel_weight": weights.astype(np.float32).reshape(-1, 1),
            "poses_c2w": poses.astype(np.float32),
            "intrinsics": intrinsics.astype(np.float32),
            "timestamps": video["timestamps"].astype(np.float32).reshape(-1, 1),
        },
        do_compression=True,
    )

    metadata = {
        "source_video": str(video_npz),
        "disp_source": video["disp_source"],
        "n_frames_total": int(n_frames),
        "n_frames_used": int(used_frames),
        "raw_static_samples": int(raw_points),
        "n_points": int(len(points)),
        "voxel_size": float(args.voxel_size),
        "per_frame_voxel": float(args.per_frame_voxel),
        "pixel_stride": int(args.pixel_stride),
        "frame_stride": int(args.frame_stride),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "uncer_thresh": float(args.uncer_thresh),
        "motion_mask_thresh": float(args.motion_mask_thresh),
        "min_voxel_weight": float(args.min_voxel_weight),
        "outputs": {
            "mat": mat_path.name,
            "ply": ply_path.name,
        },
    }
    (out_dir / "static_reconstruction_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[static] Saved MATLAB reconstruction: {mat_path}")
    print(f"[static] Saved PLY reconstruction: {ply_path}")
    print(f"[static] Final static points: {len(points)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a dense static 3D reconstruction from DROID-W video.npz."
    )
    parser.add_argument("--video-npz", required=True, help="Path to SLAM output video.npz")
    parser.add_argument("--output-dir", required=True, help="Directory for static_reconstruction.mat/.ply")
    parser.add_argument("--voxel-size", type=float, default=0.015, help="Final fusion voxel size in meters")
    parser.add_argument("--per-frame-voxel", type=float, default=0.01, help="Per-frame pre-fusion voxel size")
    parser.add_argument("--pixel-stride", type=int, default=1, help="Use every Nth depth pixel")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth keyframe")
    parser.add_argument("--min-depth", type=float, default=0.05, help="Drop points closer than this")
    parser.add_argument("--max-depth", type=float, default=8.0, help="Drop points deeper than this; <=0 disables")
    parser.add_argument("--uncer-thresh", type=float, default=0.8, help="Keep uncertainty <= threshold; <=0 disables")
    parser.add_argument("--motion-mask-thresh", type=float, default=0.0, help="Dynamic motion mask threshold")
    parser.add_argument("--min-voxel-weight", type=float, default=0.0, help="Drop final voxels below this vote weight")
    return parser.parse_args()


if __name__ == "__main__":
    export_static_reconstruction(parse_args())
