from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dynamic_prediction.d4rt_bridge import DROIDWBridge


def load_video(video_npz: Path) -> dict[str, np.ndarray]:
    if not video_npz.exists():
        raise FileNotFoundError(f"Missing video.npz: {video_npz}")
    data = np.load(video_npz, allow_pickle=True)
    required = ["images", "intrinsics", "uncertainties"]
    for key in required:
        if key not in data:
            raise KeyError(f"{video_npz} does not contain '{key}'")

    if "droid_disps" in data:
        disps = np.asarray(data["droid_disps"], dtype=np.float32)
    elif "disps" in data:
        disps = np.asarray(data["disps"], dtype=np.float32)
    else:
        raise KeyError(f"{video_npz} does not contain droid_disps/disps")

    poses = np.asarray(data["poses"], dtype=np.float32)
    timestamps = np.asarray(data["timestamps"] if "timestamps" in data else np.arange(len(poses)))
    return {
        "images": np.asarray(data["images"], dtype=np.float32),
        "poses": poses,
        "disps": disps,
        "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32),
        "uncertainties": np.asarray(data["uncertainties"], dtype=np.float32),
        "timestamps": timestamps,
        "dynamic_motions": np.asarray(data["dynamic_motions"], dtype=np.float32)
        if "dynamic_motions" in data
        else None,
        "dynamic_motion_masks": np.asarray(data["dynamic_motion_masks"], dtype=np.float32)
        if "dynamic_motion_masks" in data
        else None,
    }


def frame_to_points(
    image: np.ndarray,
    c2w: np.ndarray,
    disp: np.ndarray,
    intr: np.ndarray,
    uncertainty: np.ndarray,
    uncer_thresh: float,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world points, RGB colors, dynamic mask, and uncertainty values."""
    h_l, w_l = disp.shape
    _, h_img, w_img = image.shape

    ys, xs = np.meshgrid(
        np.arange(0, h_l, stride, dtype=np.float32),
        np.arange(0, w_l, stride, dtype=np.float32),
        indexing="ij",
    )
    xs_f = xs.reshape(-1)
    ys_f = ys.reshape(-1)
    disp_f = disp[ys_f.astype(int), xs_f.astype(int)]
    uncer_f = uncertainty[ys_f.astype(int), xs_f.astype(int)]

    valid = disp_f > 1e-6
    depth = np.zeros_like(disp_f, dtype=np.float32)
    depth[valid] = 1.0 / disp_f[valid]
    valid &= depth > 0.0
    if max_depth > 0:
        valid &= depth <= max_depth

    xs_f = xs_f[valid]
    ys_f = ys_f[valid]
    depth = depth[valid]
    uncer_f = uncer_f[valid]
    dynamic = uncer_f > uncer_thresh

    fx, fy, cx, cy = intr
    pts_cam = np.stack(
        [
            (xs_f - cx) / fx * depth,
            (ys_f - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    ).astype(np.float32)

    r_wc = c2w[:3, :3]
    t_wc = c2w[:3, 3]
    pts_world = (r_wc @ pts_cam.T).T + t_wc[None, :]

    scale_x = w_img / w_l
    scale_y = h_img / h_l
    img_x = np.clip(np.round(xs_f * scale_x).astype(int), 0, w_img - 1)
    img_y = np.clip(np.round(ys_f * scale_y).astype(int), 0, h_img - 1)
    colors = np.clip(image.transpose(1, 2, 0)[img_y, img_x] * 255.0, 0, 255).astype(np.uint8)

    pixel_ij = np.stack([ys_f, xs_f], axis=-1).astype(np.int32)
    return pts_world.astype(np.float32), colors, dynamic.astype(bool), uncer_f.astype(np.float32), pixel_ij, depth


def limit_points(
    points: np.ndarray,
    colors: np.ndarray,
    dynamic: np.ndarray,
    uncertainty: np.ndarray,
    pixel_ij: np.ndarray,
    depth: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors, dynamic, uncertainty, pixel_ij, depth

    dyn_idx = np.where(dynamic)[0]
    sta_idx = np.where(~dynamic)[0]
    n_dyn = min(len(dyn_idx), max_points // 2)
    n_sta = max_points - n_dyn
    if len(dyn_idx) > n_dyn:
        dyn_idx = rng.choice(dyn_idx, size=n_dyn, replace=False)
    if len(sta_idx) > n_sta:
        sta_idx = rng.choice(sta_idx, size=n_sta, replace=False)
    idx = np.concatenate([dyn_idx, sta_idx])
    idx.sort()
    return points[idx], colors[idx], dynamic[idx], uncertainty[idx], pixel_ij[idx], depth[idx]


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    dynamic: np.ndarray,
    uncertainty: np.ndarray,
    dynamic_color: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dynamic_color:
        out_colors = colors.copy()
        out_colors[dynamic] = np.array([255, 60, 40], dtype=np.uint8)
    else:
        out_colors = colors

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
        f.write("property uchar dynamic\n")
        f.write("property float uncertainty\n")
        f.write("end_header\n")
        for p, c, d, u in zip(points, out_colors, dynamic, uncertainty):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(d)} {float(u):.6f}\n"
            )


def export_dynamic_tracks(
    video_npz: Path,
    out_dir: Path,
    uncer_thresh: float,
    match_radius: float,
) -> dict[str, int]:
    bridge = DROIDWBridge(uncer_thresh=uncer_thresh, match_radius=match_radius)
    data = bridge.load(str(video_npz))
    frames, pids_list = bridge.extract_dynamic_points(data)
    if not frames:
        return {"n_tracks": 0, "n_observations": 0}

    all_ids = sorted({int(pid) for pids in pids_list for pid in pids})
    id_to_row = {pid: i for i, pid in enumerate(all_ids)}
    n_tracks = len(all_ids)
    n_frames = len(frames)
    tracks = np.full((n_tracks, n_frames, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((n_tracks, n_frames), dtype=bool)

    for t, (pts, pids) in enumerate(zip(frames, pids_list)):
        for pt, pid in zip(pts, pids):
            row = id_to_row[int(pid)]
            tracks[row, t] = pt
            visibility[row, t] = True

    point_ids = np.asarray(all_ids, dtype=np.int64)
    np.savez_compressed(
        out_dir / "dynamic_tracks.npz",
        tracks=tracks,
        visibility=visibility,
        point_ids=point_ids,
    )
    savemat(
        out_dir / "dynamic_tracks.mat",
        {
            "tracks": tracks,
            "visibility": visibility,
            "point_ids": point_ids + 1,
        },
        do_compression=True,
    )
    return {"n_tracks": n_tracks, "n_observations": int(visibility.sum())}


def save_matlab_model(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    dynamic: np.ndarray,
    uncertainty: np.ndarray,
    time: np.ndarray,
    timestamps: np.ndarray,
    frame_offsets: np.ndarray,
    frame_counts: np.ndarray,
    frame_dynamic_counts: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    motion_world: np.ndarray,
    predicted_next_points: np.ndarray,
    motion_valid: np.ndarray,
) -> None:
    """Save a MATLAB-friendly 4D model.

    MATLAB variables:
      points: Nx3 single world coordinates
      colors: Nx3 uint8 RGB
      dynamic: Nx1 logical
      uncertainty: Nx1 single
      time: Nx1 single source frame index/timestamp
      frame_start_1 / frame_end_1: 1-based inclusive ranges into points/colors
    """
    starts_1 = frame_offsets[:-1].astype(np.int64) + 1
    ends_1 = frame_offsets[1:].astype(np.int64)
    savemat(
        path,
        {
            "points": points.astype(np.float32),
            "colors": colors.astype(np.uint8),
            "dynamic": dynamic.reshape(-1, 1),
            "uncertainty": uncertainty.astype(np.float32).reshape(-1, 1),
            "time": time.astype(np.float32).reshape(-1, 1),
            "timestamps": timestamps.astype(np.float32).reshape(-1, 1),
            "frame_offsets0": frame_offsets.astype(np.int64).reshape(-1, 1),
            "frame_start_1": starts_1.reshape(-1, 1),
            "frame_end_1": ends_1.reshape(-1, 1),
            "frame_counts": frame_counts.astype(np.int64).reshape(-1, 1),
            "frame_dynamic_counts": frame_dynamic_counts.astype(np.int64).reshape(-1, 1),
            "poses_c2w": poses.astype(np.float32),
            "intrinsics": intrinsics.astype(np.float32),
            "motion_world": motion_world.astype(np.float32),
            "predicted_next_points": predicted_next_points.astype(np.float32),
            "motion_valid": motion_valid.reshape(-1, 1),
        },
        do_compression=True,
    )


def export_4d_model(args: argparse.Namespace) -> None:
    video_npz = Path(args.video_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "ply_frames"
    rng = np.random.default_rng(args.seed)

    video = load_video(video_npz)
    images = video["images"]
    poses = video["poses"]
    disps = video["disps"]
    intrinsics = video["intrinsics"]
    uncertainties = video["uncertainties"]
    timestamps = video["timestamps"]
    dynamic_motions = video["dynamic_motions"]
    dynamic_motion_masks = video["dynamic_motion_masks"]

    all_points = []
    all_colors = []
    all_dynamic = []
    all_uncertainty = []
    all_motion_world = []
    all_predicted_next = []
    all_motion_valid = []
    all_time = []
    frame_offsets = [0]
    frame_counts = []
    frame_dynamic_counts = []

    for t in range(len(disps)):
        pts, colors, dyn, uncer, pixel_ij, depth = frame_to_points(
            images[t],
            poses[t],
            disps[t],
            intrinsics[t],
            uncertainties[t],
            args.uncer_thresh,
            args.stride,
            args.max_depth,
        )
        pts, colors, dyn, uncer, pixel_ij, depth = limit_points(
            pts, colors, dyn, uncer, pixel_ij, depth, args.max_points_per_frame, rng
        )

        motion_world = np.zeros_like(pts, dtype=np.float32)
        motion_valid = np.zeros((len(pts),), dtype=bool)
        if dynamic_motions is not None and dynamic_motion_masks is not None and t + 1 < len(poses):
            yy = pixel_ij[:, 0]
            xx = pixel_ij[:, 1]
            mask_v = dynamic_motion_masks[t, yy, xx] > 0
            if np.any(mask_v):
                motion_cam = dynamic_motions[t, yy[mask_v], xx[mask_v]]
                motion_world[mask_v] = (poses[t + 1][:3, :3] @ motion_cam.T).T
                motion_valid[mask_v] = True
                dyn = dyn | mask_v
        predicted_next = pts + motion_world

        if args.write_ply:
            write_ply(
                ply_dir / f"frame_{t:04d}_ts_{int(timestamps[t]):06d}.ply",
                pts,
                colors,
                dyn,
                uncer,
                dynamic_color=args.dynamic_color,
            )

        all_points.append(pts)
        all_colors.append(colors)
        all_dynamic.append(dyn)
        all_uncertainty.append(uncer)
        all_motion_world.append(motion_world)
        all_predicted_next.append(predicted_next)
        all_motion_valid.append(motion_valid)
        all_time.append(np.full((len(pts),), float(timestamps[t]), dtype=np.float32))
        frame_counts.append(len(pts))
        frame_dynamic_counts.append(int(dyn.sum()))
        frame_offsets.append(frame_offsets[-1] + len(pts))

        print(
            f"\r[4D] frame {t + 1:04d}/{len(disps)} "
            f"points={len(pts):5d} dynamic={int(dyn.sum()):5d}",
            end="",
            flush=True,
        )
    print()

    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)
    dynamic = np.concatenate(all_dynamic, axis=0) if all_dynamic else np.zeros((0,), dtype=bool)
    uncertainty = (
        np.concatenate(all_uncertainty, axis=0) if all_uncertainty else np.zeros((0,), dtype=np.float32)
    )
    motion_world = (
        np.concatenate(all_motion_world, axis=0) if all_motion_world else np.zeros((0, 3), dtype=np.float32)
    )
    predicted_next_points = (
        np.concatenate(all_predicted_next, axis=0) if all_predicted_next else np.zeros((0, 3), dtype=np.float32)
    )
    motion_valid = (
        np.concatenate(all_motion_valid, axis=0) if all_motion_valid else np.zeros((0,), dtype=bool)
    )
    time = np.concatenate(all_time, axis=0) if all_time else np.zeros((0,), dtype=np.float32)

    frame_offsets_arr = np.asarray(frame_offsets, dtype=np.int64)
    frame_counts_arr = np.asarray(frame_counts, dtype=np.int64)
    frame_dynamic_counts_arr = np.asarray(frame_dynamic_counts, dtype=np.int64)

    np.savez_compressed(
        out_dir / "model_4d.npz",
        points=points,
        colors=colors,
        dynamic=dynamic,
        uncertainty=uncertainty,
        motion_world=motion_world,
        predicted_next_points=predicted_next_points,
        motion_valid=motion_valid,
        time=time,
        timestamps=timestamps,
        frame_offsets=frame_offsets_arr,
        frame_counts=frame_counts_arr,
        frame_dynamic_counts=frame_dynamic_counts_arr,
        poses=poses,
        intrinsics=intrinsics,
    )

    if args.write_mat:
        save_matlab_model(
            out_dir / "model_4d.mat",
            points,
            colors,
            dynamic,
            uncertainty,
            time,
            timestamps,
            frame_offsets_arr,
            frame_counts_arr,
            frame_dynamic_counts_arr,
            poses,
            intrinsics,
            motion_world,
            predicted_next_points,
            motion_valid,
        )

    track_stats = {"n_tracks": 0, "n_observations": 0}
    if args.write_tracks:
        track_stats = export_dynamic_tracks(
            video_npz,
            out_dir,
            uncer_thresh=args.uncer_thresh,
            match_radius=args.match_radius,
        )

    metadata = {
        "source_video": str(video_npz),
        "n_frames": int(len(disps)),
        "n_points_total": int(len(points)),
        "n_dynamic_points_total": int(dynamic.sum()),
        "n_motion_valid_points_total": int(motion_valid.sum()),
        "mean_points_per_frame": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "mean_dynamic_per_frame": float(np.mean(frame_dynamic_counts)) if frame_dynamic_counts else 0.0,
        "uncer_thresh": args.uncer_thresh,
        "stride": args.stride,
        "max_depth": args.max_depth,
        "max_points_per_frame": args.max_points_per_frame,
        "dynamic_tracks": track_stats,
        "outputs": {
            "model_npz": "model_4d.npz",
            "model_mat": "model_4d.mat" if args.write_mat else None,
            "dynamic_tracks_npz": "dynamic_tracks.npz" if args.write_tracks else None,
            "dynamic_tracks_mat": "dynamic_tracks.mat" if args.write_tracks and args.write_mat else None,
            "ply_frames": "ply_frames" if args.write_ply else None,
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[4D] Saved model: {out_dir / 'model_4d.npz'}")
    if args.write_mat:
        print(f"[4D] Saved MATLAB model: {out_dir / 'model_4d.mat'}")
    if args.write_ply:
        print(f"[4D] Saved PLY sequence: {ply_dir}")
    if args.write_tracks:
        print(f"[4D] Saved dynamic tracks: {out_dir / 'dynamic_tracks.npz'}")
        if args.write_mat:
            print(f"[4D] Saved MATLAB tracks: {out_dir / 'dynamic_tracks.mat'}")
    print(f"[4D] Saved metadata: {out_dir / 'metadata.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a 4D point-cloud model from DROID-W video.npz."
    )
    parser.add_argument("--video-npz", required=True, help="Path to SLAM output video.npz")
    parser.add_argument("--output-dir", required=True, help="Directory to write the 4D model")
    parser.add_argument("--uncer-thresh", type=float, default=0.8, help="Dynamic uncertainty threshold")
    parser.add_argument("--stride", type=int, default=1, help="Low-res pixel stride for point sampling")
    parser.add_argument("--max-depth", type=float, default=8.0, help="Drop points deeper than this; <=0 disables")
    parser.add_argument("--max-points-per-frame", type=int, default=0, help="Random cap per frame; <=0 disables")
    parser.add_argument("--match-radius", type=float, default=0.3, help="Dynamic track nearest-neighbor radius")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for point caps")
    parser.add_argument("--no-ply", dest="write_ply", action="store_false", help="Do not export per-frame PLY files")
    parser.add_argument("--no-tracks", dest="write_tracks", action="store_false", help="Do not export dynamic_tracks.npz")
    parser.add_argument("--no-mat", dest="write_mat", action="store_false", help="Do not export MATLAB .mat files")
    parser.add_argument("--rgb-color", dest="dynamic_color", action="store_false", help="Keep original RGB for dynamic points in PLY")
    parser.set_defaults(write_ply=True, write_tracks=True, write_mat=True, dynamic_color=True)
    return parser.parse_args()


if __name__ == "__main__":
    export_4d_model(parse_args())
