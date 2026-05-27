from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.ndimage import binary_closing, binary_dilation, label
from scipy.spatial import cKDTree


@dataclass
class DynamicObject:
    frame: int
    local_id: int
    global_id: int
    pixels_ij: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    centroid: np.ndarray
    motion_world: np.ndarray
    motion_valid_ratio: float


def load_video(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = ["images", "poses", "droid_disps", "intrinsics", "uncertainties"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"{path} is missing {missing}")

    out = {k: np.asarray(data[k]) for k in required}
    out["timestamps"] = np.asarray(data["timestamps"] if "timestamps" in data.files else np.arange(len(out["poses"])))
    out["dynamic_motions"] = (
        np.asarray(data["dynamic_motions"], dtype=np.float32)
        if "dynamic_motions" in data.files
        else np.zeros((*out["droid_disps"].shape, 3), dtype=np.float32)
    )
    out["dynamic_motion_masks"] = (
        np.asarray(data["dynamic_motion_masks"], dtype=np.float32)
        if "dynamic_motion_masks" in data.files
        else np.zeros_like(out["droid_disps"], dtype=np.float32)
    )
    return out


def mask_to_components(mask: np.ndarray, min_area: int, dilate_iters: int) -> list[np.ndarray]:
    work = mask.astype(bool)
    if dilate_iters > 0:
        work = binary_dilation(work, iterations=dilate_iters)
        work = binary_closing(work, iterations=1)
    labels, n_labels = label(work)
    comps = []
    for lab in range(1, n_labels + 1):
        comp = labels == lab
        if int(comp.sum()) >= min_area:
            comps.append(comp)
    return comps


def pixels_to_points(
    image: np.ndarray,
    pose_c2w: np.ndarray,
    disp: np.ndarray,
    intr: np.ndarray,
    comp_mask: np.ndarray,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(comp_mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 2), dtype=np.int32)

    disp_v = disp[ys, xs].astype(np.float32)
    valid = disp_v > 1e-6
    depth = np.zeros_like(disp_v)
    depth[valid] = 1.0 / disp_v[valid]
    valid &= depth > 0
    if max_depth > 0:
        valid &= depth <= max_depth

    ys = ys[valid]
    xs = xs[valid]
    depth = depth[valid]
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 2), dtype=np.int32)

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

    _, h_img, w_img = image.shape
    h_l, w_l = disp.shape
    img_x = np.clip(np.round(xs * (w_img / w_l)).astype(int), 0, w_img - 1)
    img_y = np.clip(np.round(ys * (h_img / h_l)).astype(int), 0, h_img - 1)
    colors = np.clip(image.transpose(1, 2, 0)[img_y, img_x] * 255.0, 0, 255).astype(np.uint8)
    pixels = np.stack([ys, xs], axis=-1).astype(np.int32)
    return pts_world.astype(np.float32), colors, pixels


def extract_objects(video: dict[str, np.ndarray], args: argparse.Namespace) -> list[list[DynamicObject]]:
    images = video["images"].astype(np.float32)
    poses = video["poses"].astype(np.float32)
    disps = video["droid_disps"].astype(np.float32)
    intrinsics = video["intrinsics"].astype(np.float32)
    uncertainties = video["uncertainties"].astype(np.float32)
    motions = video["dynamic_motions"].astype(np.float32)
    motion_masks = video["dynamic_motion_masks"].astype(np.float32)

    all_objects: list[list[DynamicObject]] = []
    for t in range(len(disps)):
        dyn_mask = (uncertainties[t] > args.uncer_thresh) | (motion_masks[t] > 0)
        comps = mask_to_components(dyn_mask, args.min_area, args.dilate_iters)
        frame_objects: list[DynamicObject] = []

        for local_id, comp in enumerate(comps, start=1):
            pts, colors, pixels = pixels_to_points(images[t], poses[t], disps[t], intrinsics[t], comp, args.max_depth)
            if len(pts) < args.min_points:
                continue

            yy = pixels[:, 0]
            xx = pixels[:, 1]
            valid_motion = motion_masks[t, yy, xx] > 0
            motion_world = np.zeros(3, dtype=np.float32)
            if np.any(valid_motion) and t + 1 < len(poses):
                motion_cam = motions[t, yy[valid_motion], xx[valid_motion]]
                motion_world = np.median((poses[t + 1][:3, :3] @ motion_cam.T).T, axis=0).astype(np.float32)

            frame_objects.append(
                DynamicObject(
                    frame=t,
                    local_id=local_id,
                    global_id=-1,
                    pixels_ij=pixels,
                    points=pts,
                    colors=colors,
                    centroid=np.median(pts, axis=0).astype(np.float32),
                    motion_world=motion_world,
                    motion_valid_ratio=float(valid_motion.mean()) if len(valid_motion) else 0.0,
                )
            )
        all_objects.append(frame_objects)
    return all_objects


def assign_global_ids(objects_by_frame: list[list[DynamicObject]], max_match_dist: float) -> None:
    next_gid = 1
    for obj in objects_by_frame[0] if objects_by_frame else []:
        obj.global_id = next_gid
        next_gid += 1

    for t in range(len(objects_by_frame) - 1):
        prev = objects_by_frame[t]
        curr = objects_by_frame[t + 1]
        unmatched = set(range(len(curr)))

        for po in prev:
            if not curr:
                continue
            pred = po.centroid + po.motion_world
            dists = np.asarray([np.linalg.norm(co.centroid - pred) for co in curr], dtype=np.float32)
            order = np.argsort(dists)
            for idx in order:
                if int(idx) in unmatched and dists[idx] <= max_match_dist:
                    curr[int(idx)].global_id = po.global_id
                    unmatched.remove(int(idx))
                    break

        for idx in sorted(unmatched):
            curr[idx].global_id = next_gid
            next_gid += 1


def refine_object_motions(objects_by_frame: list[list[DynamicObject]], blend: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    by_frame_gid = [{obj.global_id: obj for obj in objs} for objs in objects_by_frame]

    for t in range(len(objects_by_frame) - 1):
        for obj in objects_by_frame[t]:
            nxt = by_frame_gid[t + 1].get(obj.global_id)
            if nxt is None:
                continue
            centroid_motion = nxt.centroid - obj.centroid
            if np.linalg.norm(obj.motion_world) > 1e-8 and obj.motion_valid_ratio > 0:
                motion = (1.0 - blend) * centroid_motion + blend * obj.motion_world
            else:
                motion = centroid_motion
            obj.motion_world = motion.astype(np.float32)
            rows.append(
                {
                    "frame": float(t),
                    "object_id": float(obj.global_id),
                    "n_points": float(len(obj.points)),
                    "motion_norm": float(np.linalg.norm(obj.motion_world)),
                    "motion_valid_ratio": float(obj.motion_valid_ratio),
                }
            )
    return rows


def evaluate_object_consistency(objects_by_frame: list[list[DynamicObject]], max_nn_dist: float) -> dict[str, float]:
    by_frame_gid = [{obj.global_id: obj for obj in objs} for objs in objects_by_frame]
    static_all = []
    object_all = []

    for t in range(len(objects_by_frame) - 1):
        for obj in objects_by_frame[t]:
            nxt = by_frame_gid[t + 1].get(obj.global_id)
            if nxt is None or len(obj.points) == 0 or len(nxt.points) == 0:
                continue
            tree = cKDTree(nxt.points)
            static_dist, _ = tree.query(obj.points, k=1)
            moved_dist, _ = tree.query(obj.points + obj.motion_world[None, :], k=1)
            keep = (static_dist <= max_nn_dist) | (moved_dist <= max_nn_dist)
            if not np.any(keep):
                continue
            static_all.extend(static_dist[keep].tolist())
            object_all.extend(moved_dist[keep].tolist())

    static = np.asarray(static_all, dtype=np.float32)
    moved = np.asarray(object_all, dtype=np.float32)
    if static.size == 0:
        return {
            "n_eval_points": 0,
            "static_mean": float("nan"),
            "object_mean": float("nan"),
            "gain_percent": float("nan"),
        }
    return {
        "n_eval_points": int(static.size),
        "static_mean": float(static.mean()),
        "static_median": float(np.median(static)),
        "object_mean": float(moved.mean()),
        "object_median": float(np.median(moved)),
        "gain_percent": float((static.mean() - moved.mean()) / max(static.mean(), 1e-8) * 100.0),
    }


def export_model(objects_by_frame: list[list[DynamicObject]], timestamps: np.ndarray, out_dir: Path, metadata: dict) -> None:
    points_all = []
    colors_all = []
    frame_all = []
    object_all = []
    motion_all = []
    predicted_all = []
    frame_offsets = [0]
    frame_object_counts = []

    object_rows = []
    for t, objs in enumerate(objects_by_frame):
        frame_count = 0
        for obj in objs:
            n = len(obj.points)
            points_all.append(obj.points)
            colors_all.append(obj.colors)
            frame_all.append(np.full(n, t, dtype=np.int32))
            object_all.append(np.full(n, obj.global_id, dtype=np.int32))
            motion = np.repeat(obj.motion_world.reshape(1, 3), n, axis=0).astype(np.float32)
            motion_all.append(motion)
            predicted_all.append((obj.points + motion).astype(np.float32))
            frame_count += n
            object_rows.append(
                [
                    t,
                    obj.global_id,
                    obj.local_id,
                    n,
                    obj.centroid[0],
                    obj.centroid[1],
                    obj.centroid[2],
                    obj.motion_world[0],
                    obj.motion_world[1],
                    obj.motion_world[2],
                    obj.motion_valid_ratio,
                ]
            )
        frame_offsets.append(frame_offsets[-1] + frame_count)
        frame_object_counts.append(len(objs))

    points = np.concatenate(points_all, axis=0) if points_all else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(colors_all, axis=0) if colors_all else np.zeros((0, 3), dtype=np.uint8)
    frame_ids = np.concatenate(frame_all, axis=0) if frame_all else np.zeros((0,), dtype=np.int32)
    object_ids = np.concatenate(object_all, axis=0) if object_all else np.zeros((0,), dtype=np.int32)
    motion_world = np.concatenate(motion_all, axis=0) if motion_all else np.zeros((0, 3), dtype=np.float32)
    predicted_next_points = (
        np.concatenate(predicted_all, axis=0) if predicted_all else np.zeros((0, 3), dtype=np.float32)
    )
    object_table = np.asarray(object_rows, dtype=np.float32)
    frame_offsets = np.asarray(frame_offsets, dtype=np.int64)

    np.savez_compressed(
        out_dir / "object_4d_model.npz",
        points=points,
        colors=colors,
        frame_ids=frame_ids,
        object_ids=object_ids,
        motion_world=motion_world,
        predicted_next_points=predicted_next_points,
        timestamps=timestamps,
        frame_offsets=frame_offsets,
        frame_object_counts=np.asarray(frame_object_counts, dtype=np.int32),
        object_table=object_table,
    )
    savemat(
        out_dir / "object_4d_model.mat",
        {
            "points": points.astype(np.float32),
            "colors": colors.astype(np.uint8),
            "frame_ids": frame_ids.reshape(-1, 1),
            "object_ids": object_ids.reshape(-1, 1),
            "motion_world": motion_world.astype(np.float32),
            "predicted_next_points": predicted_next_points.astype(np.float32),
            "timestamps": timestamps.astype(np.float32).reshape(-1, 1),
            "frame_offsets0": frame_offsets.reshape(-1, 1),
            "frame_start_1": (frame_offsets[:-1] + 1).reshape(-1, 1),
            "frame_end_1": frame_offsets[1:].reshape(-1, 1),
            "frame_object_counts": np.asarray(frame_object_counts, dtype=np.int32).reshape(-1, 1),
            "object_table": object_table,
        },
        do_compression=True,
    )

    object_header = (
        "frame,object_id,local_id,n_points,cx,cy,cz,motion_x,motion_y,motion_z,motion_valid_ratio"
    )
    np.savetxt(out_dir / "object_table.csv", object_table, delimiter=",", header=object_header, comments="")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build object-level dynamic 4D model from DROID-W output.")
    parser.add_argument("--video-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--uncer-thresh", type=float, default=0.9)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--dilate-iters", type=int, default=1)
    parser.add_argument("--match-dist", type=float, default=0.35)
    parser.add_argument("--motion-blend", type=float, default=0.25)
    parser.add_argument("--max-nn-dist", type=float, default=0.5)
    args = parser.parse_args()

    video_npz = Path(args.video_npz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video = load_video(video_npz)
    objects_by_frame = extract_objects(video, args)
    assign_global_ids(objects_by_frame, args.match_dist)
    motion_rows = refine_object_motions(objects_by_frame, args.motion_blend)
    metrics = evaluate_object_consistency(objects_by_frame, args.max_nn_dist)

    n_objects = sum(len(objs) for objs in objects_by_frame)
    n_tracks = len({obj.global_id for objs in objects_by_frame for obj in objs})
    n_points = sum(len(obj.points) for objs in objects_by_frame for obj in objs)
    metadata = {
        "source_video": str(video_npz),
        "n_frames": int(len(objects_by_frame)),
        "n_object_instances": int(n_objects),
        "n_object_tracks": int(n_tracks),
        "n_dynamic_points": int(n_points),
        "params": vars(args),
        "object_consistency": metrics,
    }
    export_model(objects_by_frame, video["timestamps"], out_dir, metadata)
    if motion_rows:
        header = "frame,object_id,n_points,motion_norm,motion_valid_ratio"
        table = np.asarray([[r[k] for k in ["frame", "object_id", "n_points", "motion_norm", "motion_valid_ratio"]] for r in motion_rows])
        np.savetxt(out_dir / "object_motion_summary.csv", table, delimiter=",", header=header, comments="")

    print(f"saved object model: {out_dir / 'object_4d_model.npz'}")
    print(f"saved MATLAB model: {out_dir / 'object_4d_model.mat'}")
    print(f"object instances: {n_objects}, tracks: {n_tracks}, points: {n_points}")
    print(
        "object-level dynamic consistency: "
        f"static_mean={metrics['static_mean']:.6f} m, "
        f"object_mean={metrics['object_mean']:.6f} m, "
        f"gain={metrics['gain_percent']:+.2f}%"
    )


if __name__ == "__main__":
    main()
