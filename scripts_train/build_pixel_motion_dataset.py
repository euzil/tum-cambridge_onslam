from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def downsample_image_to_lowres(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Average-pool CHW image to the low-resolution DROID grid."""
    c, hi, wi = image.shape
    if hi % h == 0 and wi % w == 0:
        sy, sx = hi // h, wi // w
        return image.reshape(c, h, sy, w, sx).mean(axis=(2, 4)).astype(np.float32)
    ys = np.linspace(0, hi - 1, h).round().astype(np.int64)
    xs = np.linspace(0, wi - 1, w).round().astype(np.int64)
    return image[:, ys][:, :, xs].astype(np.float32)


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))
    return np.clip((x - lo) / max(hi - lo, eps), 0.0, 1.0).astype(np.float32)


def frame_points(
    pose_c2w: np.ndarray,
    disp: np.ndarray,
    intr: np.ndarray,
    dyn_mask: np.ndarray,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = disp.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    valid = dyn_mask & (disp > 1e-6)
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[disp > 1e-6] = 1.0 / disp[disp > 1e-6]
    valid &= depth > 0
    if max_depth > 0:
        valid &= depth <= max_depth

    ys = yy[valid].astype(np.int32)
    xs = xx[valid].astype(np.int32)
    z = depth[valid].astype(np.float32)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), ys, xs

    fx, fy, cx, cy = intr.astype(np.float32)
    pts_cam = np.stack(
        [
            (xs.astype(np.float32) - cx) / fx * z,
            (ys.astype(np.float32) - cy) / fy * z,
            z,
        ],
        axis=-1,
    )
    pts_world = (pose_c2w[:3, :3] @ pts_cam.T).T + pose_c2w[:3, 3][None, :]
    return pts_world.astype(np.float32), ys, xs


def select_dynamic_mask(
    uncertainty: np.ndarray,
    motion_mask: np.ndarray,
    disp: np.ndarray,
    uncer_thresh: float,
    min_points: int,
    max_points: int,
) -> np.ndarray:
    valid = disp > 1e-6
    dyn = ((uncertainty > uncer_thresh) | (motion_mask > 0)) & valid
    valid_count = int(valid.sum())
    if valid_count == 0:
        return dyn

    if min_points > 0 and int(dyn.sum()) < min_points:
        k = min(min_points, valid_count)
        scores = uncertainty[valid]
        kth = np.partition(scores, -k)[-k]
        dyn = valid & (uncertainty >= kth)

    if max_points > 0 and int(dyn.sum()) > max_points:
        flat = np.flatnonzero(dyn.reshape(-1))
        scores = uncertainty.reshape(-1)[flat]
        keep = flat[np.argpartition(scores, -max_points)[-max_points:]]
        limited = np.zeros(dyn.size, dtype=bool)
        limited[keep] = True
        dyn = limited.reshape(dyn.shape)

    return dyn


def build_pair_label(
    src_pts: np.ndarray,
    src_y: np.ndarray,
    src_x: np.ndarray,
    tgt_pts: np.ndarray,
    tgt_y: np.ndarray,
    tgt_x: np.ndarray,
    h: int,
    w: int,
    max_nn_dist: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    flow = np.zeros((2, h, w), dtype=np.float32)
    valid = np.zeros((1, h, w), dtype=np.float32)
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        return flow, valid, 0

    tree = cKDTree(tgt_pts)
    dist, idx = tree.query(src_pts, k=1)
    keep = dist <= max_nn_dist
    if not np.any(keep):
        return flow, valid, 0

    sy = src_y[keep]
    sx = src_x[keep]
    ty = tgt_y[idx[keep]]
    tx = tgt_x[idx[keep]]
    flow[0, sy, sx] = tx.astype(np.float32) - sx.astype(np.float32)
    flow[1, sy, sx] = ty.astype(np.float32) - sy.astype(np.float32)
    valid[0, sy, sx] = 1.0
    return flow, valid, int(keep.sum())


def append_video_samples(
    video_path: Path,
    source_id: int,
    args: argparse.Namespace,
    inputs: list[np.ndarray],
    flows: list[np.ndarray],
    valids: list[np.ndarray],
    frame_ids: list[int],
    source_ids: list[int],
    label_counts: list[int],
) -> tuple[int, int, int, int]:
    data = np.load(video_path)
    required = ["images", "poses", "droid_disps", "intrinsics", "uncertainties"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{video_path} is missing required fields: {missing}")

    images = np.asarray(data["images"], dtype=np.float32)
    poses = np.asarray(data["poses"], dtype=np.float32)
    disps = np.asarray(data["droid_disps"], dtype=np.float32)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    uncertainties = np.asarray(data["uncertainties"], dtype=np.float32)
    motion_masks = (
        np.asarray(data["dynamic_motion_masks"], dtype=np.float32)
        if "dynamic_motion_masks" in data.files
        else np.zeros_like(disps, dtype=np.float32)
    )

    n, h, w = disps.shape
    if inputs and inputs[0].shape[-2:] != (h, w):
        raise ValueError(
            f"{video_path} low-res shape {(h, w)} does not match existing "
            f"dataset shape {inputs[0].shape[-2:]}"
        )

    dyn_masks = [
        select_dynamic_mask(
            uncertainties[t],
            motion_masks[t],
            disps[t],
            args.uncer_thresh,
            args.min_dynamic_points,
            args.max_dynamic_points,
        )
        for t in range(n)
    ]

    used_pairs = 0
    skipped_pairs = 0
    local_label_counts: list[int] = []
    for t in range(n - 1):
        src_pts, src_y, src_x = frame_points(poses[t], disps[t], intrinsics[t], dyn_masks[t], args.max_depth)
        tgt_pts, tgt_y, tgt_x = frame_points(
            poses[t + 1], disps[t + 1], intrinsics[t + 1], dyn_masks[t + 1], args.max_depth
        )
        flow, valid, n_labels = build_pair_label(
            src_pts, src_y, src_x, tgt_pts, tgt_y, tgt_x, h, w, args.max_nn_dist
        )
        if n_labels < args.min_labels:
            skipped_pairs += 1
            continue

        rgb = downsample_image_to_lowres(images[t], h, w)
        disp_norm = normalize01(disps[t])[None]
        uncer_norm = normalize01(uncertainties[t])[None]
        dyn = dyn_masks[t][None].astype(np.float32)
        feat = np.concatenate([rgb, disp_norm, uncer_norm, dyn], axis=0).astype(np.float32)

        inputs.append(feat)
        flows.append(flow)
        valids.append(valid)
        frame_ids.append(t)
        source_ids.append(source_id)
        label_counts.append(n_labels)
        local_label_counts.append(n_labels)
        used_pairs += 1
        print(f"[dataset] src={source_id:02d} pair {t:04d}->{t + 1:04d} labels={n_labels}")

    label_mean = int(round(float(np.mean(local_label_counts)))) if local_label_counts else 0
    return n, used_pairs, skipped_pairs, label_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Build low-resolution pixel motion training data from video.npz.")
    parser.add_argument("--video-npz", required=True, nargs="+", help="One or more video.npz files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--uncer-thresh", type=float, default=0.8)
    parser.add_argument("--min-dynamic-points", type=int, default=300)
    parser.add_argument("--max-dynamic-points", type=int, default=800)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--max-nn-dist", type=float, default=0.5)
    parser.add_argument("--min-labels", type=int, default=10)
    parser.add_argument("--skip-missing", action="store_true", help="Ignore missing video.npz paths")
    args = parser.parse_args()

    video_paths = [Path(p) for p in args.video_npz]
    if args.skip_missing:
        missing = [p for p in video_paths if not p.exists()]
        for path in missing:
            print(f"[dataset] skip missing: {path}")
        video_paths = [p for p in video_paths if p.exists()]
    if not video_paths:
        raise FileNotFoundError("No valid video.npz files were provided.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs: list[np.ndarray] = []
    flows: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    frame_ids: list[int] = []
    source_ids: list[int] = []
    label_counts: list[int] = []
    source_summaries = []

    for source_id, video_path in enumerate(video_paths):
        n_frames, used_pairs, skipped_pairs, label_mean = append_video_samples(
            video_path,
            source_id,
            args,
            inputs,
            flows,
            valids,
            frame_ids,
            source_ids,
            label_counts,
        )
        source_summaries.append(
            {
                "source_id": source_id,
                "video_npz": str(video_path),
                "frames": n_frames,
                "used_pairs": used_pairs,
                "skipped_pairs": skipped_pairs,
                "label_count_mean": label_mean,
            }
        )

    if not inputs:
        raise RuntimeError("No training samples were generated. Lower --min-labels or increase --max-nn-dist.")

    samples_path = out_dir / "samples.npz"
    np.savez_compressed(
        samples_path,
        inputs=np.stack(inputs).astype(np.float32),
        flows=np.stack(flows).astype(np.float32),
        valids=np.stack(valids).astype(np.float32),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        source_ids=np.asarray(source_ids, dtype=np.int32),
        label_counts=np.asarray(label_counts, dtype=np.int32),
    )

    meta = {
        "source_videos": [str(p) for p in video_paths],
        "sources": source_summaries,
        "samples": len(inputs),
        "height": int(inputs[0].shape[-2]),
        "width": int(inputs[0].shape[-1]),
        "channels": 6,
        "uncer_thresh": args.uncer_thresh,
        "min_dynamic_points": args.min_dynamic_points,
        "max_dynamic_points": args.max_dynamic_points,
        "max_depth": args.max_depth,
        "max_nn_dist": args.max_nn_dist,
        "min_labels": args.min_labels,
        "label_count_mean": float(np.mean(label_counts)),
        "label_count_min": int(np.min(label_counts)),
        "label_count_max": int(np.max(label_counts)),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"saved dataset: {samples_path}")
    print(f"saved metadata: {out_dir / 'metadata.json'}")
    print(f"sources={len(video_paths)} samples={len(inputs)} labels_mean={meta['label_count_mean']:.2f}")


if __name__ == "__main__":
    main()
