from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamic_prediction.pixel_motion_model import SmallPixelMotionUNet
from scripts_train.build_pixel_motion_dataset import downsample_image_to_lowres, normalize01, select_dynamic_mask


def flow_to_camera_motion(
    flow: np.ndarray,
    disp: np.ndarray,
    next_disp: np.ndarray | None,
    intr: np.ndarray,
    next_intr: np.ndarray | None,
    pose_c2w: np.ndarray | None,
    next_pose_c2w: np.ndarray | None,
    mask: np.ndarray,
    max_flow_px: float,
    max_depth: float,
    depth_mode: str,
    max_motion_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = disp.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    depth = np.zeros_like(disp, dtype=np.float32)
    valid_depth = disp > 1e-6
    depth[valid_depth] = 1.0 / disp[valid_depth]

    valid = mask & valid_depth & (depth > 0)
    if max_depth > 0:
        valid &= depth <= max_depth

    du = flow[0].astype(np.float32)
    dv = flow[1].astype(np.float32)
    mag = np.sqrt(du * du + dv * dv)
    if max_flow_px > 0:
        scale = np.minimum(1.0, max_flow_px / np.maximum(mag, 1e-6))
        du = du * scale
        dv = dv * scale
        mag = np.sqrt(du * du + dv * dv)

    valid &= np.isfinite(mag)
    motion = np.zeros((h, w, 3), dtype=np.float32)
    fx, fy, cx, cy = intr.astype(np.float32)

    if depth_mode == "next" and next_disp is not None and next_intr is not None and pose_c2w is not None and next_pose_c2w is not None:
        u1 = np.clip(np.round(xx + du).astype(np.int32), 0, w - 1)
        v1 = np.clip(np.round(yy + dv).astype(np.int32), 0, h - 1)
        next_depth = np.zeros_like(next_disp, dtype=np.float32)
        next_valid_depth = next_disp > 1e-6
        next_depth[next_valid_depth] = 1.0 / next_disp[next_valid_depth]
        z1 = next_depth[v1, u1]
        valid &= z1 > 0
        if max_depth > 0:
            valid &= z1 <= max_depth

        pts0_cam = np.stack(
            [
                (xx - cx) / fx * depth,
                (yy - cy) / fy * depth,
                depth,
            ],
            axis=-1,
        )
        nfx, nfy, ncx, ncy = next_intr.astype(np.float32)
        pts1_cam = np.stack(
            [
                (u1.astype(np.float32) - ncx) / nfx * z1,
                (v1.astype(np.float32) - ncy) / nfy * z1,
                z1,
            ],
            axis=-1,
        )
        pts0_world = (pose_c2w[:3, :3] @ pts0_cam.reshape(-1, 3).T).T + pose_c2w[:3, 3][None, :]
        pts1_world = (next_pose_c2w[:3, :3] @ pts1_cam.reshape(-1, 3).T).T + next_pose_c2w[:3, 3][None, :]
        motion_world = pts1_world - pts0_world
        motion_target_cam = (next_pose_c2w[:3, :3].T @ motion_world.T).T.reshape(h, w, 3)
        motion = motion_target_cam.astype(np.float32)
    else:
        x0 = (xx - cx) / fx * depth
        y0 = (yy - cy) / fy * depth
        x1 = (xx + du - cx) / fx * depth
        y1 = (yy + dv - cy) / fy * depth
        motion[..., 0] = x1 - x0
        motion[..., 1] = y1 - y0
        motion[..., 2] = 0.0

    if max_motion_m > 0:
        motion_mag = np.linalg.norm(motion, axis=-1)
        valid &= motion_mag <= max_motion_m
    motion[~valid] = 0.0
    return motion, valid.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a learned pixel-flow model to video.npz dynamic motion fields.")
    parser.add_argument("--video-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-video", required=True)
    parser.add_argument("--uncer-thresh", type=float, default=0.8)
    parser.add_argument("--min-dynamic-points", type=int, default=300)
    parser.add_argument("--max-dynamic-points", type=int, default=800)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--max-flow-px", type=float, default=8.0)
    parser.add_argument("--max-motion-m", type=float, default=0.5)
    parser.add_argument("--depth-mode", choices=["next", "same"], default="next")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    video_path = Path(args.video_npz)
    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(video_path)
    required = ["images", "droid_disps", "intrinsics", "uncertainties", "poses"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{video_path} is missing required fields: {missing}")

    images = np.asarray(data["images"], dtype=np.float32)
    disps = np.asarray(data["droid_disps"], dtype=np.float32)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    uncertainties = np.asarray(data["uncertainties"], dtype=np.float32)
    poses = np.asarray(data["poses"], dtype=np.float32)
    old_masks = (
        np.asarray(data["dynamic_motion_masks"], dtype=np.float32)
        if "dynamic_motion_masks" in data.files
        else np.zeros_like(disps, dtype=np.float32)
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SmallPixelMotionUNet(
        in_channels=int(ckpt.get("in_channels", 6)),
        base_channels=int(ckpt.get("base_channels", 32)),
        out_channels=int(ckpt.get("out_channels", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n, h, w = disps.shape
    feats = []
    dyn_masks = []
    for t in range(n):
        dyn = select_dynamic_mask(
            uncertainties[t],
            old_masks[t],
            disps[t],
            args.uncer_thresh,
            args.min_dynamic_points,
            args.max_dynamic_points,
        )
        rgb = downsample_image_to_lowres(images[t], h, w)
        disp_norm = normalize01(disps[t])[None]
        uncer_norm = normalize01(uncertainties[t])[None]
        feat = np.concatenate([rgb, disp_norm, uncer_norm, dyn[None].astype(np.float32)], axis=0)
        feats.append(feat.astype(np.float32))
        dyn_masks.append(dyn)

    pred_flows = np.zeros((n, h, w, 2), dtype=np.float32)
    learned_motions = np.zeros((n, h, w, 3), dtype=np.float32)
    learned_masks = np.zeros((n, h, w), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            x = torch.from_numpy(np.stack(feats[start:end])).to(device)
            pred = model(x).detach().cpu().numpy()
            for bi, t in enumerate(range(start, end)):
                flow = pred[bi]
                motion, valid = flow_to_camera_motion(
                    flow,
                    disps[t],
                    disps[t + 1] if t + 1 < n else None,
                    intrinsics[t],
                    intrinsics[t + 1] if t + 1 < n else None,
                    poses[t],
                    poses[t + 1] if t + 1 < n else None,
                    dyn_masks[t],
                    args.max_flow_px,
                    args.max_depth,
                    args.depth_mode,
                    args.max_motion_m,
                )
                pred_flows[t] = np.moveaxis(flow, 0, -1)
                learned_motions[t] = motion
                learned_masks[t] = valid
            print(f"\r[learned-motion] frames {end:04d}/{n}", end="", flush=True)
    print()

    out = {key: data[key] for key in data.files}
    out["dynamic_motion_flow_learned"] = pred_flows.astype(np.float32)
    out["dynamic_motions_original"] = (
        np.asarray(data["dynamic_motions"], dtype=np.float32)
        if "dynamic_motions" in data.files
        else np.zeros_like(learned_motions)
    )
    out["dynamic_motion_masks_original"] = old_masks.astype(np.float32)
    out["dynamic_motions"] = learned_motions.astype(np.float32)
    out["dynamic_motion_priors"] = learned_motions.astype(np.float32)
    out["dynamic_motion_masks"] = learned_masks.astype(np.float32)
    np.savez_compressed(out_path, **out)

    active = learned_masks > 0
    motion_mag = np.linalg.norm(learned_motions[active], axis=-1) if np.any(active) else np.zeros((0,))
    flow_mag = np.linalg.norm(pred_flows[active], axis=-1) if np.any(active) else np.zeros((0,))
    meta = {
        "source_video": str(video_path),
        "checkpoint": str(args.checkpoint),
        "out_video": str(out_path),
        "frames": int(n),
        "active_pixels": int(active.sum()),
        "active_ratio": float(active.mean()),
        "flow_mean_px": float(flow_mag.mean()) if flow_mag.size else 0.0,
        "flow_p90_px": float(np.percentile(flow_mag, 90)) if flow_mag.size else 0.0,
        "motion_mean_m": float(motion_mag.mean()) if motion_mag.size else 0.0,
        "motion_p90_m": float(np.percentile(motion_mag, 90)) if motion_mag.size else 0.0,
        "uncer_thresh": args.uncer_thresh,
        "min_dynamic_points": args.min_dynamic_points,
        "max_dynamic_points": args.max_dynamic_points,
        "max_depth": args.max_depth,
        "max_flow_px": args.max_flow_px,
        "max_motion_m": args.max_motion_m,
        "depth_mode": args.depth_mode,
    }
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"saved learned-motion video: {out_path}")
    print(f"saved metadata: {out_path.with_suffix('.json')}")
    print(
        f"active={meta['active_pixels']} ({meta['active_ratio']:.4%}) "
        f"flow_mean={meta['flow_mean_px']:.4f}px motion_mean={meta['motion_mean_m']:.6f}m"
    )


if __name__ == "__main__":
    main()
