#!/usr/bin/env python3
"""Build D4RT correspondence caches for the src D4RT-SLAM frontend.

The produced npz is read by ``src.modules.d4rt_frontend.D4RTFrontend`` and
contains SLAM-ready targets in DROID tracking-grid coordinates:

    targets: [T, T, Ht, Wt, 2]
    valids:  [T, T, Ht, Wt]

Optional D4RT 3D tracks are also saved for later depth/pose experiments, but
the current SLAM integration only requires targets/valids.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
cv2 = None
np = None
torch = None
yaml = None


def import_runtime_deps():
    global cv2, np, torch, yaml
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    if np is None:
        import numpy as _np
        np = _np
    if torch is None:
        import torch as _torch
        torch = _torch
    if yaml is None:
        import yaml as _yaml
        yaml = _yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a D4RT cache for D4RT-driven SLAM.")
    parser.add_argument("--slam-config", help="SLAM yaml config. Used for image paths and H_out/W_out.")
    parser.add_argument("--image-dir", help="Override image directory.")
    parser.add_argument(
        "--image-glob",
        default="",
        help="Override image glob, e.g. 'rgb/*.png'. Relative to image-dir if provided.",
    )
    parser.add_argument("--output", required=True, help="Output cache npz path.")
    parser.add_argument("--opend4rt-root", default="Open-d4rt", help="Path to OpenD4RT repository.")
    parser.add_argument(
        "--model-config",
        default="Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml",
        help="OpenD4RT model yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        default="Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt",
        help="OpenD4RT checkpoint.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--down-scale", type=int, default=8, help="DROID tracking downscale.")
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=1,
        help="Subsample the DROID tracking grid before querying D4RT. "
        "2 quarters the number of queries; 4 keeps 1/16.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--source-batch-size",
        type=int,
        default=1,
        help="Number of source frames queried per D4RT call. Increase only if VRAM allows.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Sigmoid visibility threshold for valid correspondences.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=-1.0,
        help="Optional confidence threshold. <0 disables.",
    )
    parser.add_argument(
        "--save-xyz",
        action="store_true",
        help="Also save D4RT xyz tracks. This can make the cache much larger.",
    )
    parser.add_argument(
        "--allow-missing-depth",
        action="store_true",
        help="Kept for future compatibility. This script does not require depth.",
    )
    return parser.parse_args()


def load_yaml_recursive(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherit = cfg.get("inherit_from")
    if inherit:
        inherit_path = Path(inherit)
        if not inherit_path.is_absolute():
            candidates = [path.parent / inherit_path, Path.cwd() / inherit_path]
            inherit_path = next((cand for cand in candidates if cand.exists()), candidates[0])
        base = load_yaml_recursive(inherit_path)
        return update_recursive(base, cfg)
    return cfg


def update_recursive(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = update_recursive(out[key], value)
        else:
            out[key] = value
    return out


def resolve_input_folder(cfg: dict[str, Any]) -> str:
    data = cfg.get("data", {})
    folder = str(data.get("input_folder", ""))
    root = str(data.get("root_folder", ""))
    if "ROOT_FOLDER_PLACEHOLDER" in folder:
        folder = folder.replace("ROOT_FOLDER_PLACEHOLDER", root)
    return folder


def natural_key(path: str) -> list[Any]:
    name = os.path.basename(path)
    parts: list[Any] = []
    token = ""
    is_digit = False
    for ch in name:
        if ch.isdigit() != is_digit and token:
            parts.append(int(token) if is_digit else token)
            token = ""
        token += ch
        is_digit = ch.isdigit()
    if token:
        parts.append(int(token) if is_digit else token)
    return parts


def list_images(image_dir: str, image_glob: str = "") -> list[str]:
    if image_glob:
        pattern = image_glob
        if image_dir and not os.path.isabs(pattern):
            pattern = os.path.join(image_dir, pattern)
        paths = glob.glob(pattern, recursive=True)
    else:
        candidates = [
            "rgb/*.png",
            "rgb/*.jpg",
            "color/*.jpg",
            "color/*.png",
            "images/*.png",
            "images/*.jpg",
            "images_anonymized/*.jpg",
            "results/frame*.jpg",
            "frame*.jpg",
            "frame*.png",
            "*.png",
            "*.jpg",
        ]
        paths = []
        for pattern in candidates:
            paths = glob.glob(os.path.join(image_dir, pattern))
            if paths:
                break
        if not paths:
            paths = [
                str(p)
                for p in Path(image_dir).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]

    paths = sorted(paths, key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No RGB images found under {image_dir!r} with glob {image_glob!r}")
    return paths


def preprocess_video_rgb(paths: list[str], cfg: dict[str, Any], model_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    cam = cfg.get("cam", {})
    h_out = int(cam.get("H_out", model_hw[0]))
    w_out = int(cam.get("W_out", model_hw[1]))
    h_edge = int(cam.get("H_edge", 0))
    w_edge = int(cam.get("W_edge", 0))
    h_resize = h_out + 2 * h_edge
    w_resize = w_out + 2 * w_edge

    frames = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        resized = cv2.resize(bgr, (w_resize, h_resize), interpolation=cv2.INTER_AREA)
        if w_edge > 0:
            resized = resized[:, w_edge:-w_edge]
        if h_edge > 0:
            resized = resized[h_edge:-h_edge, :]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    slam_rgb = np.stack(frames, axis=0).astype(np.uint8)
    model_h, model_w = model_hw
    if (slam_rgb.shape[1], slam_rgb.shape[2]) != (model_h, model_w):
        model_rgb = np.stack(
            [
                cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
                for frame in slam_rgb
            ],
            axis=0,
        ).astype(np.uint8)
    else:
        model_rgb = slam_rgb

    return model_rgb, (h_out, w_out)


def droid_tracking_grid(h_out: int, w_out: int, down_scale: int, grid_stride: int) -> tuple[np.ndarray, int, int, int, int]:
    ht = h_out // down_scale
    wt = w_out // down_scale
    grid_stride = max(1, int(grid_stride))
    ht_cache = max(1, int(np.ceil(ht / grid_stride)))
    wt_cache = max(1, int(np.ceil(wt / grid_stride)))
    ys = np.linspace(0.0, float(max(ht - 1, 0)), num=ht_cache, dtype=np.float32)
    xs = np.linspace(0.0, float(max(wt - 1, 0)), num=wt_cache, dtype=np.float32)
    ys, xs = np.meshgrid(ys, xs, indexing="ij")
    # DROID grid coordinates live in tracking-grid pixels. D4RT expects normalized
    # image coordinates, so query the center-equivalent full-resolution locations.
    u = np.clip(xs / float(max(wt - 1, 1)), 0.0, 1.0)
    v = np.clip(ys / float(max(ht - 1, 1)), 0.0, 1.0)
    return np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32), ht_cache, wt_cache, ht, wt


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
    return torch.device(raw)


def import_opend4rt(root: str):
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"OpenD4RT root not found: {root_path}")
    sys.path.insert(0, str(root_path))

    core = importlib.import_module("src.core")
    model_mod = importlib.import_module("src.model")
    infer_mod = importlib.import_module("infer_track_3d")
    return core, model_mod, infer_mod


def load_d4rt_model(args: argparse.Namespace):
    core, model_mod, infer_mod = import_opend4rt(args.opend4rt_root)
    cfg = core.load_yaml_config(args.model_config)
    model = model_mod.build_model(cfg["model"]).eval()
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = core.load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = infer_mod._unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    result = model.load_state_dict(state_dict, strict=False)
    device = resolve_device(args.device)
    model.to(device).eval()
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    model_hw = (int(image_size[0]), int(image_size[1]))
    print(
        f"Loaded D4RT checkpoint: {ckpt_path} "
        f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
    )
    return model, infer_mod, model_hw


def main() -> int:
    args = parse_args()
    import_runtime_deps()
    if args.source_batch_size <= 0:
        raise ValueError("--source-batch-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.grid_stride <= 0:
        raise ValueError("--grid-stride must be positive")

    cfg = load_yaml_recursive(args.slam_config) if args.slam_config else {}
    if args.image_dir:
        image_dir = args.image_dir
    else:
        image_dir = resolve_input_folder(cfg)
    if not image_dir:
        raise ValueError("Provide --image-dir or --slam-config with data.input_folder.")

    image_paths = list_images(image_dir, args.image_glob)
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    image_paths = image_paths[:: args.stride]
    if len(image_paths) < 2:
        raise ValueError("Need at least two frames to build D4RT correspondence cache.")

    model, infer_mod, model_hw = load_d4rt_model(args)
    video_model_rgb, (h_out, w_out) = preprocess_video_rgb(image_paths, cfg, model_hw)
    query_uv_norm, ht, wt, ht_full, wt_full = droid_tracking_grid(
        h_out, w_out, args.down_scale, args.grid_stride
    )

    num_frames = int(video_model_rgb.shape[0])
    num_grid = int(query_uv_norm.shape[0])
    targets = np.full((num_frames, num_frames, ht, wt, 2), np.nan, dtype=np.float32)
    valids = np.zeros((num_frames, num_frames, ht, wt), dtype=bool)
    confidences = np.full((num_frames, num_frames, ht, wt), np.nan, dtype=np.float32)
    depths = np.full((num_frames, ht, wt), np.nan, dtype=np.float32)

    save_xyz = bool(args.save_xyz)
    tracks_xyz_local = None
    tracks_xyz_ref0 = None
    if save_xyz:
        tracks_xyz_local = np.full((num_frames, num_frames, ht, wt, 3), np.nan, dtype=np.float32)
        tracks_xyz_ref0 = np.full_like(tracks_xyz_local, np.nan)

    print(
        f"Building D4RT cache: frames={num_frames}, grid={ht}x{wt}, "
        f"queries/source={num_grid}, source_batch={args.source_batch_size}"
    )

    for start in range(0, num_frames, args.source_batch_size):
        end = min(num_frames, start + args.source_batch_size)
        source_ids = np.arange(start, end, dtype=np.int64)
        batch_uv = np.tile(query_uv_norm, (len(source_ids), 1))
        batch_src = np.repeat(source_ids, num_grid)

        print(f"  source frames {start}:{end} ...", flush=True)
        payload = infer_mod._infer_tracks(
            model=model,
            video_model_rgb=video_model_rgb,
            query_uv_norm=batch_uv,
            query_chunk_size=int(args.query_chunk_size),
            query_src_indices_global=batch_src,
        )

        uv_norm = np.asarray(payload["tracks_uv_norm"], dtype=np.float32)
        visibility_logits = np.asarray(payload["tracks_visibility_logits"], dtype=np.float32)
        visibility = 1.0 / (1.0 + np.exp(-visibility_logits)) > float(args.visibility_threshold)
        confidence = np.asarray(payload["tracks_confidence"], dtype=np.float32)
        xyz_local = np.asarray(payload["tracks_xyz_local"], dtype=np.float32)
        uv_grid = uv_norm.copy()
        uv_grid[..., 0] *= float(max(wt_full - 1, 1))
        uv_grid[..., 1] *= float(max(ht_full - 1, 1))

        if args.confidence_threshold >= 0:
            visibility &= confidence >= float(args.confidence_threshold)
        finite = np.isfinite(uv_grid).all(axis=-1)
        in_bounds = (
            (uv_grid[..., 0] >= 0.0)
            & (uv_grid[..., 0] <= float(max(wt_full - 1, 1)))
            & (uv_grid[..., 1] >= 0.0)
            & (uv_grid[..., 1] <= float(max(ht_full - 1, 1)))
        )
        valid = visibility & finite & in_bounds

        for local_idx, src_idx in enumerate(source_ids):
            q0 = local_idx * num_grid
            q1 = q0 + num_grid
            targets[src_idx] = uv_grid[q0:q1].transpose(1, 0, 2).reshape(num_frames, ht, wt, 2)
            valids[src_idx] = valid[q0:q1].transpose(1, 0).reshape(num_frames, ht, wt)
            confidences[src_idx] = confidence[q0:q1].transpose(1, 0).reshape(num_frames, ht, wt)
            valids[src_idx, src_idx] = True
            depths[src_idx] = xyz_local[q0:q1, src_idx, 2].reshape(ht, wt)

            if save_xyz and tracks_xyz_local is not None and tracks_xyz_ref0 is not None:
                xyz_ref0 = np.asarray(payload["tracks_xyz_ref0"], dtype=np.float32)
                tracks_xyz_local[src_idx] = xyz_local[q0:q1].transpose(1, 0, 2).reshape(num_frames, ht, wt, 3)
                tracks_xyz_ref0[src_idx] = xyz_ref0[q0:q1].transpose(1, 0, 2).reshape(num_frames, ht, wt, 3)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_payload: dict[str, Any] = {
        "targets": targets,
        "valids": valids,
        "confidences": confidences,
        "depths": depths,
        "frame_paths": np.asarray(image_paths),
        "slam_image_size": np.asarray([h_out, w_out], dtype=np.int32),
        "tracking_grid_size": np.asarray([ht, wt], dtype=np.int32),
        "full_tracking_grid_size": np.asarray([ht_full, wt_full], dtype=np.int32),
        "down_scale": np.asarray(args.down_scale, dtype=np.int32),
        "grid_stride": np.asarray(args.grid_stride, dtype=np.int32),
        "target_coord_scale": np.asarray("none"),
    }
    if save_xyz and tracks_xyz_local is not None and tracks_xyz_ref0 is not None:
        save_payload["tracks_xyz_local"] = tracks_xyz_local
        save_payload["tracks_xyz_ref0"] = tracks_xyz_ref0

    np.savez_compressed(output, **save_payload)
    valid_ratio = float(valids.mean())
    print(f"Saved D4RT SLAM cache: {output}")
    print(f"Valid correspondence ratio: {valid_ratio:.4f}")
    print("Use tracking.d4rt.target_coord_scale: \"none\" for this cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
