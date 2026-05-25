from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_frame(model_npz: Path, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(model_npz)
    offsets = np.asarray(data["frame_offsets"], dtype=np.int64)
    if frame_idx < 0 or frame_idx >= len(offsets) - 1:
        raise IndexError(f"frame_idx must be in [0, {len(offsets) - 2}]")
    lo, hi = offsets[frame_idx], offsets[frame_idx + 1]
    return data["points"][lo:hi], data["colors"][lo:hi], data["dynamic"][lo:hi]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play exported 4D model_4d.npz with Open3D.")
    parser.add_argument("--model-npz", required=True, help="Path to model_4d.npz")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--dynamic-red", action="store_true", help="Show dynamic points in red")
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Install open3d to use this viewer: pip install open3d") from exc

    data = np.load(args.model_npz)
    n_frames = len(data["frame_offsets"]) - 1

    vis = o3d.visualization.Visualizer()
    vis.create_window("4D model viewer", width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.03, 0.03, 0.03])

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    first = True
    delay = 1.0 / args.fps if args.fps > 0 else 0.0

    import time

    for frame_idx in range(n_frames):
        pts, colors, dynamic = load_frame(Path(args.model_npz), frame_idx)
        colors_f = colors.astype(np.float32) / 255.0
        if args.dynamic_red:
            colors_f = colors_f.copy()
            colors_f[dynamic] = np.array([1.0, 0.05, 0.02], dtype=np.float32)

        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors_f)
        vis.update_geometry(pcd)
        if first:
            vis.reset_view_point(True)
            first = False
        if not vis.poll_events():
            break
        vis.update_renderer()
        if delay > 0:
            time.sleep(delay)

    vis.destroy_window()


if __name__ == "__main__":
    main()
