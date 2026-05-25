import argparse
from pathlib import Path

import numpy as np


def summarize(name, arr, mask=None):
    if mask is not None:
        values = arr[mask]
    else:
        values = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 1 and arr.shape[-1] == 3 else arr.reshape(-1)

    if values.size == 0:
        print(f"{name}: empty")
        return

    if values.ndim == 2 and values.shape[-1] == 3:
        mag = np.linalg.norm(values, axis=-1)
        print(
            f"{name}: count={values.shape[0]} "
            f"mag_mean={mag.mean():.6f} mag_p50={np.percentile(mag, 50):.6f} "
            f"mag_p90={np.percentile(mag, 90):.6f} mag_max={mag.max():.6f}"
        )
    else:
        print(
            f"{name}: count={values.size} mean={values.mean():.6f} "
            f"p50={np.percentile(values, 50):.6f} p90={np.percentile(values, 90):.6f} "
            f"max={values.max():.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect CUDA dynamic motion BA tensors saved in video.npz."
    )
    parser.add_argument("--video-npz", required=True, help="Path to output video.npz")
    args = parser.parse_args()

    video_npz = Path(args.video_npz)
    data = np.load(video_npz)
    required = ["dynamic_motions", "dynamic_motion_priors", "dynamic_motion_masks"]
    missing = [k for k in required if k not in data.files]
    if missing:
        print(f"{video_npz} does not contain {missing}.")
        print("Rerun SLAM after the latest save_video change, then run this script again.")
        return

    motions = np.asarray(data["dynamic_motions"], dtype=np.float32)
    priors = np.asarray(data["dynamic_motion_priors"], dtype=np.float32)
    masks = np.asarray(data["dynamic_motion_masks"], dtype=np.float32)
    active = masks > 0

    print(f"video: {video_npz}")
    print(f"frames={motions.shape[0]} lowres={motions.shape[1]}x{motions.shape[2]}")
    print(f"active pixels={int(active.sum())} / {active.size} ({active.mean() * 100:.4f}%)")
    summarize("motion", motions, active)
    summarize("prior ", priors, active)
    summarize("delta ", motions - priors, active)

    per_frame_active = active.reshape(active.shape[0], -1).sum(axis=1)
    used_frames = np.where(per_frame_active > 0)[0]
    if used_frames.size:
        print(
            f"used frames={used_frames.size}, first={used_frames[0]}, last={used_frames[-1]}, "
            f"active/frame mean={per_frame_active[used_frames].mean():.2f}, "
            f"max={per_frame_active[used_frames].max()}"
        )


if __name__ == "__main__":
    main()
