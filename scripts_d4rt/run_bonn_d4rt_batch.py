#!/usr/bin/env python3
"""Batch D4RT-SLAM smoke/ablation runner for Bonn dynamic scenes.

The script does three things per scene:

1. Build a D4RT cache with scripts_d4rt/build_d4rt_slam_cache.py.
2. Generate a temporary SLAM config that enables tracking.d4rt.
3. Run python run.py --config <generated-config>.

Use --cache-python and --slam-python if D4RT and DROID-W live in different
conda environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENES = [
    "bonn_balloon",
    "bonn_balloon2",
    "bonn_person_tracking",
    "bonn_person_tracking2",
    "bonn_moving_nonobstructing_box",
    "bonn_moving_nonobstructing_box2",
    "bonn_crowd",
    "bonn_crowd2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch D4RT-SLAM on Bonn scenes.")
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES, help="Bonn scene names without .yaml.")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--grid-stride", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--source-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--warmup", type=int, default=0, help="0 means auto.")
    parser.add_argument("--tag", default="", help="Optional suffix for generated scene names.")
    parser.add_argument("--cache-dir", default="output/d4rt_cache")
    parser.add_argument("--generated-config-dir", default="configs/generated/d4rt_bonn")
    parser.add_argument("--output-csv", default="", help="Optional summary csv path.")
    parser.add_argument("--cache-python", default=sys.executable, help="Python used to generate D4RT cache.")
    parser.add_argument("--slam-python", default=sys.executable, help="Python used to run SLAM.")
    parser.add_argument("--opend4rt-root", default="Open-d4rt")
    parser.add_argument(
        "--model-config",
        default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml",
    )
    parser.add_argument(
        "--ckpt-path",
        default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt",
    )
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-depth-init", action="store_true")
    parser.add_argument("--enable-depth-reg", action="store_true")
    parser.add_argument("--enable-edge-selection", action="store_true")
    parser.add_argument("--enable-pose-init", action="store_true")
    parser.add_argument("--enable-final-ba", action="store_true")
    parser.add_argument("--eval-before-final-ba", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], *, dry_run: bool) -> int:
    print("\n$", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def scene_suffix(args: argparse.Namespace) -> str:
    suffix = f"f{args.max_frames}_g{args.grid_stride}"
    if args.enable_depth_init:
        suffix += "_dinit"
    if args.enable_depth_reg:
        suffix += "_dreg"
    if args.enable_edge_selection:
        suffix += "_edge"
    if args.enable_pose_init:
        suffix += "_pinit"
    if args.tag:
        suffix += f"_{args.tag}"
    return suffix


def auto_warmup(max_frames: int) -> int:
    if max_frames <= 8:
        return 4
    if max_frames <= 16:
        return 6
    return 12


def write_config(
    *,
    base_config: Path,
    generated_config: Path,
    scene_name: str,
    max_frames: int,
    warmup: int,
    cache_path: Path,
    args: argparse.Namespace,
) -> None:
    generated_config.parent.mkdir(parents=True, exist_ok=True)
    rel_base = os.path.relpath(base_config, generated_config.parent)
    text = f"""inherit_from: {rel_base}

scene: {scene_name}
max_frames: {max_frames}

mapping:
  eval_before_final_ba: {str(bool(args.eval_before_final_ba)).lower()}

tracking:
  warmup: {warmup}
  backend:
    final_ba: {str(bool(args.enable_final_ba)).lower()}
  d4rt:
    activate: true
    mode: "offline"
    cache_path: "{cache_path.as_posix()}"
    target_coord_scale: "none"
    use_target: true
    use_depth_init: {str(bool(args.enable_depth_init)).lower()}
    use_depth_reg: {str(bool(args.enable_depth_reg)).lower()}
    use_pose_init: {str(bool(args.enable_pose_init)).lower()}
    use_edge_selection: {str(bool(args.enable_edge_selection)).lower()}
"""
    generated_config.write_text(text, encoding="utf-8")


def parse_metrics(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"statistics:\s*\n(\{.*?\})", text, flags=re.S)
    if not match:
        return {}
    try:
        # Metrics files use a Python dict with single quotes; json cannot parse it.
        import ast

        stats = ast.literal_eval(match.group(1))
        return {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}
    except Exception as exc:
        return {"parse_error": str(exc)}


def write_summary(rows: list[dict[str, object]], output_csv: str) -> None:
    if not output_csv:
        return
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSaved summary: {path}")


def main() -> int:
    args = parse_args()
    suffix = scene_suffix(args)
    cache_dir = Path(args.cache_dir)
    generated_dir = Path(args.generated_config_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for scene in args.scenes:
        base_config = Path("configs/Dynamic/Bonn") / f"{scene}.yaml"
        if not base_config.exists():
            print(f"Skipping missing config: {base_config}", file=sys.stderr)
            continue

        cache_path = cache_dir / f"{scene}_{suffix}.npz"
        generated_scene = f"{scene}_d4rt_{suffix}"
        generated_config = generated_dir / f"{generated_scene}.yaml"
        warmup = int(args.warmup) if int(args.warmup) > 0 else auto_warmup(args.max_frames)

        print("\n" + "=" * 80)
        print(f"Scene: {scene}")
        print(f"Generated scene: {generated_scene}")
        print(f"Cache: {cache_path}")
        print(f"Config: {generated_config}")

        if not args.skip_cache and (args.overwrite_cache or not cache_path.exists()):
            cmd = [
                args.cache_python,
                "scripts_d4rt/build_d4rt_slam_cache.py",
                "--slam-config",
                str(base_config),
                "--opend4rt-root",
                args.opend4rt_root,
                "--model-config",
                args.model_config,
                "--ckpt-path",
                args.ckpt_path,
                "--output",
                str(cache_path),
                "--device",
                args.device,
                "--max-frames",
                str(args.max_frames),
                "--grid-stride",
                str(args.grid_stride),
                "--query-chunk-size",
                str(args.query_chunk_size),
                "--source-batch-size",
                str(args.source_batch_size),
            ]
            code = run_cmd(cmd, dry_run=args.dry_run)
            if code != 0:
                rows.append({"scene": scene, "status": "cache_failed", "returncode": code})
                if not args.dry_run:
                    continue
        elif cache_path.exists():
            print(f"Cache exists, reusing: {cache_path}")

        if args.overwrite_config or not generated_config.exists():
            write_config(
                base_config=base_config,
                generated_config=generated_config,
                scene_name=generated_scene,
                max_frames=args.max_frames,
                warmup=warmup,
                cache_path=cache_path,
                args=args,
            )

        if not args.skip_run:
            cmd = [args.slam_python, "run.py", "--config", str(generated_config)]
            code = run_cmd(cmd, dry_run=args.dry_run)
            status = "ok" if code == 0 else "slam_failed"
        else:
            code = 0
            status = "skipped_run"

        metrics_path = Path("Outputs/Bonn") / generated_scene / "traj" / "metrics_full_traj.txt"
        row: dict[str, object] = {
            "scene": scene,
            "generated_scene": generated_scene,
            "cache": str(cache_path),
            "config": str(generated_config),
            "status": status,
            "returncode": code,
            "max_frames": args.max_frames,
            "grid_stride": args.grid_stride,
            "warmup": warmup,
            "depth_init": bool(args.enable_depth_init),
            "depth_reg": bool(args.enable_depth_reg),
            "edge_selection": bool(args.enable_edge_selection),
            "pose_init": bool(args.enable_pose_init),
        }
        row.update(parse_metrics(metrics_path))
        rows.append(row)

    write_summary(rows, args.output_csv)
    print("\nBatch summary:")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0 if all(row.get("returncode", 0) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
