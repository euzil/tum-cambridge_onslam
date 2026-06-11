# D4RT Cache Generation Next Steps

This note records the practical steps after downloading OpenD4RT checkpoints.

## 1. Check Checkpoints

Expected files:

```text
Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml
Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml
Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
```

Check locally:

```bash
find Open-d4rt/checkpoints -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
```

## 2. Why Cache Generation May Be Killed

If the command prints only:

```text
Killed
```

or the shell says it was killed, it is most likely the Linux OOM killer.

Two common memory pressure points:

```text
1. Loading the ~13 GB OpenD4RT checkpoint causes a high CPU RAM peak.
2. Dense pairwise cache generation has cost roughly T * T * Ht * Wt.
```

Check memory:

```bash
free -h
```

If even the smallest smoke command is killed, the problem is probably checkpoint loading memory rather than query size.

## 3. Low-Memory Smoke Cache

The cache script supports:

```text
--max-frames
    Reduce the number of video frames.

--grid-stride
    Subsample the DROID tracking grid before querying D4RT.
    2 keeps roughly 1/4 of grid queries.
    4 keeps roughly 1/16 of grid queries.

--query-chunk-size
    Reduce D4RT query chunk size to lower GPU memory.

--source-batch-size
    Keep this at 1 for low-memory runs.
```

The script saves target coordinates in DROID tracking-grid coordinates. Use:

```yaml
tracking:
  d4rt:
    target_coord_scale: "none"
```

## 4. TUM Freiburg3 Walking Smoke

Start with the 32-frame checkpoint and a very sparse cache:

```bash
conda activate d4rt

python scripts_d4rt/build_d4rt_slam_cache.py \
  --slam-config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml \
  --opend4rt-root Open-d4rt \
  --model-config Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --ckpt-path Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --output output/d4rt_cache/freiburg3_walking_xyz_smoke8_g4.npz \
  --device cuda \
  --max-frames 8 \
  --grid-stride 4 \
  --query-chunk-size 128 \
  --source-batch-size 1
```

If this succeeds, gradually increase:

```text
8 frames,  grid-stride 4
8 frames,  grid-stride 2
16 frames, grid-stride 4
16 frames, grid-stride 2
16 frames, grid-stride 1
```

## 5. Bonn Balloon Smoke

For `bonn_balloon`:

```bash
conda activate d4rt

python scripts_d4rt/build_d4rt_slam_cache.py \
  --slam-config configs/Dynamic/Bonn/bonn_balloon.yaml \
  --opend4rt-root Open-d4rt \
  --model-config Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --ckpt-path Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --output output/d4rt_cache/bonn_balloon_smoke8_g4.npz \
  --device cuda \
  --max-frames 8 \
  --grid-stride 4 \
  --query-chunk-size 128 \
  --source-batch-size 1
```

## 6. Inspect Generated Cache

```bash
python - <<'PY'
import numpy as np
p = "output/d4rt_cache/freiburg3_walking_xyz_smoke8_g4.npz"
d = np.load(p)
for k in d.files:
    print(k, d[k].shape if hasattr(d[k], "shape") else type(d[k]))
print("valid ratio:", d["valids"].mean())
PY
```

Expected core keys:

```text
targets      [T, T, H_cache, W_cache, 2]
valids       [T, T, H_cache, W_cache]
confidences  [T, T, H_cache, W_cache]
depths       [T, H_cache, W_cache]
```

## 7. Run D4RT-SLAM Smoke

For TUM smoke, set:

```yaml
tracking:
  d4rt:
    activate: true
    mode: "offline"
    cache_path: "output/d4rt_cache/freiburg3_walking_xyz_smoke8_g4.npz"
    target_coord_scale: "none"
    use_target: true
    use_depth_init: false
    use_depth_reg: false
    use_pose_init: false
    use_edge_selection: false
```

Then run:

```bash
python run.py --config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz_d4rt_smoke.yaml
```

For Bonn balloon smoke, set:

```yaml
tracking:
  d4rt:
    cache_path: "output/d4rt_cache/bonn_balloon_smoke8_g4.npz"
    target_coord_scale: "none"
```

Then run:

```bash
python run.py --config configs/Dynamic/Bonn/bonn_balloon_d4rt_smoke.yaml
```

## 8. Recommended Experiment Order

Do not enable every D4RT option at once.

```text
1. D4RT target only
2. D4RT target + depth initialization
3. D4RT target + depth regularization
4. D4RT target + edge selection
5. Full D4RT-SLAM
```

Recommended smoke progression:

```text
cache generation succeeds
    ↓
SLAM smoke does not crash
    ↓
ATE output exists
    ↓
increase frames
    ↓
increase grid density
    ↓
enable depth/edge ablations
```

## 9. Notes

The first goal is not to get the best ATE. The first goal is:

```text
D4RT target only + BA runs end-to-end without crashing.
```

Only after this works should cache size, grid density, and D4RT options be increased.
