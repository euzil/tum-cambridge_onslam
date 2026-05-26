# Bonn Crowd Test Commands

This file records the command sequence for testing the same dynamic-object 4D mapping pipeline on `bonn_crowd`.

## Environment

```bash
cd /home/youran/文档/DROID-W/TUM-Cambridge_onSLAM
conda activate droid-w
```

## 1. Run SLAM

```bash
python run.py --config configs/Dynamic/Bonn/bonn_crowd.yaml
```

Expected main output:

```text
Outputs/Bonn/bonn_crowd/video.npz
```

## 2. Export Previous/Kalman 4D Model

```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_crowd/video.npz \
  --output-dir Outputs/Bonn/bonn_crowd/4d_model_previous_motion \
  --disp-source up \
  --no-ply \
  --no-tracks
```

Expected output:

```text
Outputs/Bonn/bonn_crowd/4d_model_previous_motion/model_4d.npz
Outputs/Bonn/bonn_crowd/4d_model_previous_motion/model_4d.mat
```

## 3. Apply Learned Pixel Motion

```bash
python scripts_train/apply_pixel_motion_to_video.py \
  --video-npz Outputs/Bonn/bonn_crowd/video.npz \
  --checkpoint checkpoints/pixel_motion_bonn_dynamic/best.pt \
  --out-video Outputs/Bonn/bonn_crowd/video_learned_motion.npz \
  --device cpu
```

Expected output:

```text
Outputs/Bonn/bonn_crowd/video_learned_motion.npz
Outputs/Bonn/bonn_crowd/video_learned_motion.json
```

## 4. Smooth Learned Motion By Region

```bash
python scripts_train/smooth_video_motion_regions.py \
  --video-npz Outputs/Bonn/bonn_crowd/video_learned_motion.npz \
  --out-video Outputs/Bonn/bonn_crowd/video_learned_motion_region.npz
```

Expected output:

```text
Outputs/Bonn/bonn_crowd/video_learned_motion_region.npz
Outputs/Bonn/bonn_crowd/video_learned_motion_region.json
```

## 5. Export Learned 4D Model

```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_crowd/video_learned_motion_region.npz \
  --output-dir Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region \
  --disp-source up \
  --no-ply \
  --no-tracks
```

Expected output:

```text
Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/model_4d.npz
Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/model_4d.mat
```

## 6. Compare Kalman And Learned

```bash
python scripts_eval/compare_4d_methods.py \
  --old-model Outputs/Bonn/bonn_crowd/4d_model_previous_motion/model_4d.npz \
  --new-model Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/model_4d.npz \
  --old-name KalmanPrevious \
  --new-name LearnedRegion \
  --out-prefix Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/experiment_baseline_kalman_learned
```

Expected output:

```text
Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/experiment_baseline_kalman_learned.csv
Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/experiment_baseline_kalman_learned.json
Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/experiment_baseline_kalman_learned.png
Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/experiment_baseline_kalman_learned_experiment_report.md
```

## 7. Package MATLAB Comparison

```bash
python scripts_eval/package_matlab_4d_comparison.py \
  --kalman-model Outputs/Bonn/bonn_crowd/4d_model_previous_motion/model_4d.npz \
  --learned-model Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/model_4d.npz \
  --output-dir Outputs/Bonn/bonn_crowd/matlab_4d_comparison
```

Expected output:

```text
Outputs/Bonn/bonn_crowd/matlab_4d_comparison/baseline_static/model_4d.mat
Outputs/Bonn/bonn_crowd/matlab_4d_comparison/kalman_previous/model_4d.mat
Outputs/Bonn/bonn_crowd/matlab_4d_comparison/learned_region/model_4d.mat
Outputs/Bonn/bonn_crowd/matlab_4d_comparison/README.md
```

## MATLAB Visualization

```matlab
addpath('scripts_eval');
view_4d_comparison_matlab('Outputs/Bonn/bonn_crowd/matlab_4d_comparison', 8, 'none', true);
```

## Crowd2 Variant

To run `bonn_crowd2`, replace every `bonn_crowd` path above with `bonn_crowd2`, and use:

```bash
python run.py --config configs/Dynamic/Bonn/bonn_crowd2.yaml
```

The rest of the commands stay the same after the path replacement.
