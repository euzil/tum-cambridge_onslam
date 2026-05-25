# Dynamic 4D Workflow (Clean Version)

## 目标
只保留一条主流程：
1. 从 `video.npz` 导出帧级 4D 点云模型（含动态点预测）。
2. 将视频中的静态部分逐帧融合到全局静态环境。
3. 可视化时：
- 静态环境累积显示；
- 动态部分按当前帧显示（不累计）；
- 预测点按当前帧显示（不累计）；
- 支持手动滑块逐帧查看。

## 当前保留脚本（scripts_eval）
- `export_4d_model.py`
- `build_global_slam_from_4d_matlab.m`
- `view_static_env_dynamic_timeline_matlab.m`
- `view_4d_model_matlab.m`
- `export_static_reconstruction.py`
- `validate_4d_model_file.py`
- `optimize_static_map_matlab.m`

说明：历史重复/诊断脚本已归档到 `scripts_eval/_archived_legacy_2026-05-25`。

## 推荐流程

### 1) 导出 4D 模型（Python）
```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_balloon_visual/video.npz \
  --output-dir Outputs/Bonn/bonn_balloon_visual/4d_model_checked \
  --uncer-thresh 0.9 \
  --stride 1 \
  --max-depth 8.0 \
  --no-ply
```

输出核心文件：
- `model_4d.npz`
- `model_4d.mat`

### 2) 可选校验（Python）
```bash
python scripts_eval/validate_4d_model_file.py \
  --model-npz Outputs/Bonn/bonn_balloon_visual/4d_model_checked/model_4d.npz \
  --model-mat Outputs/Bonn/bonn_balloon_visual/4d_model_checked/model_4d.mat
```

### 3) 从 4D 模型构建全局静态环境（MATLAB）
```matlab
addpath('scripts_eval')

stats = build_global_slam_from_4d_matlab( ...
    'Outputs/Bonn/bonn_balloon_visual/4d_model_checked/model_4d.mat', ...
    'Outputs/Bonn/bonn_balloon_visual/slam_global', ...
    0.02, 3, 0.20, 'none', true);
```

输出核心文件（在 `slam_global` 下）：
- `global_static_map.mat`
- `global_dynamic_points.mat`

### 4) 主可视化：静态全局 + 动态逐帧 + 预测逐帧 + 滑块（MATLAB）
```matlab
addpath('scripts_eval')

view_static_env_dynamic_timeline_matlab( ...
    'Outputs/Bonn/bonn_balloon_visual/slam_global', ...
    2, ...
    2, ...
    'Outputs/Bonn/bonn_balloon_visual/4d_model_checked/model_4d.mat', ...
    2);
```

行为说明：
- 静态点：显示“截至当前帧已观测到”的全局静态环境。
- 动态点：只显示当前帧动态点，不做跨帧累计。
- 预测点：只显示当前帧预测点，不做跨帧累计。
- 底部滑块：手动调帧查看。

### 5) 仅查看 `model_4d.mat`（MATLAB）
```matlab
addpath('scripts_eval')

view_4d_model_matlab( ...
    'Outputs/Bonn/bonn_balloon_visual/4d_model_checked/model_4d.mat', ...
    8, true, 'none', true)
```

该视图已支持：
- 滑块逐帧；
- `Play/Pause`；
- 键盘 `←/→` 单帧切换，`Space` 播放/暂停。

## 参数建议（Bonn balloon）
- `voxelSize`: `0.02`
- `minObs`: `3`
- `maxColorStd`: `0.20`
- `axisMode`: `'none'`（若坐标轴方向不对再改）
- 可视化点大小：静态 `2`、动态 `2~8`、预测 `2~10`

## 低分辨率复现（更稀疏、更清楚）
和主流程完全一致，只改采样/体素参数。

### 1) 导出低分辨率 4D（Python）
```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_balloon_visual/video.npz \
  --output-dir Outputs/Bonn/bonn_balloon_visual/4d_model_lowres \
  --disp-source low \
  --uncer-thresh 0.9 \
  --stride 4 \
  --max-depth 8.0 \
  --static-voxel-size 0.06 \
  --static-min-obs 2 \
  --static-max-color-std 0.25 \
  --no-ply
```

### 2) 构建低分辨率全局环境（MATLAB）
```matlab
addpath('scripts_eval')

stats = build_global_slam_from_4d_matlab( ...
    'Outputs/Bonn/bonn_balloon_visual/4d_model_lowres/model_4d.mat', ...
    'Outputs/Bonn/bonn_balloon_visual/slam_global_lowres', ...
    0.05, 2, 0.25, 'none', true);
```

### 3) 低分辨率可视化（MATLAB）
```matlab
addpath('scripts_eval')

view_static_env_dynamic_timeline_matlab( ...
    'Outputs/Bonn/bonn_balloon_visual/slam_global_lowres', ...
    4, ...   % static size
    8, ...   % dynamic(t) size
    'Outputs/Bonn/bonn_balloon_visual/4d_model_lowres/model_4d.mat', ...
    8);      % predicted size
```

## 常见问题
- 如果点云方向错：优先在可视化函数里改 `axisMode`（如 `'flip_z'`）。
- 如果动态点太稀或太多：先调整导出时 `--uncer-thresh`。
- 如果 MATLAB 卡顿：减小点大小，或在前处理阶段加大体素大小。
