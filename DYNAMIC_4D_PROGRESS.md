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
- `evaluate_dynamic_method.py`

说明：历史重复/诊断脚本已归档到 `scripts_eval/_archived_legacy_2026-05-25`。

## 点级 Kalman 在线预测版本总结

本阶段实现的是 point-level dynamic prediction：
1. 在 `tracker.py` 中启用在线 `OnlineDynamicPredictor`。
2. 从每个关键帧的不确定性图和深度图中选取动态候选点。
3. 对动态候选点做跨帧 ID 匹配。
4. 使用滑动窗口 Kalman 预测下一关键帧动态点位置。
5. 将预测结果反馈到两条在线路径：
   - 写入 `uncertainties`，改变 BA 权重；
   - 写入 `dynamic_motions / dynamic_motion_priors / dynamic_motion_masks`，供 CUDA BA 的动态 motion variable 使用。
6. SLAM 结束后，`export_4d_model.py` 将动态点、预测点和 motion field 导出到 `model_4d.npz/.mat`。

### 已完成的关键修复
- 在线预测不再只作为离线可视化，而是在下一关键帧 frontend BA 前写入反馈。
- 同一对 `(source_kf, target_kf)` 只应用一次反馈，避免重复提高 uncertainty。
- 动态候选点加入 `min_dynamic_points / max_dynamic_points`，避免每帧只有十几个点。
- 跨帧 ID 匹配从“多对一最近邻”改为“一对一贪心匹配”，避免多个点共享同一 ID 后被滑动窗口字典覆盖。
- feedback mask 超过覆盖率上限时，不再整帧跳过，而是自动缩小 splat 半径或采样到允许范围。
- 增加在线诊断日志：
  - `dynamic_id_match_stats.csv`
  - `dynamic_feedback_stats.csv`
  - `dynamic_motion_feedback_stats.csv`
  - `dynamic_ba_edge_stats.csv`

### 当前数值结果
基于 `Outputs/Bonn/bonn_balloon/4d_model_checked/model_4d.npz` 和 `Outputs/Bonn/bonn_balloon` 日志：

```text
online predictions mean: 373.442
feedback applied ratio: 1.000
BA adjacent mask ratio: 0.724
motion valid dynamic points: 1870348
4D NN error motion/static: 0.089454 / 0.087068 m
4D temporal consistency gain: -2.74%
axis signed mean [m]: x=+0.000728, y=+0.000430, z=-0.005181
```

这说明当前实现已经完成了“预测进入在线 SLAM 与 4D 导出”的链路：
- 平均每帧有数百个预测点；
- uncertainty feedback 已稳定应用；
- dynamic motion prior 已稳定写入；
- BA 中相邻动态边有明显覆盖；
- 导出的 4D 模型没有明显坐标轴系统偏差。

但是，点级 Kalman motion 在当前点级最近邻评估中没有超过 static carry-forward baseline：
- `motion_nn_m_mean = 0.089454 m`
- `static_nn_m_mean = 0.087068 m`
- 相对增益为 `-2.74%`

因此，本阶段结论是：point-level Kalman 预测链路有效接入，但点级预测本身不是当前 4D 动态物体建模的最优核心方法。

### 为什么点级 Kalman 效果有限
- RGB-D 点云中的“同一个像素点/表面点”跨帧并不稳定，视角变化会导致重采样。
- 深度噪声会让单点 3D 坐标抖动，Kalman 容易学习到噪声运动。
- 最近邻 ID 匹配是几何近邻，不是真实物理点对应。
- 动态物体通常近似刚体或局部刚体运动，点级独立预测会破坏物体整体一致性。
- 点级 NN 评估对表面采样密度、遮挡、深度厚度很敏感。

下一步建议转向 object-level 或 region-level motion：
1. 用动态 mask 聚成物体区域。
2. 对每个动态物体估计整体刚体/近刚体 motion。
3. 用物体级 motion 生成 4D 动态轨迹。
4. 使用 Chamfer distance、centroid error、temporal IoU、object trajectory consistency 等指标评估。

## 数值评估方法

新增统一评估脚本：

```bash
python scripts_eval/evaluate_dynamic_method.py \
  --model-npz Outputs/Bonn/bonn_balloon/4d_model_checked/model_4d.npz \
  --output-dir Outputs/Bonn/bonn_balloon \
  --max-nn-dist 0.5 \
  --out-prefix Outputs/Bonn/bonn_balloon/4d_model_checked/dynamic_method_eval
```

输出文件：
- `dynamic_method_eval.csv`：指标总表，适合放实验表格。
- `dynamic_method_eval.json`：完整指标，适合记录实验配置。
- `dynamic_method_eval_per_frame.csv`：逐帧 dynamic 4D consistency。

核心指标解释：
- `feedback_predictions_mean`：每帧平均在线预测点数量。
- `feedback_applied_ratio`：uncertainty feedback 实际应用比例。
- `motion_applied_ratio`：dynamic motion prior 实际写入比例。
- `ba_adjacent_mask_ratio`：BA 相邻边中可使用 dynamic mask 的比例。
- `model_motion_valid_points`：导出 4D 模型中有有效 motion 的动态点数量。
- `static_nn_m_mean`：不使用 motion，直接将当前动态点与下一帧动态点做最近邻的误差。
- `motion_nn_m_mean`：使用预测 motion 后，与下一帧动态点做最近邻的误差。
- `motion_vs_static_gain_percent`：`static_nn_m_mean` 相对 `motion_nn_m_mean` 的提升率。
- `axis_*_signed_mean_m`：三轴有符号误差，用于判断是否存在坐标轴系统偏差。

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
