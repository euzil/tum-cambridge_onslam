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

## 学习式像素点运动预测流程

为了替代点级 Kalman，本阶段新增一个轻量 learned pixel-flow baseline。目标是先在离线数据上证明模型能预测动态像素的下一帧位置，再考虑接回在线 SLAM。

新增文件：
- `dynamic_prediction/pixel_motion_model.py`
- `scripts_train/build_pixel_motion_dataset.py`
- `scripts_train/train_pixel_motion.py`
- `scripts_train/evaluate_pixel_motion.py`

### 1) 构建训练数据

单序列快速测试：

```bash
python scripts_train/build_pixel_motion_dataset.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --out datasets_train/bonn_balloon_pixel_motion \
  --uncer-thresh 0.8 \
  --min-dynamic-points 300 \
  --max-dynamic-points 800 \
  --max-depth 8 \
  --max-nn-dist 0.5 \
  --min-labels 10 \
  --skip-missing
```

输出：
- `samples.npz`
- `metadata.json`

每个训练样本包含：
- `inputs`: `[6,H,W]`，RGB、disparity、uncertainty、dynamic mask。
- `flows`: `[2,H,W]`，低分辨率像素位移 `(du,dv)`。
- `valids`: `[1,H,W]`，有效监督 mask。

当前 Bonn balloon 冒烟测试结果：

```text
samples=55
labels_mean=552.60
```

如果要使用 Bonn 数据集，应先为每个 Bonn 动态序列生成 `video.npz`：

```bash
for cfg in \
  configs/Dynamic/Bonn/bonn_balloon.yaml \
  configs/Dynamic/Bonn/bonn_balloon2.yaml \
  configs/Dynamic/Bonn/bonn_crowd.yaml \
  configs/Dynamic/Bonn/bonn_crowd2.yaml \
  configs/Dynamic/Bonn/bonn_moving_nonobstructing_box.yaml \
  configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2.yaml \
  configs/Dynamic/Bonn/bonn_person_tracking.yaml \
  configs/Dynamic/Bonn/bonn_person_tracking2.yaml
do
  python run.py --config "$cfg"
done
```

更有用的 Bonn-only 多序列训练集：

```bash
python scripts_train/build_pixel_motion_dataset.py \
  --video-npz \
    Outputs/Bonn/bonn_balloon/video.npz \
    Outputs/Bonn/bonn_balloon2/video.npz \
    Outputs/Bonn/bonn_crowd/video.npz \
    Outputs/Bonn/bonn_crowd2/video.npz \
    Outputs/Bonn/bonn_moving_nonobstructing_box/video.npz \
    Outputs/Bonn/bonn_moving_nonobstructing_box2/video.npz \
    Outputs/Bonn/bonn_person_tracking/video.npz \
    Outputs/Bonn/bonn_person_tracking2/video.npz \
  --out datasets_train/bonn_dynamic_pixel_motion \
  --uncer-thresh 0.8 \
  --min-dynamic-points 300 \
  --max-dynamic-points 800 \
  --max-depth 8 \
  --max-nn-dist 0.5 \
  --min-labels 10
```

当前本地已生成的 Bonn 子集构建结果：

```text
sources=2
available videos:
- Outputs/Bonn/bonn_balloon/video.npz
- Outputs/Bonn/bonn_person_tracking2/video.npz
```

推荐优先使用完整 Bonn 训练集 `datasets_train/bonn_dynamic_pixel_motion/samples.npz`，而不是只用单一 `bonn_balloon`。

### 2) 训练 Small U-Net

正式训练建议：

```bash
python scripts_train/train_pixel_motion.py \
  --dataset datasets_train/bonn_dynamic_pixel_motion/samples.npz \
  --out checkpoints/pixel_motion_bonn_dynamic \
  --epochs 100 \
  --batch-size 8 \
  --base-channels 32 \
  --lr 1e-3
```

快速冒烟测试：

```bash
python scripts_train/train_pixel_motion.py \
  --dataset datasets_train/bonn_balloon_pixel_motion/samples.npz \
  --out checkpoints/pixel_motion_bonn_balloon_sanity \
  --epochs 3 \
  --batch-size 8 \
  --base-channels 16 \
  --lr 1e-3
```

冒烟测试输出：

```text
best val EPE: 4.960599 px
```

### 3) 评估 learned flow

```bash
python scripts_train/evaluate_pixel_motion.py \
  --dataset datasets_train/bonn_dynamic_pixel_motion/samples.npz \
  --checkpoint checkpoints/pixel_motion_bonn_dynamic/best.pt \
  --batch-size 16 \
  --out checkpoints/pixel_motion_bonn_dynamic/eval.json
```

评估指标：
- `learned_epe_px_mean`：模型预测 flow 的平均 endpoint error。
- `zero_epe_px_mean`：不预测运动、flow 为 0 的 baseline。
- `learned_vs_zero_gain_percent`：相对 zero-flow baseline 的提升率。

3 epoch 冒烟测试仅用于验证代码链路：

```text
learned EPE: 4.948573 px
zero-flow EPE: 4.950517 px
relative gain: +0.04%
```

这个结果不能代表最终模型效果；正式判断需要使用 100 epoch 左右训练，并最好加入更多动态序列。

### 4) 将 learned flow 写回 4D motion

像素 flow 训练有效后，可以把 checkpoint 应用到 `video.npz`，生成 learned dynamic motion 版本：

```bash
python scripts_train/apply_pixel_motion_to_video.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --checkpoint checkpoints/pixel_motion_bonn_dynamic/best.pt \
  --out-video Outputs/Bonn/bonn_balloon/video_learned_motion_nextdepth_filt.npz \
  --uncer-thresh 0.8 \
  --min-dynamic-points 300 \
  --max-dynamic-points 800 \
  --max-depth 8 \
  --max-flow-px 8 \
  --max-motion-m 0.5 \
  --depth-mode next \
  --batch-size 16 \
  --device cpu
```

`--depth-mode next` 会使用预测像素在下一帧的深度和两帧 pose 计算 3D motion；`--max-motion-m` 用来过滤遮挡和错误深度导致的异常大 motion。

导出 learned 4D 模型：

```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_balloon/video_learned_motion_nextdepth_filt.npz \
  --output-dir Outputs/Bonn/bonn_balloon/4d_model_learned_motion_nextdepth_filt \
  --uncer-thresh 0.8 \
  --stride 1 \
  --max-depth 8 \
  --disp-source up \
  --no-ply
```

评估：

```bash
python scripts_eval/evaluate_dynamic_method.py \
  --model-npz Outputs/Bonn/bonn_balloon/4d_model_learned_motion_nextdepth_filt/model_4d.npz \
  --output-dir Outputs/Bonn/bonn_balloon \
  --max-nn-dist 0.5 \
  --out-prefix Outputs/Bonn/bonn_balloon/4d_model_learned_motion_nextdepth_filt/dynamic_method_eval

python scripts_eval/diagnose_4d_axis_error.py \
  --model-npz Outputs/Bonn/bonn_balloon/4d_model_learned_motion_nextdepth_filt/model_4d.npz \
  --max-nn-dist 0.5
```

当前 learned 版本结果：

```text
pixel-flow learned EPE: 4.290569 px
pixel-flow zero EPE: 5.737912 px
pixel-flow gain: +25.22%

video learned motion:
active pixels: 13674
active ratio: 7.9485%
flow mean: 3.0080 px
motion mean: 0.231909 m
motion p90: 0.427529 m

4D learned next-depth filtered:
matched predicted points: 829526
NN distance mean: 0.077701 m
NN distance median: 0.049229 m
NN distance p90: 0.184649 m
axis signed mean: x=-0.002443, y=-0.000411, z=-0.003878 m

dynamic method eval:
motion_nn_m_mean: 0.079990
static_nn_m_mean: 0.078459
motion_vs_static_gain_percent: -1.95%
```

结论：learned pixel-flow 在像素预测上明显优于 zero-flow，并且 learned 4D 的绝对 NN 误差已经降到 `0.0777 m`；但在当前逐点 static carry-forward 对照指标上仍未转正。因此 learned 方法已经比点级 Kalman 更有潜力，但若要得到稳定正收益，下一步应继续做 object-level/region-level 运动约束或更强的监督标签。

### 5) Region/Object-level motion smoothing

为了让同一个动态区域共享更稳定的 motion，新增区域级平滑脚本：

```bash
python scripts_train/smooth_video_motion_regions.py \
  --video-npz Outputs/Bonn/bonn_balloon/video_learned_motion_nextdepth_filt.npz \
  --out-video Outputs/Bonn/bonn_balloon/video_learned_motion_region.npz \
  --min-component-pixels 12 \
  --dilate-iters 1 \
  --mode median \
  --trim-quantile 0.9 \
  --blend 0.75
```

该脚本会：
- 对每帧 `dynamic_motion_masks` 做连通区域；
- 丢弃过小区域；
- 对每个区域内的 learned 3D motion 做 trimmed median；
- 将点级 motion 与区域级 motion 混合，得到更稳定的 region-level motion。

导出 region-level 4D 模型：

```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_balloon/video_learned_motion_region.npz \
  --output-dir Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region \
  --uncer-thresh 0.8 \
  --stride 1 \
  --max-depth 8 \
  --disp-source up \
  --no-ply
```

评估：

```bash
python scripts_eval/evaluate_dynamic_method.py \
  --model-npz Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region/model_4d.npz \
  --output-dir Outputs/Bonn/bonn_balloon \
  --max-nn-dist 0.5 \
  --out-prefix Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region/dynamic_method_eval

python scripts_eval/diagnose_4d_axis_error.py \
  --model-npz Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region/model_4d.npz \
  --max-nn-dist 0.5
```

当前最佳 region-level learned 结果：

```text
video learned region:
active pixels: 13035
active ratio: 7.5771%
components mean: 2.02
motion mean: 0.100019 m

4D learned region:
matched predicted points: 792619
NN distance mean: 0.065516 m
NN distance median: 0.038219 m
NN distance p90: 0.159599 m
axis signed mean: x=-0.002681, y=-0.000651, z=-0.002358 m

dynamic method eval:
motion_nn_m_mean: 0.066065
static_nn_m_mean: 0.076519
motion_vs_static_gain_percent: +13.66%
```

这是当前最好的 4D 动态建模结果。它说明：单点 Kalman 不够稳定，learned pixel-flow 有效，而 region/object-level motion smoothing 能把 learned motion 转化为真正有正收益的 4D 动态地图。

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
