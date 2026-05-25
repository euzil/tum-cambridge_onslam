# DROID-W Dynamic Prediction and 4D Model Progress

## 当前目标

本分支围绕 DROID-W 的动态区域建模展开，当前主要完成两条线：

1. 将动态点预测接入主 SLAM 流程，用预测的动态区域影响后续关键帧的 uncertainty / BA 权重。
2. 基于当前 SLAM 输出 `video.npz` 导出可查看、可分析的 4D 点云模型，并支持 MATLAB 使用。

## 已完成内容

### 1. 动态点预测模块

相关文件：

- `dynamic_prediction/point_predictor.py`
- `dynamic_prediction/sliding_window.py`
- `dynamic_prediction/d4rt_bridge.py`
- `src/dynamic_bridge.py`
- `run_dynamic_prediction.py`

实现内容：

- 从 DROID-W 的 `video.npz` 中读取：
  - `images`
  - `droid_disps`
  - `intrinsics`
  - `uncertainties`
  - `poses / tum_poses`
- 根据 uncertainty 阈值提取动态点。
- 将低分辨率动态像素反投影为世界坐标 3D 点。
- 使用最近邻匹配为动态点分配跨帧 ID。
- 使用 Kalman Filter 做逐点未来位置预测。
- 支持离线生成 `dynamic_prediction.gif`。

### 2. 位姿接口修正

发现并修正了一个关键坐标系问题：

- `DepthVideo.save_video()` 中的 `poses` 是 camera-to-world 矩阵。
- 在线 buffer 中的 `tum_poses` 是 world-to-camera SE3 7 维格式。

现在 `DROIDWBridge` 会优先使用 `tum_poses`，避免把 `c2w` 当作 `w2c` 使用。

### 3. 在线预测反馈接入 SLAM

相关文件：

- `src/tracker.py`
- `src/dynamic_bridge.py`
- `configs/droid_w.yaml`

实现内容：

- Tracker 中初始化 `OnlineDynamicPredictor`。
- 每次产生新关键帧后，读取当前关键帧的 uncertainty / disparity / pose。
- 提取动态点并更新滑动窗口预测器。
- 将上一关键帧预测出的动态位置缓存起来。
- 下一关键帧刚 append 后、frontend local BA 前，将预测动态 mask 写入当前关键帧 `video.uncertainties`。

当前在线反馈本质：

```text
动态点历史轨迹 -> 预测下一关键帧位置 -> 投影到当前关键帧 -> 提高 uncertainty -> BA 降权
```

注意：这不是 motion-compensated BA。它只是预测动态区域并提前降权，因此对 RMSE 的提升上限有限。

### 4. 滑动窗口预测修正

之前的问题：

- Kalman 状态字典中保留了所有历史点。
- 在线反馈时对所有历史点做预测。
- 旧动态点长期外推，容易导致预测 mask 覆盖过大。
- 实测曾出现平均覆盖率超过 50%，这会丢失大量静态背景约束。

已修正为真正的滑动窗口策略：

- `SlidingWindowPredictor.predict()` 只预测当前最新帧仍然可见的活跃点。
- 增加 `predict_for_ids()`，可显式指定当前点 ID。
- 增加 `get_current_point_ids()` 和 `get_active_point_ids()`。
- 增加 `prune_inactive()`，删除不在滑动窗口内的旧 Kalman 状态。
- `OnlineDynamicPredictor.update()` 现在只对当前帧 `pids` 做预测，并在每帧后剪枝旧状态。

这能避免“借助所有帧推演”导致的过度反馈。

### 5. 在线反馈保护机制

新增参数：

```yaml
dynamic_prediction:
  max_feedback_points: 120
  max_feedback_coverage: 0.05
```

作用：

- 限制每帧最多反馈的预测点数。
- 限制预测 mask 最大覆盖率。
- 如果预测 mask 超过阈值，则跳过该帧反馈，避免污染 BA。

当前建议配置：

```yaml
dynamic_prediction:
  enable: True
  online_enable: True
  online_render: False
  online_make_gif: False
  online_insert_raw_frames: False

  window_size: 8
  predict_steps: 2
  uncer_thresh: 0.9

  pred_uncer_value: 3.0
  blend_alpha: 0.5
  splat_radius: 1
  max_feedback_points: 120
  max_feedback_coverage: 0.05
```

### 6. 反馈统计

在线预测会输出：

```text
dynamic_feedback_stats.csv
```

字段：

- `kf_idx`
- `n_predictions`
- `n_projected`
- `mask_pixels`
- `coverage`
- `uncer_before_max`
- `uncer_after_max`
- `applied`

推荐检查：

```bash
python - <<'PY'
import pandas as pd
p="Outputs/Bonn/bonn_balloon/dynamic_feedback_stats.csv"
df=pd.read_csv(p)
print(df.describe())
print("applied ratio:", df.applied.mean())
print("mean coverage:", df.coverage.mean())
print("max coverage:", df.coverage.max())
PY
```

目标范围：

- `mean coverage`: 0.02 到 0.06
- `max coverage`: 不超过 0.10
- `applied ratio`: 不需要为 1.0，过高反而可能说明反馈过强

### 7. RMSE 对比可视化

新增脚本：

- `scripts_eval/compare_dynamic_improvement.py`

用途：

- 对比 baseline 与 dynamic run。
- 画出 GT / baseline / dynamic 轨迹。
- 画出逐关键帧误差曲线。
- 标记哪些帧 dynamic 更好，哪些帧更差。
- 叠加 dynamic feedback coverage。

示例：

```bash
python scripts_eval/compare_dynamic_improvement.py \
  --baseline-dir bonn_balloon_baseline \
  --dynamic-dir bonn_balloon \
  --dataset-dir datasets/Bonn/rgbd_bonn_balloon \
  --out bonn_balloon/dynamic_improvement_summary.png
```

### 8. 4D 点云模型导出

新增脚本：

- `scripts_eval/export_4d_model.py`
- `scripts_eval/view_4d_model.py`

导出内容：

- `model_4d.npz`
- `dynamic_tracks.npz`
- `metadata.json`
- `ply_frames/*.ply`

示例：

```bash
python scripts_eval/export_4d_model.py \
  --video-npz bonn_balloon/video.npz \
  --output-dir bonn_balloon/4d_model \
  --uncer-thresh 0.8 \
  --stride 1 \
  --max-depth 8.0
```

播放：

```bash
python scripts_eval/view_4d_model.py \
  --model-npz bonn_balloon/4d_model/model_4d.npz \
  --dynamic-red \
  --fps 8
```

### 9. MATLAB 模型导出

`export_4d_model.py` 现在也会导出 MATLAB 文件：

- `model_4d.mat`
- `dynamic_tracks.mat`

新增 MATLAB 查看脚本：

- `scripts_eval/view_4d_model_matlab.m`

MATLAB 使用：

```matlab
addpath('scripts_eval')
view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat', 8, true)
```

直接加载：

```matlab
S = load('bonn_balloon/4d_model/model_4d.mat');

points = S.points;              % Nx3 world coordinates
colors = S.colors;              % Nx3 uint8 RGB
dynamic = logical(S.dynamic);   % Nx1 dynamic flag
uncertainty = S.uncertainty;    % Nx1 uncertainty
frameStart = S.frame_start_1;   % MATLAB 1-based frame start index
frameEnd = S.frame_end_1;       % MATLAB 1-based frame end index
```

动态轨迹：

```matlab
T = load('bonn_balloon/4d_model/dynamic_tracks.mat');

tracks = T.tracks;              % N_tracks x N_frames x 3
visibility = logical(T.visibility);
point_ids = T.point_ids;
```

## 当前结论

1. 单纯“预测动态区域并降权”对 RMSE 的提升可能不明显。
2. 过大的预测 mask 会削弱静态背景约束，可能让结果变差。
3. 在线预测应使用滑动窗口活跃点，而不是所有历史点。
4. 当前更可靠的产物是：
   - 动态点轨迹
   - 4D 点云序列
   - MATLAB 可加载的 4D 模型
5. 如果后续要显著提升 RMSE，需要进一步实现 motion-compensated BA，而不只是 uncertainty feedback。

## 推荐下一步

1. 使用当前滑动窗口修正版重新跑 Bonn balloon。
2. 检查 `dynamic_feedback_stats.csv`，确认 coverage 是否被压到合理范围。
3. 用 `compare_dynamic_improvement.py` 看逐帧误差变化。
4. 基于 `model_4d.mat` 在 MATLAB 中分析动态区域与轨迹。
5. 若目标转为更强的 SLAM 提升，需要设计动态点运动补偿 residual。
