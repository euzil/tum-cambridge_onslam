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

### 7. CUDA Motion-Compensated BA

当前已经从旧的 target offset 方案升级为 CUDA BA 内的动态点运动变量方案。

相关文件：

- `src/depth_video.py`
- `src/factor_graph.py`
- `src/dynamic_bridge.py`
- `src/lib/droid.cpp`
- `src/lib/droid_kernels.cu`
- `droid_backends.cpython-310-x86_64-linux-gnu.so`

核心思想：

```text
静态 BA residual:
  target_ij - project(T_j * T_i^-1 * X_i)

动态 BA residual:
  target_ij - project(T_j * T_i^-1 * X_i + dynamic_motion_i)
```

其中 `dynamic_motion_i` 是源关键帧低分辨率像素上的独立 3D motion variable，坐标系为目标关键帧相机坐标系。它不是相机 pose，也不是 depth，而是动态物体点从源帧到下一关键帧的局部 3D 位移。

预测给 BA 提供两个量：

```text
X_i       当前动态点世界坐标
X_pred_j 预测下一关键帧世界坐标

motion_prior = R_cw_j * (X_pred_j - X_i)
motion_mask  = 动态像素及其邻域权重
```

实现方式：

1. `OnlineDynamicPredictor.update()` 保存当前动态点的源帧像素、世界坐标和预测位置。
2. 下一关键帧刚加入 `DepthVideo` 后，`apply_pending_feedback()` 调用 `_write_motion_compensation()`。
3. `_write_motion_compensation()` 写入 CUDA BA 需要的动态状态：
   - `video.dynamic_motions[source_kf]`: motion 变量初值，初始化为预测 prior
   - `video.dynamic_motion_priors[source_kf]`: 预测 prior
   - `video.dynamic_motion_masks[source_kf]`: 动态像素权重/mask
4. `DepthVideo.ba()` 把上述 tensor 传给 `droid_backends.ba()`。
5. `src/lib/droid.cpp` 扩展 pybind 接口，把动态 motion tensor 和优化参数传入 `ba_cuda()`。
6. `projective_transform_kernel()` 在相邻正向边 `ii -> ii+1` 上把 `dynamic_motions[ii]` 加到投影点上。
7. `dynamic_motion_accum_kernel()` 计算 residual 对每个动态点 3D motion variable 的近似对角 Hessian 和梯度。
8. `dynamic_motion_retr_kernel()` 以预测 prior 为正则项，执行独立动态 motion 变量更新。

CUDA 中的动态 motion 更新近似为：

```text
delta_m = (J_m^T W r + lambda * mask * (prior - motion))
          / (J_m^T W J_m + lambda * mask + damping)

motion <- motion + lr * delta_m
```

当前作用范围：

- 只对相邻正向边 `ii -> ii+1` 生效，避免把长时间预测直接用于远距离 BA。
- 每个低分辨率动态像素有一个独立 3D motion variable。
- 现在是 alternating update：先按当前 motion 做 BA 投影/pose/depth 更新，再更新 dynamic motion variable。
- 还不是完整联合 Schur 系统中的动态点 block Hessian；但已经进入底层 CUDA BA residual，而不是只改 uncertainty。

配置：

```yaml
dynamic_prediction:
  motion_comp_ba: True
  motion_comp_weight: 1.0
  motion_comp_radius: 1
  motion_comp_max_px: 8.0
  motion_comp_prior_weight: 0.1
  motion_comp_lr: 0.5
  motion_comp_damping: 0.001
```

编译命令：

```bash
CUDA_HOME=/home/youran/miniconda3/envs/droid-w \
PATH=/home/youran/miniconda3/envs/droid-w/bin:$PATH \
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 \
/home/youran/miniconda3/envs/droid-w/bin/python setup.py build_ext --inplace
```

验证命令：

```bash
LD_LIBRARY_PATH=/home/youran/miniconda3/envs/droid-w/lib:/home/youran/miniconda3/envs/droid-w/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
/home/youran/miniconda3/envs/droid-w/bin/python - <<'PY'
import droid_backends
print("droid_backends import ok")
print("has ba:", hasattr(droid_backends, "ba"))
PY
```

### 8. 评价目标调整：从相机 RMSE 转向动态物体 4D 精度

实验结论：

```text
相机轨迹 RMSE 对动态物体建模并不敏感。
```

在 Bonn balloon 上，动态模块对相机 RMSE 的影响非常小，甚至可能因为动态点 residual 过强而拉坏静态相机轨迹。因此当前版本不再把 ATE/RMSE 作为动态模块的主指标。

当前主目标改为：

```text
动态物体 4D 建模精度
= 动态区域在时间上的点云连续性
+ 动态物体运动场是否合理
+ 物体级轨迹是否稳定
```

仍然保留 RMSE 对比脚本用于检查是否破坏相机轨迹：

- `scripts_eval/compare_dynamic_improvement.py`

用途：

- 检查 dynamic run 是否明显破坏 camera trajectory。
- 作为辅助指标，不再作为 4D 动态物体建模主指标。

示例：

```bash
python scripts_eval/compare_dynamic_improvement.py \
  --baseline-dir Outputs/Bonn/bonn_balloon_baseline \
  --dynamic-dir Outputs/Bonn/bonn_balloon \
  --dataset-dir datasets/Bonn/rgbd_bonn_balloon \
  --out Outputs/Bonn/bonn_balloon/dynamic_improvement_summary.png
```

### 9. 动态 motion 诊断与 4D 一致性评价

#### 9.1 CUDA motion variable 诊断

新增脚本：

- `scripts_eval/inspect_dynamic_motion_ba.py`

用途：

- 检查 `video.npz` 中是否保存了：
  - `dynamic_motions`
  - `dynamic_motion_priors`
  - `dynamic_motion_masks`
- 统计 motion variable 是否真的被 BA 修改。

示例：

```bash
python scripts_eval/inspect_dynamic_motion_ba.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz
```

关键输出：

```text
active pixels
motion mag_mean / mag_p90 / mag_max
prior  mag_mean / mag_p90 / mag_max
delta  mag_mean / mag_p90 / mag_max
```

含义：

- `motion`: CUDA BA 当前动态运动变量。
- `prior`: Kalman / 在线预测提供的先验运动。
- `delta = motion - prior`: BA 对预测运动的修正量。

如果 `delta` 接近 0，说明 BA 几乎只是接受预测先验；如果 `delta` 非 0，说明 CUDA residual 正在修正动态 motion。

#### 9.2 点级 dynamic 4D 一致性

新增脚本：

- `scripts_eval/evaluate_dynamic_4d_quality.py`

评价方式：

```text
static carry-forward:
  t 帧动态点不动，直接和 t+1 动态点云做最近邻匹配

motion-compensated:
  t 帧动态点 + dynamic_motions，再和 t+1 动态点云做最近邻匹配
```

如果 motion variable 有效，motion-compensated NN error 应小于 static carry-forward NN error。

示例：

```bash
python scripts_eval/evaluate_dynamic_4d_quality.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --out-csv Outputs/Bonn/bonn_balloon/dynamic_4d_quality.csv \
  --uncer-thresh 0.9
```

当前 Bonn balloon 点级结果大致为：

```text
static carry-forward NN error: 0.111386 m
motion-compensated NN error: 0.111500 m
relative dynamic 4D consistency gain: -0.10%
```

结论：

```text
点级独立 motion variable 已经被优化，但对动态物体点云时序一致性的帮助很弱。
```

这也是后续转向 object-level fusion 的原因。

### 10. 普通帧级 4D 点云模型导出

新增脚本：

- `scripts_eval/export_4d_model.py`
- `scripts_eval/view_4d_model.py`
- `scripts_eval/view_4d_model_matlab.m`

导出内容：

- `model_4d.npz`
- `model_4d.mat`
- `dynamic_tracks.npz`
- `dynamic_tracks.mat`
- `metadata.json`
- `ply_frames/*.ply`

当前 `model_4d.npz/.mat` 中包含：

```text
points                 Nx3 当前帧世界坐标点云
colors                 Nx3 RGB
dynamic                Nx1 动态标记
uncertainty            Nx1 uncertainty
motion_world           Nx3 动态点世界坐标运动向量
predicted_next_points  Nx3 points + motion_world
motion_valid           Nx1 是否有 CUDA dynamic motion
frame_start_1/end_1    MATLAB 1-based frame range
poses_c2w              每帧相机位姿
intrinsics             每帧内参
```

示例：

```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --output-dir Outputs/Bonn/bonn_balloon/4d_model_motion \
  --uncer-thresh 0.9 \
  --stride 1 \
  --max-depth 8.0 \
  --no-ply
```

Python 播放：

```bash
python scripts_eval/view_4d_model.py \
  --model-npz Outputs/Bonn/bonn_balloon/4d_model_motion/model_4d.npz \
  --dynamic-red \
  --fps 8
```

MATLAB 使用：

```matlab
addpath('scripts_eval')
view_4d_model_matlab('Outputs/Bonn/bonn_balloon/4d_model_motion/model_4d.mat', 8, true)
```

注意：

```text
这是 4D 点云动画模型，不是 mesh / NeRF / 连续表面模型。
4D = 3D 点云 + 时间帧序列。
```

### 11. Object-Level Dynamic 4D Fusion

为了提高动态物体建模稳定性，当前新增 object-level 4D fusion。原因是：

```text
点级 Kalman / 点级 BA motion 容易受单点噪声影响；
bonn_balloon 这类场景中的动态物体更接近整体运动物体；
因此应估计 object-level motion，而不是只依赖独立动态点 motion。
```

新增脚本：

- `scripts_eval/object_level_4d_fusion.py`
- `scripts_eval/view_object_4d_model_matlab.m`

核心流程：

```text
video.npz
  -> uncertainty + dynamic_motion_masks 提取动态区域
  -> 连通域 / 形态学膨胀得到每帧 object instance
  -> 每个 object instance 反投影成 3D 点云
  -> 用 centroid + motion_world 做相邻帧 object matching
  -> 得到 global object_id / object track
  -> 对每个 object 估计统一 motion_world
  -> 导出 object-level 4D model
```

object motion 估计：

```text
object centroid motion:
  m_centroid = centroid_{t+1} - centroid_t

CUDA BA motion:
  m_cuda = median(dynamic_motions inside object)

融合：
  object_motion = (1 - motion_blend) * m_centroid
                  + motion_blend * m_cuda
```

其中 `motion_blend=0.25` 表示主要相信 object centroid motion，同时保留少量 CUDA BA motion prior。

导出命令：

```bash
python scripts_eval/object_level_4d_fusion.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --output-dir Outputs/Bonn/bonn_balloon/object_4d_model \
  --uncer-thresh 0.9 \
  --min-area 8 \
  --min-points 8 \
  --dilate-iters 1 \
  --match-dist 0.35 \
  --motion-blend 0.25
```

输出文件：

```text
object_4d_model.npz
object_4d_model.mat
object_table.csv
object_motion_summary.csv
metadata.json
```

`object_4d_model.mat` 主要变量：

```matlab
S = load('Outputs/Bonn/bonn_balloon/object_4d_model/object_4d_model.mat');

points = S.points;                         % Nx3 当前帧动态物体点云
colors = S.colors;                         % Nx3 uint8 RGB
object_ids = S.object_ids;                 % Nx1 object track id
motion_world = S.motion_world;             % Nx3 object-level 运动向量
predicted_next_points = S.predicted_next_points; % Nx3 预测下一帧位置
frameStart = S.frame_start_1;              % MATLAB 1-based frame start
frameEnd = S.frame_end_1;                  % MATLAB 1-based frame end
objectTable = S.object_table;              % 每帧 object instance 参数
```

MATLAB 查看：

```matlab
addpath('scripts_eval')
view_object_4d_model_matlab('Outputs/Bonn/bonn_balloon/object_4d_model/object_4d_model.mat', 8, true)
```

`true` 表示显示 object motion 箭头。

当前 Bonn balloon 验证结果：

```text
object instances: 47
object tracks: 21
dynamic object points: 21342

static_mean = 0.101287 m
object_mean = 0.099880 m
object-level dynamic consistency gain = +1.39%
```

这说明 object-level fusion 已经比点级 motion 更适合当前动态物体建模任务。

### 12. 当前整体流程

当前推荐流程：

```bash
# 1. 跑动态 SLAM，生成 video.npz
python run.py --config configs/Dynamic/Bonn/bonn_balloon.yaml

# 2. 检查 CUDA dynamic motion 是否写入/被优化
python scripts_eval/inspect_dynamic_motion_ba.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz

# 3. 辅助检查相机轨迹是否被明显破坏
python scripts_eval/compare_dynamic_improvement.py \
  --baseline-dir Outputs/Bonn/bonn_balloon_baseline \
  --dynamic-dir Outputs/Bonn/bonn_balloon \
  --dataset-dir datasets/Bonn/rgbd_bonn_balloon \
  --out Outputs/Bonn/bonn_balloon/dynamic_improvement_summary.png

# 4. 点级 4D consistency 诊断
python scripts_eval/evaluate_dynamic_4d_quality.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --out-csv Outputs/Bonn/bonn_balloon/dynamic_4d_quality.csv \
  --uncer-thresh 0.9

# 5. 生成 object-level 动态 4D 模型
python scripts_eval/object_level_4d_fusion.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --output-dir Outputs/Bonn/bonn_balloon/object_4d_model \
  --uncer-thresh 0.9 \
  --min-area 8 \
  --min-points 8 \
  --dilate-iters 1 \
  --match-dist 0.35 \
  --motion-blend 0.25
```

MATLAB 查看最终 object-level 4D 动画：

```matlab
addpath('scripts_eval')
view_object_4d_model_matlab('Outputs/Bonn/bonn_balloon/object_4d_model/object_4d_model.mat', 8, true)
```

### 13. 静态 3D 重建导出与 MATLAB 查看

除了动态 4D 点云动画，当前还可以从 `video.npz` 导出静态背景 3D 重建。

示例命令：

```bash
/home/youran/miniconda3/envs/droid-w/bin/python scripts_eval/export_static_reconstruction.py \
  --video-npz Outputs/Bonn/bonn_balloon/video.npz \
  --output-dir Outputs/Bonn/bonn_balloon/static_3d
```

当前 Bonn balloon 输出：

```text
[static] Saved MATLAB reconstruction: Outputs/Bonn/bonn_balloon/static_3d/static_reconstruction.mat
[static] Saved PLY reconstruction: Outputs/Bonn/bonn_balloon/static_3d/static_reconstruction.ply
[static] Final static points: 8634181
```

`static_reconstruction.mat` 主要变量：

```text
points        Nx3 single，静态点云世界坐标
colors        Nx3 uint8，RGB 颜色
colors01      Nx3 single，0-1 范围 RGB
voxel_weight  Nx1 single，体素融合权重
poses_c2w     Tx4x4 single，相机位姿
intrinsics    Tx4 single，相机内参
timestamps    Tx1 single，时间戳 / 帧索引
```

MATLAB 直接显示 `.mat`：

```matlab
S = load('Outputs/Bonn/bonn_balloon/static_3d/static_reconstruction.mat');

pts = double(S.points);
rgb = double(S.colors) / 255.0;

figure('Color','k');
pcshow(pts, rgb, 'MarkerSize', 20);
axis equal;
grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Static 3D Reconstruction');
```

由于点数可能非常大，例如当前约 `8.63M` 点，MATLAB 全量显示可能较卡。建议优先使用随机降采样：

```matlab
S = load('Outputs/Bonn/bonn_balloon/static_3d/static_reconstruction.mat');

N = size(S.points, 1);
idx = randperm(N, min(500000, N));

pts = double(S.points(idx, :));
rgb = double(S.colors(idx, :)) / 255.0;

figure('Color','k');
pcshow(pts, rgb, 'MarkerSize', 20);
axis equal;
grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Static 3D Reconstruction Downsampled');
```

也可以直接读取 PLY 文件：

```matlab
pc = pcread('Outputs/Bonn/bonn_balloon/static_3d/static_reconstruction.ply');

figure('Color','k');
pcshow(pc, 'MarkerSize', 20);
axis equal;
grid on;
title('Static 3D Reconstruction PLY');
```

## 当前结论

1. 相机 RMSE 不是动态物体 4D 建模的主指标。
2. CUDA BA motion variable 已经接入并可导出，但点级 motion 对动态点云一致性提升有限。
3. 动态物体建模应更多依赖 object-level fusion，而不是只做独立点预测。
4. 当前最重要产物是：
   - `video.npz` 中的 `dynamic_motions / dynamic_motion_masks`
   - `model_4d.mat` 帧级 4D 点云动画
   - `object_4d_model.mat` object-level 动态 4D 动画
5. 当前 object-level 结果已经在 Bonn balloon 上取得正向动态一致性提升：

```text
static_mean = 0.101287 m
object_mean = 0.099880 m
gain = +1.39%
```

## 推荐下一步

1. 以 `object-level dynamic consistency gain` 为主指标调参。
2. 优先尝试：

```bash
--min-area 12 --match-dist 0.45 --motion-blend 0.0
```

3. 在 MATLAB 中检查 object track 是否稳定，是否出现同一物体 ID 断裂。
4. 如果 object ID 断裂明显，下一步应增强 object matching：
   - 引入颜色直方图相似度
   - 引入 3D IoU / Chamfer 距离
   - 引入多帧 tracklet smoothing
5. 如果点云闪烁明显，下一步应做 object-centric temporal fusion，把同一 object track 的多帧点云按 object motion 对齐后融合。
   
addpath('scripts_eval')
stats = build_global_slam_from_4d_matlab( ...
    'Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', ...
    'Outputs/Bonn/bonn_balloon/slam_global', ...
    0.02, 3, 0.20, 'flip_z', true);

view_static_env_dynamic_timeline_matlab('Outputs/Bonn/bonn_balloon/slam_global', 2, 8)

view_static_env_dynamic_timeline_matlab( ...
    'Outputs/Bonn/bonn_balloon/slam_global', ...
    2, ...   % 静态点大小
    2, ...   % 动态点大小
    'Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', ...
    2);     % 预测点大小