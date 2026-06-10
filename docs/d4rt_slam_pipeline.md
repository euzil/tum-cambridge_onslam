# D4RT-SLAM Pipeline and Experiments

## 1. 目标

本方案的目标是将 Open-D4RT 作为 SLAM 前端直接嵌入 DROID-W / WildGS-SLAM 的几何优化流程中。

核心思想不是后处理、拼接、调权重或引入不确定性，而是：

```text
用 D4RT 的 4D correspondence / depth / optional camera prediction
替代 DROID-W 原来的 learned optical update frontend，
并将 D4RT 输出直接转化为 SLAM 的重投影因子。
```

最终形式：

```text
D4RT 负责生成几何观测
DROID-W 后端负责 BA、关键帧管理和重建
```

也可以概括为：

```text
D4RT-driven reprojection-factor SLAM
```

## 2. 总体 Pipeline

```text
RGB / RGB-D video
    ↓
D4RT window inference
    ↓
4D tracks / depth / optional camera initialization
    ↓
D4RT factor construction
    ↓
DROID-W style BA optimization
    ↓
optimized camera poses + optimized depths
    ↓
Gaussian / point cloud reconstruction
    ↓
ATE / reconstruction / tracking evaluation
```

逐帧流程：

```text
输入第 t 帧
    ↓
1. 关键帧选择
    ↓
2. 收集 sliding window: [t-K+1, ..., t]
    ↓
3. D4RT 对 window 做 4D prediction
    ↓
4. 对每条 SLAM factor edge (i, j) 查询:
       source pixel: u in frame i
       target timestep: j
       camera frame: j
    得到:
       q_d4rt(i, u -> j)
       D4RT_depth_i(u)
       valid_mask
    ↓
5. 构建几何残差:
       project(T_j * T_i^-1, D_i(u), u) ≈ q_d4rt(i, u -> j)
    ↓
6. BA 优化:
       poses T
       depths D
       optional scale / shift
    ↓
7. 更新 video.poses / video.disps
    ↓
8. mapping / reconstruction 使用优化后的 pose 和 depth
```

## 3. 核心目标函数

最终 SLAM 后端应优化：

```text
min_{T, D}
Σ_{(i,j),u} || π(T_j * T_i^-1, D_i(u), u) - q_d4rt(i,u→j) ||²
+
λd Σ_{i,u} || D_i(u) - D4RT_depth_i(u) ||²
+
λs Σ_i smooth(D_i)
```

其中：

```text
T_i                 SLAM 优化的相机位姿
D_i(u)              SLAM 优化的深度
q_d4rt(i,u→j)       D4RT 预测的像素或 3D tracking target
D4RT_depth_i(u)     D4RT 预测的深度
π                   投影函数
```

注意：

```text
不使用 learned weight
不使用 uncertainty-aware update
最多保留 binary valid_mask
```

`valid_mask` 只表示基本几何合法性：

```text
D4RT query 是否成功
目标点是否出界
深度是否合法
track 是否存在
```

它不是不确定性机制，而是避免非法残差进入 BA。

## 4. 与原 DROID-W 的区别

原 DROID-W：

```text
image pair
    ↓
correlation volume
    ↓
update_op predicts delta and weight
    ↓
target = coords1 + delta
    ↓
BA optimizes pose / depth
```

D4RT-SLAM：

```text
image window
    ↓
D4RT predicts 4D correspondence and depth
    ↓
target = q_d4rt(i,u→j)
    ↓
binary valid mask
    ↓
BA optimizes pose / depth
```

也就是说：

```text
DROID learned update operator 被 D4RT 4D tracking frontend 替代。
```

## 5. 模块设计

### 5.1 D4RTFrontend

建议新增：

```text
src/modules/d4rt_frontend.py
```

职责：

```python
class D4RTFrontend:
    def __init__(self, cfg):
        """Load Open-D4RT model and checkpoint."""

    def encode_window(self, frame_ids, images, intrinsics=None):
        """Run D4RT inference on the current sliding window."""

    def query_pair(self, i, j, grid_uv):
        """Query D4RT correspondence from frame i to frame j."""
        return target_2d, depth_i, valid_mask

    def query_depth(self, i):
        """Return D4RT depth for frame i."""
        return depth_i, valid_mask
```

第一阶段建议使用离线缓存，降低调试难度：

```text
先对整段视频运行 D4RT
保存:
    d4rt_tracks.npz
    d4rt_depths.npz
    d4rt_valids.npz

SLAM 运行时读取这些观测
```

### 5.2 DepthVideo 扩展

在 `src/depth_video.py` 中扩展 D4RT 缓存：

```python
self.d4rt_depths
self.d4rt_depths_up
self.d4rt_valid_mask
self.d4rt_targets
```

用途：

```text
新关键帧进入时:
    用 D4RT depth 初始化 disps / disps_up

BA 前:
    从缓存读取 D4RT correspondence target

BA 后:
    写回优化后的 poses / disps
```

### 5.3 D4RTFactorGraph

建议新增：

```text
src/factor_graph_d4rt.py
```

保留原 FactorGraph 中的：

```text
ii / jj edge 管理
add_factors
rm_factors
add_neighborhood_factors
add_proximity_factors
rm_keyframe
BA 调用
```

删除或绕开：

```text
CorrBlock
AltCorrBlock
update_op
net
inp
corr
learned delta
learned weight
uncertainty update
```

核心 `update()` 逻辑：

```python
coords1, mask_geom = self.video.reproject(self.ii, self.jj)

target_d4rt, depth_d4rt, mask_d4rt = self.d4rt.query_edges(
    self.ii, self.jj, self.coords0
)

target = target_d4rt
weight = torch.ones_like(target)

valid = mask_geom & mask_d4rt
weight = weight * valid[..., None]

self.video.ba(
    target,
    weight,
    damping,
    self.ii,
    self.jj,
    t0,
    t1,
    ...
)
```

如果要在表达上彻底避免“权重机制”，可以将 `weight` 仅视作 binary residual mask：

```text
1 = 使用该残差
0 = 不使用该残差
```

### 5.4 D4RT Depth Initialization

新关键帧插入时：

```text
D4RT_depth_i
    ↓
video.disps[i] = 1 / D4RT_depth_i
video.disps_up[i] = 1 / D4RT_depth_i_up
```

这使 D4RT 同时提供：

```text
tracking target
depth initialization
```

### 5.5 D4RT Camera Pose Initialization

如果 Open-D4RT 能输出 camera pose，可用于初始化：

```text
video.poses[t] = D4RT_camera_pose[t]
```

但最终轨迹必须仍由 SLAM BA 输出：

```text
D4RT camera prediction is used only for initialization.
Final camera trajectory is optimized by the SLAM backend.
```

这样可以避免方法被理解为直接读取 D4RT pose。

### 5.6 Reconstruction / Mapping

第一阶段尽量不改 reconstruction / mapping：

```text
optimized poses + optimized depths
    ↓
原 DROID-W / WildGS-SLAM mapping
    ↓
point cloud / Gaussian reconstruction
```

方法贡献集中在：

```text
D4RT-driven geometry factors improve SLAM pose/depth optimization.
```

## 6. 完整运行流程

每个新帧进入时：

```text
1. 读取 RGB / depth / intrinsics
2. 判断是否插入关键帧
3. 若插入关键帧:
       保存 image
       D4RT 预测该帧 depth
       初始化 video.disps
4. 更新 sliding window
5. D4RT 对 window 编码
6. 构建 factor edges:
       邻近边: t ↔ t-1, t-2, ...
       长程边: D4RT track consistency 较好的边
7. 对每条 edge 查询 D4RT target
8. 用 D4RT target 替代 DROID target
9. BA 优化 pose / depth
10. 更新 reconstruction
11. 输出轨迹和重建结果
```

## 7. 实验步骤

### Experiment 0: Baseline

目的：确认 DROID-W baseline 已复现。

```text
Method:
    DROID-W baseline

Datasets:
    TUM RGB-D dynamic
    Bonn dynamic
    optional: DyCheck / YouTube dynamic

Metrics:
    ATE RMSE
    RPE trans
    RPE rot
    tracking failure count
    reconstruction visual quality
```

输出：

```text
eval/baseline/xxx.json
eval/baseline/xxx.csv
eval/baseline/xxx.report.md
```

### Experiment 1: D4RT Target Only

这是第一组关键实验。

关闭：

```text
DROID update_op target
DROID learned delta
learned weight
uncertainty update
```

保留：

```text
DROID-W BA
keyframe management
原 depth initialization
mapping / reconstruction
```

替换：

```text
target = D4RT correspondence target
residual mask = binary valid mask
```

验证问题：

```text
只替换几何观测后，ATE 是否下降？
BA 是否稳定？
轨迹是否比直接 D4RT camera pose 更好？
```

对比：

```text
DROID-W baseline
D4RT raw camera prediction
D4RT target + SLAM BA
```

### Experiment 2: D4RT Depth Initialization

在 Experiment 1 基础上增加：

```text
新关键帧 depth 初始化:
    video.disps = 1 / D4RT_depth
```

对比：

```text
D4RT target only
D4RT target + D4RT depth init
```

观察：

```text
ATE 是否下降
BA 收敛速度是否更快
弱纹理区域深度是否更稳定
```

### Experiment 3: D4RT Depth Regularization

进一步加入目标函数中的 depth term：

```text
E_depth = || D_slam - D4RT_depth ||²
```

这不是 uncertainty / weight，而是明确的几何先验。

对比：

```text
D4RT target + depth init
D4RT target + depth init + depth regularization
```

建议做 lambda ablation：

```text
λd = 0
λd = 0.01
λd = 0.05
λd = 0.1
λd = 0.5
```

### Experiment 4: D4RT Edge Selection

替换 factor graph 的选边逻辑。

原始 DROID-W：

```text
根据 video.distance / proximity 选边
```

D4RT 版本：

```text
根据 D4RT track 可用性和跨帧一致性选边
```

例子：

```text
edge_score(i,j) =
    average_valid_track_ratio(i,j)
    - α * average_track_cycle_error(i,j)
```

保留高质量边：

```text
i ↔ i-1
i ↔ i-2
long-range edges with high D4RT consistency
```

对比：

```text
original edge selection
D4RT edge selection
D4RT edge selection + loop edges
```

### Experiment 5: Full D4RT-SLAM

完整版本：

```text
D4RT target
+ D4RT depth init
+ D4RT depth regularization
+ D4RT edge selection
+ optional D4RT pose init
+ DROID-W BA / reconstruction
```

对比表：

```text
DROID-W baseline
D4RT raw
D4RT target only
D4RT target + depth
D4RT target + depth + edge
Full D4RT-SLAM
```

### Experiment 6: Against Post-processing Baseline

目的：证明该方法不是拼接或后处理。

构造一个后处理 baseline：

```text
DROID-W baseline trajectory
+ D4RT reconstruction post-processing
```

或：

```text
D4RT predicted point cloud aligned to DROID-W pose
```

与真正嵌入 BA 的方法比较：

```text
D4RT inside BA
```

如果 `D4RT inside BA` 的 ATE 更低，说明 D4RT 确实改变了 SLAM 优化过程。

## 8. 推荐指标

轨迹：

```text
ATE RMSE
ATE mean / median
RPE trans
RPE rot
scale drift
tracking lost count
```

重建：

```text
depth RMSE
depth AbsRel
Chamfer distance
F-score
visual comparison
```

动态场景专项：

```text
static-background reprojection error
dynamic-object residual contamination
moving-object influence on camera pose
```

效率：

```text
D4RT inference time
BA time
total FPS
GPU memory
```

## 9. 推荐数据集顺序

先小后大：

```text
1. TUM RGB-D fr3/walking_xyz
2. TUM RGB-D fr3/walking_halfsphere
3. TUM RGB-D fr3/sitting_xyz
4. Bonn person_tracking
5. Bonn moving_nonobstructing_box
6. Bonn crowd
7. DyCheck / YouTube dynamic
```

优先选择动态人、动态物体明显的序列，因为 D4RT 的优势更容易体现。

## 10. 推荐开发顺序

```text
Phase 1:
    离线运行 D4RT，保存 tracks / depths / valids

Phase 2:
    编写 D4RTFactorGraph，用离线 tracks 替换 target

Phase 3:
    运行 Experiment 1，确认 BA 可以收敛

Phase 4:
    加 D4RT depth initialization

Phase 5:
    加 D4RT depth regularization

Phase 6:
    加 D4RT edge selection

Phase 7:
    改成在线或半在线 D4RT inference
```

不要一开始就在线集成 D4RT。先使用 `.npz` 缓存把 SLAM 优化链路跑通，调试会简单很多。

## 11. 推荐配置项

建议新增配置：

```yaml
tracking:
  d4rt:
    activate: true
    mode: "offline"          # offline | online
    cache_dir: "output/d4rt_cache"
    checkpoint: "Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt"
    model_config: "Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml"
    window_size: 32
    query_stride: 8
    use_target: true
    use_depth_init: false
    use_depth_reg: false
    use_pose_init: false
    use_edge_selection: false
    depth_reg_lambda: 0.05
```

第一阶段推荐：

```yaml
tracking:
  d4rt:
    activate: true
    mode: "offline"
    use_target: true
    use_depth_init: false
    use_depth_reg: false
    use_pose_init: false
    use_edge_selection: false
```

当前实现支持的离线 D4RT cache 格式：

```text
npz keys:
    targets / target_2d / tracks / d4rt_targets
        shape: [T, T, H, W, 2] 或 [T, T, 2, H, W]

    valids / valid / masks / d4rt_valids
        shape: [T, T, H, W]，可选

    depths / depth / d4rt_depths
        shape: [T, H, W] 或 [T, 1, H, W]，可选

    poses / camera_poses / d4rt_poses
        shape: [T, 7]，可选
```

坐标约定：

```text
target_coord_scale: "auto"
    自动判断 target 是否是 full-resolution pixel coordinate。
    如果是，会缩放到 DROID tracking grid。

target_coord_scale: "full_to_tracking"
    强制从 full-resolution pixel coordinate 缩放到 tracking grid。

target_coord_scale: "normalized_to_tracking"
    将 [-1, 1] 归一化坐标转换到 tracking grid。

target_coord_scale: "none"
    假设 cache 中已经是 DROID tracking-grid 坐标。
```

## 12. 最终方法表述

英文：

```text
We replace DROID-W's learned optical update frontend with a D4RT-based 4D correspondence frontend. The predicted 4D tracks are converted into reprojection factors and optimized by the original SLAM bundle adjustment backend.
```

中文：

```text
我们用 D4RT 的 4D 时空对应关系替代 DROID-W 的学习式前端更新，将 D4RT 预测直接转化为 SLAM 重投影因子，并由 BA 后端联合优化相机位姿和深度。
```

方法名称候选：

```text
D4RT-SLAM
D4RT-Driven DROID-W
D4RT Reprojection-Factor SLAM
D4RT Frontend for Dynamic SLAM
```

## 13. 核心结论

该方案中：

```text
D4RT 不是辅助权重
D4RT 不是不确定性估计
D4RT 不是后处理
D4RT 不是简单拼接
```

而是：

```text
D4RT 直接成为 SLAM 观测方程的来源。
```

这使得 D4RT 的 4D prediction 能够真实影响 BA 中的 pose / depth 优化过程，从而有机会降低 ATE 并改善动态场景重建。
