# Dynamic 4D SLAM Integrated Report

本文档整合以下三份记录，并补充当前学习模型方法原理与实验设计：

- `Notebook_AI.md`：早期动态预测系统设计与 Kalman/D4RT 思路。
- `DYNAMIC_4D_PROGRESS.md`：当前 4D 动态建图实现、数值结果与 learned-region 方法进展。
- `CROWD_TEST_COMMANDS.md`：Bonn crowd 与 Bonn 全序列复现实验命令。

## 1. 研究目标

传统 SLAM 默认世界静止：

```text
世界点变化 = 相机运动造成的投影变化
```

动态物体出现后，传统系统通常把动态区域当成 outlier：

```text
动态点 -> residual 大 -> 降权 / mask / 丢弃
```

本项目的目标不是简单丢弃动态物体，而是构建动态物体的 4D 世界模型：

```text
静态部分：稳定全局 3D 地图
动态部分：随时间变化的 3D 点云 + 下一帧运动预测
```

最终希望证明：

```text
学习模型预测的动态 motion
  优于无预测 baseline
  优于点级 Kalman / previous motion
  能让动态物体 4D 建图更准确
```

## 2. 范式转变

### 2.1 从 Outlier Rejection 到 Dynamic Geometry Modeling

传统 BA 对静态点的假设是：

```text
X_j = T_j T_i^{-1} X_i
```

动态点不满足这个假设：

```text
X_j != T_j T_i^{-1} X_i
```

传统动态 SLAM 的处理方式是：

```text
检测动态区域 -> mask 掉 -> 只用静态区域优化
```

DROID-W 的处理更柔和：

```text
多视角 feature 不一致 -> uncertainty 变大 -> BA 权重下降
```

但本质上，动态点仍然没有被建模。本项目的核心变化是：

```text
动态点不是错误，而是带有独立运动的有效几何信息。
```

### 2.2 Dynamic Motion Field

动态世界中，一个点从帧 `t` 到 `t+1` 的变化既包含相机运动，也包含物体自身运动。

因此动态点需要额外的 motion field：

```text
X_{t+1} = X_t + ΔX_t
```

这里的 `ΔX_t` 是动态物体在 3D 世界中的运动向量。

| 概念 | 空间 | 含义 |
|---|---|---|
| Optical Flow | 2D image | 像素位置变化 |
| Scene Flow / 3D Motion | 3D world | 世界坐标中的点运动 |
| 4D Model | 3D + time | 每一帧的动态点位置与预测位置 |

## 3. 当前整体流程

当前保留的主流程如下：

```text
SLAM run.py
  -> Outputs/Bonn/<scene>/video.npz

Previous/Kalman 4D export
  -> 4d_model_previous_motion/model_4d.npz/.mat

Learned pixel motion
  -> video_learned_motion.npz

Region-level smoothing
  -> video_learned_motion_region.npz

Learned-region 4D export
  -> 4d_model_learned_motion_region/model_4d.npz/.mat

Numeric comparison
  -> experiment_baseline_kalman_learned.csv/.json/.png/.md

MATLAB comparison package
  -> matlab_4d_comparison/
```

关键文件含义：

| 文件 | 含义 |
|---|---|
| `video.npz` | 原始 SLAM 输出，包含 RGB、pose、disparity、uncertainty、dynamic motion/mask |
| `video_learned_motion.npz` | 学习模型预测 pixel flow 后写入 3D dynamic motion 的视频包 |
| `video_learned_motion_region.npz` | 对 learned motion 做区域级平滑后的最终 motion 视频包 |
| `4d_model_previous_motion/model_4d.npz` | 从原始 `video.npz` 导出的 previous/Kalman 4D 模型 |
| `4d_model_learned_motion_region/model_4d.npz` | 从 learned-region 视频包导出的当前最佳 4D 模型 |
| `matlab_4d_comparison/*/model_4d.mat` | MATLAB 可视化使用的三种模型 |

## 4. 点级 Kalman / Previous Motion 方法

### 4.1 方法

早期方法是 point-level dynamic prediction：

1. 在关键帧中根据 uncertainty 和 depth 选取动态候选点。
2. 对相邻关键帧动态点做最近邻 ID 匹配。
3. 对每个点维护滑动窗口轨迹。
4. 使用 Kalman Filter 或匀速假设预测下一帧位置。
5. 将预测结果反馈到：
   - `uncertainties`：改变 BA 权重；
   - `dynamic_motions / dynamic_motion_priors / dynamic_motion_masks`：写入动态 motion prior。
6. 导出 4D 模型时，将 `motion_world` 和 `predicted_next_points` 写入 `model_4d.npz/.mat`。

Kalman 状态可写作：

```text
state = [x, y, z, vx, vy, vz]
```

### 4.2 已完成修复

- 在线预测不再只离线可视化，而是在 frontend BA 前反馈。
- 同一 `(source_kf, target_kf)` 只应用一次，避免重复抬高 uncertainty。
- 动态候选点数量加入 `min_dynamic_points / max_dynamic_points` 控制。
- ID 匹配改为一对一贪心匹配，避免多个点共享同一个 ID。
- feedback mask 覆盖率过高时自动缩小 splat 半径或采样。
- 增加诊断日志：
  - `dynamic_id_match_stats.csv`
  - `dynamic_feedback_stats.csv`
  - `dynamic_motion_feedback_stats.csv`
  - `dynamic_ba_edge_stats.csv`

### 4.3 结论

点级 Kalman 链路已经接入，但效果有限。主要原因：

- RGB-D 点云中的单个表面点跨帧不稳定；
- 最近邻 ID 不等于真实物理对应；
- 深度噪声会被 Kalman 当成运动；
- 独立点预测破坏动态物体整体一致性；
- 点级 NN 评估受遮挡、表面采样密度和深度厚度影响明显。

因此当前结论是：

```text
点级 Kalman 可以证明“预测能进系统”，
但不是动态物体 4D 建图的最佳运动模型。
```

## 5. 学习式像素运动预测方法

### 5.1 核心思想

学习模型不直接预测稀疏 3D 点轨迹，而是在 DROID 低分辨率网格上预测动态像素下一帧的位置：

```text
input(t) -> flow(t -> t+1)
```

再利用深度、相机位姿和 intrinsics，把 2D flow 转成 3D motion：

```text
pixel_t + predicted_flow -> pixel_{t+1}
depth_{t+1}(pixel_{t+1}) -> X_{t+1}
X_{t+1} - X_t -> dynamic_motions[t]
```

这样做的原因：

- 像素运动监督比 3D 点 ID 更稳定；
- 低分辨率网格与 DROID 内部 BA 表示一致；
- 学习模型能利用 RGB、depth、uncertainty 和动态 mask 的局部上下文；
- 后续 region smoothing 能恢复动态物体的整体一致性。

### 5.2 理论建模

对于第 `t` 帧低分辨率像素 `p = (u, v)`，由 disparity 和相机内参可以反投影得到相机坐标点：

```text
z_t(p) = 1 / disp_t(p)
X_t^cam(p) = z_t(p) * K_t^{-1} [u, v, 1]^T
```

再由当前相机位姿 `T_t` 转成世界坐标：

```text
X_t^world(p) = T_t X_t^cam(p)
```

学习模型预测的是低分辨率像素位移：

```text
f_theta(I_t, disp_t, uncertainty_t, mask_t) = flow_t(p) = (du, dv)
```

预测下一帧像素位置：

```text
p' = p + flow_t(p)
```

然后在下一帧深度图中取 `p'` 的深度，并用下一帧位姿得到下一帧世界坐标：

```text
X_{t+1}^world(p') = T_{t+1} X_{t+1}^cam(p')
```

最终写回 SLAM/4D 模型的 3D 动态运动是：

```text
ΔX_t(p) = X_{t+1}^world(p') - X_t^world(p)
```

导出的 4D 预测点为：

```text
predicted_next_points = points + ΔX_t
```

这个设计把问题拆成两个更稳定的子问题：

```text
学习模型负责：预测 2D 动态像素运动
几何模块负责：利用深度和位姿转成 3D motion
```

相比直接预测 3D motion，这样可以减少网络需要学习的几何负担，也更容易使用 DROID-W 已经稳定输出的 depth、pose 和 intrinsics。

### 5.3 为什么不是直接学习 3D 点轨迹

点级 3D 轨迹学习要求跨帧点 ID 稳定，但 RGB-D 重建中的点并不是物理点级永久追踪：

- 相机视角变化会改变表面采样；
- 遮挡和反遮挡会让点出现或消失；
- 深度噪声会造成 3D 坐标抖动；
- 最近邻匹配容易把不同物理点误认为同一个点。

因此直接学习：

```text
[X_t, X_{t-1}, ...] -> X_{t+1}
```

会把几何噪声、重采样噪声和真实运动混在一起。

当前方法改为学习：

```text
[RGB, disp, uncertainty, dynamic mask] -> pixel flow
```

优势是：

- 像素空间连续性更强；
- flow 标签更密集；
- 局部纹理、边界、uncertainty 能辅助判断运动方向；
- 转成 3D motion 时仍保持几何一致性。

### 5.4 与 Kalman / Previous Motion 的本质区别

| 维度 | 点级 Kalman / Previous | Learned Pixel Motion |
|---|---|---|
| 预测对象 | 单个 3D 点 ID | 低分辨率动态像素 |
| 输入信息 | 历史 3D 坐标 | RGB + depth + uncertainty + dynamic mask |
| 对应关系 | 最近邻 ID 匹配 | 相邻帧 3D NN 生成 flow 标签 |
| 运动假设 | 近似匀速 / 线性 | 数据驱动，能学习局部运动模式 |
| 物体一致性 | 弱，每点独立 | 后接 region smoothing |
| 主要风险 | 点 ID 不稳定、噪声大 | 标签噪声、泛化能力不足 |

Kalman 方法的核心假设是：

```text
同一个 ID 的点在短时间内做近似匀速运动。
```

学习模型的核心假设是：

```text
动态区域的外观、深度、不确定性和 mask 可以预测下一帧像素位移。
```

这也是为什么学习模型更适合当前目标：我们不是只想平滑一个点，而是希望动态物体区域整体运动更接近下一帧观测。

### 5.5 输入与输出

训练样本来自 `scripts_train/build_pixel_motion_dataset.py`。

每个样本：

| 字段 | 形状 | 含义 |
|---|---|---|
| `inputs` | `[6,H,W]` | RGB 3 通道 + disparity + uncertainty + dynamic mask |
| `flows` | `[2,H,W]` | 监督标签 `(du,dv)` |
| `valids` | `[1,H,W]` | 有效监督 mask |

动态 mask 的来源：

```text
uncertainty > threshold
OR dynamic_motion_masks > 0
```

如果动态点太少，会自动选 uncertainty 最高的像素补足到 `min_dynamic_points`。

### 5.6 标签生成

对相邻帧 `t -> t+1`：

1. 根据 disparity + intrinsics + pose 反投影出当前帧动态点。
2. 同样反投影下一帧动态点。
3. 使用 3D nearest neighbor 找到当前动态点在下一帧的近邻。
4. 如果距离小于 `max_nn_dist`，把像素位移作为 flow 标签：

```text
flow_x = x_{t+1} - x_t
flow_y = y_{t+1} - y_t
```

这不是完美真值，但比纯点级 Kalman 的在线 ID 更适合作为监督信号。

标签生成的约束包括：

| 参数 | 作用 |
|---|---|
| `uncer-thresh` | 选择动态候选区域 |
| `min-dynamic-points` | 防止动态监督太稀疏 |
| `max-dynamic-points` | 防止大面积不确定区域污染标签 |
| `max-depth` | 过滤过远、深度不可靠点 |
| `max-nn-dist` | 只接受合理的 3D 最近邻对应 |
| `min-labels` | 丢弃监督点太少的帧对 |

这套标签不是语义级真值，而是从 SLAM 输出中自动挖出的 weak supervision。它的目标不是训练通用 optical flow，而是训练一个服务于当前 DROID-W 低分辨率动态建图的 motion prior。

### 5.7 网络结构

当前模型是轻量 Small U-Net：

```text
ConvBlock -> Downsample -> ConvBlock -> Downsample
          -> Bottleneck
          -> Upsample + Skip
          -> Upsample + Skip
          -> 1x1 Conv -> 2D flow
```

实现文件：

```text
dynamic_prediction/pixel_motion_model.py
```

核心类：

```python
class SmallPixelMotionUNet(nn.Module):
```

模型输入输出：

```text
input : [B, 6, H, W]
        RGB 3 channels
        disparity 1 channel
        uncertainty 1 channel
        dynamic mask 1 channel

output: [B, 2, H, W]
        du, dv pixel motion
```

网络由以下模块组成：

```text
enc1 -> down1 -> enc2 -> down2 -> bottleneck
     -> up2 + skip enc2 -> dec2
     -> up1 + skip enc1 -> dec1
     -> 1x1 conv head -> 2D flow
```

其中：

| 模块 | 作用 |
|---|---|
| `ConvBlock` | 两层 `Conv2d + BatchNorm + ReLU`，提取局部特征 |
| `down1 / down2` | `MaxPool2d` 下采样，扩大感受野 |
| `bottleneck` | 聚合更大范围的运动上下文 |
| `up2 / up1` | `ConvTranspose2d` 上采样，恢复空间分辨率 |
| skip connection | 保留像素级定位信息 |
| `head` | `1x1 Conv2d`，输出 `(du, dv)` |

模型在训练脚本中实例化：

```text
scripts_train/train_pixel_motion.py
```

对应代码逻辑：

```python
model = SmallPixelMotionUNet(
    in_channels=inputs.shape[1],
    base_channels=args.base_channels,
    out_channels=flows.shape[1],
).to(device)
```

模型在推理/写回 `video.npz` 时再次实例化：

```text
scripts_train/apply_pixel_motion_to_video.py
```

对应代码逻辑：

```python
model = SmallPixelMotionUNet(
    in_channels=int(ckpt.get("in_channels", 6)),
    base_channels=int(ckpt.get("base_channels", 32)),
    out_channels=int(ckpt.get("out_channels", 2)),
).to(device)
```

checkpoint 中会保存：

| 字段 | 含义 |
|---|---|
| `model` | 网络权重 |
| `in_channels` | 输入通道数，当前为 6 |
| `base_channels` | U-Net 基础通道数，默认 32 |
| `out_channels` | 输出通道数，当前为 2 |
| `epoch` | 最优 checkpoint 对应 epoch |
| `val_epe` | 验证集 EPE |
| `dataset` | 训练数据路径 |

损失函数：

```text
masked Smooth L1(pred_flow, target_flow)
```

只在 `valids == 1` 的像素上监督。

选择 Small U-Net 的原因：

- 输入输出分辨率较低，模型不需要很大；
- encoder 可以聚合局部上下文；
- skip connection 保留像素级定位信息；
- 输出 dense flow，方便直接写回 `dynamic_motions`。

损失目标可以写成：

```text
L(theta) = mean_{p in valid} SmoothL1(f_theta(x_t)(p), flow_label_t(p))
```

评估时使用 endpoint error：

```text
EPE = || flow_pred - flow_label ||_2
```

并与 zero-flow baseline 对比：

```text
zero-flow: flow_pred = 0
```

如果 learned EPE 明显低于 zero-flow EPE，说明模型至少学到了动态像素位移的非零规律。

### 5.8 从 Flow 到 4D Motion

推理脚本：

```text
scripts_train/apply_pixel_motion_to_video.py
```

流程：

1. 读取 `video.npz`。
2. 构建 `[RGB, disp, uncertainty, dynamic_mask]` 输入。
3. U-Net 输出低分辨率 flow。
4. 使用 `depth-mode next`：
   - 当前像素反投影到 `X_t`；
   - 预测像素在下一帧取深度，反投影到 `X_{t+1}`；
   - 计算 `X_{t+1} - X_t`。
5. 写回：
   - `dynamic_motion_flow_learned`
   - `dynamic_motions`
   - `dynamic_motion_priors`
   - `dynamic_motion_masks`

推理阶段的重要过滤：

| 参数 | 作用 |
|---|---|
| `max-flow-px` | 限制异常大的像素位移 |
| `max-motion-m` | 限制异常大的 3D motion |
| `depth-mode next` | 使用下一帧预测像素处深度来恢复真实 3D 位移 |
| `min/max-dynamic-points` | 控制每帧动态区域大小 |

`depth-mode next` 很关键。若只用当前帧深度并假设深度不变，模型只能产生近似平面内运动；使用下一帧深度后，才能表达朝向相机或远离相机的 3D 运动。

### 5.9 Region-Level Motion Smoothing

单点 learned motion 仍可能有噪声，因此进一步做区域平滑：

```text
scripts_train/smooth_video_motion_regions.py
```

每帧处理：

1. 对 `dynamic_motion_masks` 做连通区域。
2. 丢弃小区域。
3. 每个区域内对 3D motion 做 trimmed median。
4. 将点级 motion 与区域 motion 按 `blend` 混合。

作用：

```text
把像素级预测变成物体/区域级一致运动。
```

当前最佳方法就是：

```text
Learned pixel flow + next-depth 3D motion + region-level smoothing
```

从理论上看，region smoothing 引入了一个弱刚体/局部刚体先验：

```text
同一个连通动态区域内的点，短时间内应具有相近的 3D motion。
```

这不是强制整个物体完全刚体，而是用 trimmed median 抑制局部错误预测：

```text
motion_region = median_trimmed({ΔX_p | p in component})
ΔX_p_new = (1 - blend) * ΔX_p + blend * motion_region
```

这样可以保留局部差异，同时让整体运动更稳定。

## 6. 实验设计

### 6.1 实验假设

本实验验证的不是“相机轨迹 RMSE 一定大幅提升”，而是更具体的动态建图假设：

```text
如果动态 motion 预测有效，
那么当前帧动态点经过 motion 补偿后，
应该比不补偿时更接近下一帧实际观测到的动态点。
```

也就是：

```text
distance(points_t + predicted_motion_t, dynamic_points_{t+1})
<
distance(points_t, dynamic_points_{t+1})
```

这正对应 `motion_nn_m_mean` 和 `static_nn_m_mean` 的比较。

实验分三层证明：

| 层级 | 要证明什么 | 对应指标 |
|---|---|---|
| Pixel motion | 学习模型能预测动态像素位移 | learned EPE vs zero-flow EPE |
| 3D dynamic motion | 预测 motion 能让点靠近下一帧动态观测 | motion NN vs static NN |
| 方法对比 | learned-region 优于 previous/Kalman | LearnedRegion NN vs KalmanPrevious NN |

### 6.2 控制变量

为了让比较公平，三种方法应尽量使用同一套 SLAM 结果：

```text
同一个 video.npz
同一组 pose / disparity / intrinsics
同一套 export_4d_model.py
同一个 max_nn_dist
同一个评价脚本 compare_4d_methods.py
```

唯一改变的是 dynamic motion 的来源：

| 方法 | dynamic motion 来源 |
|---|---|
| Baseline Static | motion 全部为 0 |
| KalmanPrevious | 原始 online/previous motion |
| LearnedRegion | 学习模型预测 flow 转 3D motion，再做区域平滑 |

这样可以把效果差异主要归因于：

```text
动态 motion 预测质量
```

而不是相机轨迹、深度源或可视化脚本差异。

### 6.3 评价对象

评价重点是动态物体部分，而不是整幅静态场景：

```text
static map 已经较好；
本实验关注 dynamic points 的 temporal consistency。
```

因此评估时只选：

```text
dynamic == True
motion_valid == True
```

的点作为预测查询点，并与下一帧动态点做最近邻匹配。

这个指标是近似评估，不是真实物体级 GT，但它适合当前自动化实验：

- 不需要人工语义标注；
- 可以批量跑所有 Bonn dynamic 序列；
- 能直接比较 motion 补偿前后是否更接近下一帧。

### 6.4 成功标准

一个序列上认为 learned-region 有效，需要同时满足：

```text
motion_nn_m_mean < static_nn_m_mean
LearnedRegion motion_nn_m_mean < KalmanPrevious motion_nn_m_mean
axis signed mean 没有明显单轴系统偏差
```

更理想的结果是：

```text
motion_vs_static_gain_percent > 0
method_to_method_gain_percent > 0
```

如果 pixel EPE 提升明显，但 4D NN 不提升，说明问题可能出在：

- flow 转 3D motion 的深度不稳定；
- dynamic mask 选点不合适；
- region smoothing 参数不合适；
- 3D NN 评估受到遮挡或表面采样影响；
- 当前序列的 learned model 泛化不足。

### 6.5 对比方法

| 方法 | 输入 | 含义 |
|---|---|---|
| Baseline Static | `video.npz` | 不预测动态点，动态点保持当前位置 |
| KalmanPrevious | `4d_model_previous_motion/model_4d.npz` | previous / point-level Kalman motion |
| LearnedRegion | `4d_model_learned_motion_region/model_4d.npz` | 学习模型 flow + 3D motion + 区域平滑 |

### 6.6 主要指标

核心指标是动态点下一帧 3D 最近邻误差：

```text
predicted dynamic point at frame t+1
vs
observed dynamic point in frame t+1
```

单位是米，越低越好。

在 `evaluate_dynamic_method.py` 中：

| 指标 | 含义 |
|---|---|
| `static_nn_m_mean` | 不使用 motion，当前动态点直接和下一帧动态点匹配的误差 |
| `motion_nn_m_mean` | 使用预测 motion 后和下一帧动态点匹配的误差 |
| `motion_vs_static_gain_percent` | motion 相对 static carry-forward 的提升 |
| `model_motion_valid_points` | 有有效 motion 的动态点数量 |
| `axis_*_signed_mean_m` | 三轴有符号偏差，用于检查坐标轴/尺度问题 |

在 `compare_4d_methods.py` 中：

```text
KalmanPrevious motion NN
LearnedRegion motion NN
Method-to-method gain
LearnedRegion vs static baseline
```

### 6.7 Pixel Flow 指标

学习模型本身还用 pixel endpoint error 评估：

| 指标 | 含义 |
|---|---|
| `learned EPE` | 模型 flow 与标签 flow 的平均 endpoint error |
| `zero-flow EPE` | 不预测运动、flow=0 的 baseline |
| `relative gain` | learned 相对 zero-flow 的提升 |

当前已观察到：

```text
learned EPE: 4.290569 px
zero-flow EPE: 5.737912 px
relative gain: +25.22%
```

### 6.8 当前 Bonn Balloon 结果

点级 Kalman / previous motion 在当前评估中没有稳定超过 static carry-forward：

```text
motion_nn_m_mean: 0.089454 m
static_nn_m_mean: 0.087068 m
motion_vs_static_gain_percent: -2.74%
```

learned next-depth 版本绝对误差更低，但未稳定转正：

```text
NN distance mean: 0.077701 m
motion_nn_m_mean: 0.079990 m
static_nn_m_mean: 0.078459 m
motion_vs_static_gain_percent: -1.95%
```

当前最佳 learned-region 结果：

```text
video learned region active pixels: 13035
active ratio: 7.5771%
components mean: 2.02
motion mean: 0.100019 m

4D learned region NN distance mean: 0.065516 m
NN distance median: 0.038219 m
NN distance p90: 0.159599 m

motion_nn_m_mean: 0.066065 m
static_nn_m_mean: 0.076519 m
motion_vs_static_gain_percent: +13.66%
```

Kalman vs learned-region：

```text
KalmanPrevious motion NN: 0.089766 m
LearnedRegion motion NN: 0.066065 m
Method-to-method gain: +26.40%
```

结论：

```text
学习模型在 pixel flow 上优于 zero-flow；
learned-region 4D motion 明显优于 previous/Kalman；
区域级一致性是把 learned motion 转化为 4D 建图收益的关键。
```

## 7. 复现实验命令

### 7.1 环境

```bash
cd /home/youran/文档/DROID-W/TUM-Cambridge_onSLAM
conda activate droid-w
```

### 7.2 单序列：Bonn Crowd

运行 SLAM：

```bash
python run.py --config configs/Dynamic/Bonn/bonn_crowd.yaml
```

导出 previous/Kalman 4D：

```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_crowd/video.npz \
  --output-dir Outputs/Bonn/bonn_crowd/4d_model_previous_motion \
  --disp-source up \
  --no-ply \
  --no-tracks
```

应用 learned pixel motion：

```bash
python scripts_train/apply_pixel_motion_to_video.py \
  --video-npz Outputs/Bonn/bonn_crowd/video.npz \
  --checkpoint checkpoints/pixel_motion_bonn_dynamic/best.pt \
  --out-video Outputs/Bonn/bonn_crowd/video_learned_motion.npz \
  --device cpu
```

区域平滑：

```bash
python scripts_train/smooth_video_motion_regions.py \
  --video-npz Outputs/Bonn/bonn_crowd/video_learned_motion.npz \
  --out-video Outputs/Bonn/bonn_crowd/video_learned_motion_region.npz
```

导出 learned-region 4D：

```bash
python scripts_eval/export_4d_model.py \
  --video-npz Outputs/Bonn/bonn_crowd/video_learned_motion_region.npz \
  --output-dir Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region \
  --disp-source up \
  --no-ply \
  --no-tracks
```

比较 previous/Kalman 与 learned-region：

```bash
python scripts_eval/compare_4d_methods.py \
  --old-model Outputs/Bonn/bonn_crowd/4d_model_previous_motion/model_4d.npz \
  --new-model Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/model_4d.npz \
  --old-name KalmanPrevious \
  --new-name LearnedRegion \
  --out-prefix Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/experiment_baseline_kalman_learned
```

打包 MATLAB 对比：

```bash
python scripts_eval/package_matlab_4d_comparison.py \
  --kalman-model Outputs/Bonn/bonn_crowd/4d_model_previous_motion/model_4d.npz \
  --learned-model Outputs/Bonn/bonn_crowd/4d_model_learned_motion_region/model_4d.npz \
  --output-dir Outputs/Bonn/bonn_crowd/matlab_4d_comparison
```

MATLAB 查看：

```matlab
addpath('scripts_eval');
view_4d_comparison_matlab('Outputs/Bonn/bonn_crowd/matlab_4d_comparison', 8, 'none', true);
```

### 7.3 Bonn 全序列批量命令

全序列列表：

```bash
SCENES="bonn_balloon bonn_balloon2 bonn_crowd bonn_crowd2 bonn_moving_nonobstructing_box bonn_moving_nonobstructing_box2 bonn_person_tracking bonn_person_tracking2"
```

批量运行 SLAM：

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

批量导出 previous/Kalman 4D：

```bash
for scene in $SCENES
do
  python scripts_eval/export_4d_model.py \
    --video-npz "Outputs/Bonn/${scene}/video.npz" \
    --output-dir "Outputs/Bonn/${scene}/4d_model_previous_motion" \
    --disp-source up \
    --no-ply \
    --no-tracks
done
```

批量应用 learned pixel motion：

```bash
for scene in $SCENES
do
  python scripts_train/apply_pixel_motion_to_video.py \
    --video-npz "Outputs/Bonn/${scene}/video.npz" \
    --checkpoint checkpoints/pixel_motion_bonn_dynamic/best.pt \
    --out-video "Outputs/Bonn/${scene}/video_learned_motion.npz" \
    --device cpu
done
```

批量区域平滑：

```bash
for scene in $SCENES
do
  python scripts_train/smooth_video_motion_regions.py \
    --video-npz "Outputs/Bonn/${scene}/video_learned_motion.npz" \
    --out-video "Outputs/Bonn/${scene}/video_learned_motion_region.npz"
done
```

批量导出 learned-region 4D：

```bash
for scene in $SCENES
do
  python scripts_eval/export_4d_model.py \
    --video-npz "Outputs/Bonn/${scene}/video_learned_motion_region.npz" \
    --output-dir "Outputs/Bonn/${scene}/4d_model_learned_motion_region" \
    --disp-source up \
    --no-ply \
    --no-tracks
done
```

批量比较：

```bash
for scene in $SCENES
do
  python scripts_eval/compare_4d_methods.py \
    --old-model "Outputs/Bonn/${scene}/4d_model_previous_motion/model_4d.npz" \
    --new-model "Outputs/Bonn/${scene}/4d_model_learned_motion_region/model_4d.npz" \
    --old-name KalmanPrevious \
    --new-name LearnedRegion \
    --out-prefix "Outputs/Bonn/${scene}/4d_model_learned_motion_region/experiment_baseline_kalman_learned"
done
```

批量打包 MATLAB：

```bash
for scene in $SCENES
do
  python scripts_eval/package_matlab_4d_comparison.py \
    --kalman-model "Outputs/Bonn/${scene}/4d_model_previous_motion/model_4d.npz" \
    --learned-model "Outputs/Bonn/${scene}/4d_model_learned_motion_region/model_4d.npz" \
    --output-dir "Outputs/Bonn/${scene}/matlab_4d_comparison"
done
```

## 8. 训练学习模型

### 8.1 构建 Bonn 训练集

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
  --min-labels 10 \
  --skip-missing
```

### 8.2 训练

```bash
python scripts_train/train_pixel_motion.py \
  --dataset datasets_train/bonn_dynamic_pixel_motion/samples.npz \
  --out checkpoints/pixel_motion_bonn_dynamic \
  --epochs 100 \
  --batch-size 8 \
  --base-channels 32 \
  --lr 1e-3
```

### 8.3 评估 pixel flow

```bash
python scripts_train/evaluate_pixel_motion.py \
  --dataset datasets_train/bonn_dynamic_pixel_motion/samples.npz \
  --checkpoint checkpoints/pixel_motion_bonn_dynamic/best.pt \
  --batch-size 16 \
  --device cpu \
  --out checkpoints/pixel_motion_bonn_dynamic/eval.json
```

## 9. 可视化设计与 MATLAB 展示

### 9.1 可视化目标

可视化不是单纯显示点云，而是要直观看到三件事：

```text
1. 静态环境是否稳定；
2. 动态物体当前帧位置是否正确；
3. 预测点是否更接近下一帧真实动态点。
```

因此当前可视化采用 4D 动画方式：

```text
3D 空间坐标 + 时间帧 t
```

每一帧显示：

- 当前帧重建点云；
- 当前帧动态点；
- 当前帧动态点预测到下一帧的位置；
- 下一帧实际观测到的动态点，用于视觉对比预测是否准确。

### 9.2 `model_4d.mat` 中的关键变量

MATLAB 可视化主要读取 `model_4d.mat`：

| 变量 | 含义 |
|---|---|
| `points` | 所有帧拼接后的 3D 世界点 |
| `colors` | 每个点的 RGB |
| `dynamic` | 是否为动态点 |
| `frame_start_1 / frame_end_1` | 每一帧在 `points` 中的索引范围 |
| `motion_world` | 当前点的 3D 运动向量 |
| `predicted_next_points` | 当前点预测到下一帧的位置 |
| `motion_valid` | 该点是否有有效 motion |
| `static_map_points` | 多帧融合后的静态地图点，可选 |

核心关系：

```text
predicted_next_points = points + motion_world
```

baseline static 中：

```text
motion_world = 0
predicted_next_points = points
```

因此 baseline 可视化时，青色预测点会和当前动态点重合。

### 9.3 颜色约定

| 颜色 | 含义 |
|---|---|
| 原始 RGB | 当前帧重建点 |
| 红色 | 当前帧动态点 |
| 青色 | 当前帧动态点预测到下一帧的位置 |
| 绿色 | 下一帧实际观测动态点 |
| 浅色线段 | 从当前动态点指向预测点的 motion vector |

看图时重点观察：

```text
青色点越靠近绿色点，说明预测越准确。
```

### 9.4 单模型查看

使用：

```text
scripts_eval/view_4d_model_matlab.m
```

示例：

```matlab
addpath('scripts_eval');
view_4d_model_matlab('Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region/model_4d.mat', 8, true, 'none', true);
```

参数含义：

| 参数 | 含义 |
|---|---|
| 第 1 个参数 | `model_4d.mat` 路径 |
| 第 2 个参数 | 播放 fps |
| 第 3 个参数 | 是否将动态点强制显示为红色 |
| 第 4 个参数 | 坐标轴模式，如 `'none'`, `'swap_yz'`, `'flip_z'` |
| 第 5 个参数 | 是否显示预测点和 motion vector |

如果发现 MATLAB 中某个轴方向看起来不对，可以只改可视化坐标，不改原始数据：

```matlab
view_4d_model_matlab('路径/model_4d.mat', 8, true, 'swap_yz', true);
view_4d_model_matlab('路径/model_4d.mat', 8, true, 'flip_z', true);
```

### 9.5 三方法同步对比

使用：

```text
scripts_eval/view_4d_comparison_matlab.m
```

该脚本用于同时显示：

| 子图 | 方法 | 含义 |
|---|---|---|
| 左 | Baseline Static | 不预测，动态点保持原位置 |
| 中 | KalmanPrevious | previous / 点级 Kalman motion |
| 右 | LearnedRegion | 学习模型 + 区域平滑 motion |

先打包 MATLAB 对比文件：

```bash
python scripts_eval/package_matlab_4d_comparison.py \
  --kalman-model Outputs/Bonn/bonn_balloon/4d_model_previous_motion/model_4d.npz \
  --learned-model Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region/model_4d.npz \
  --output-dir Outputs/Bonn/bonn_balloon/matlab_4d_comparison
```

然后在 MATLAB 中运行：

```matlab
addpath('scripts_eval');
view_4d_comparison_matlab('Outputs/Bonn/bonn_balloon/matlab_4d_comparison', 8, 'none', true);
```

输出目录结构：

```text
matlab_4d_comparison/
├── baseline_static/model_4d.mat
├── kalman_previous/model_4d.mat
├── learned_region/model_4d.mat
├── manifest.json
├── README.md
└── GENERALIZE_TO_OTHER_DATASETS.md
```

### 9.6 三种模型在可视化中的解释

#### Baseline Static

```text
无动态预测。
当前动态点 = 预测点。
```

如果绿色下一帧动态点明显远离青色预测点，说明动态物体确实发生了运动，而 baseline 无法解释这种运动。

#### KalmanPrevious

```text
点级 Kalman / previous motion。
每个动态点独立预测。
```

常见现象：

- 部分点方向正确；
- 但同一个物体内部 motion 不一致；
- 点云可能显得发散或抖动。

这正是点级 Kalman 效果有限的原因。

#### LearnedRegion

```text
学习模型预测像素 flow；
转成 3D motion；
再做动态区域级 smoothing。
```

理想现象：

- 青色预测点整体靠近绿色下一帧动态点；
- 同一动态物体区域运动方向更一致；
- 比 KalmanPrevious 更少发散。

### 9.7 静态环境 + 动态时间线可视化

如果要看“静态环境累计 + 动态物体逐帧运动”，使用：

```text
scripts_eval/build_global_slam_from_4d_matlab.m
scripts_eval/view_static_env_dynamic_timeline_matlab.m
```

先构建全局静态环境：

```matlab
addpath('scripts_eval');

stats = build_global_slam_from_4d_matlab( ...
    'Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region/model_4d.mat', ...
    'Outputs/Bonn/bonn_balloon/slam_global_learned_region', ...
    0.02, 3, 0.20, 'none', true);
```

再播放静态环境和动态时间线：

```matlab
addpath('scripts_eval');

view_static_env_dynamic_timeline_matlab( ...
    'Outputs/Bonn/bonn_balloon/slam_global_learned_region', ...
    2, ...
    4, ...
    'Outputs/Bonn/bonn_balloon/4d_model_learned_motion_region/model_4d.mat', ...
    6);
```

显示逻辑：

| 元素 | 显示方式 |
|---|---|
| 静态点 | 累积显示，形成全局环境 |
| 动态点 | 只显示当前帧，不跨帧累计 |
| 预测点 | 只显示当前帧预测，不跨帧累计 |

这样做是为了避免动态物体在空间中拖出一整条残影，影响对当前预测质量的判断。

### 9.8 MATLAB 可视化建议参数

对于 Bonn balloon：

| 参数 | 推荐值 |
|---|---|
| `axisMode` | `'none'` |
| 单模型 fps | `8` |
| 静态点大小 | `2` |
| 动态点大小 | `4` 到 `8` |
| 预测点大小 | `6` 到 `10` |
| 静态 voxel size | `0.02` |
| 静态 minObs | `3` |
| 静态 maxColorStd | `0.20` |

如果 MATLAB 卡顿：

- 降低 fps；
- 增大静态 voxel size；
- 减小点大小；
- 优先使用三方法同步对比，而不是同时显示完整静态累计地图。

### 9.9 可视化与数值评估的关系

可视化用于回答：

```text
预测点看起来是否朝正确方向移动？
动态物体是否整体一致？
是否存在明显坐标轴偏差？
```

数值评估用于回答：

```text
motion_nn_m_mean 是否降低？
LearnedRegion 是否优于 KalmanPrevious？
LearnedRegion 是否优于 static carry-forward？
```

二者要一起看：

- 如果数值提升但可视化发散，说明评估 mask 或 NN 匹配可能有偏；
- 如果可视化合理但数值不提升，可能是遮挡、深度噪声或动态 mask 选点影响；
- 如果某一轴明显错位，先运行 `diagnose_4d_axis_error.py` 检查轴向偏差。

## 10. 当前推荐结论

当前最值得保留的技术路线是：

```text
DROID-W 静态 SLAM
  + learned pixel motion
  + next-depth 3D motion conversion
  + region/object-level smoothing
  + 4D dynamic model export
```

原因：

- 静态部分已经较好，不应过度改动；
- 点级 Kalman 证明了反馈链路，但预测质量有限；
- 学习模型在 pixel motion 指标上明显优于 zero-flow；
- region-level smoothing 让动态物体 motion 更符合物理一致性；
- 当前 Bonn balloon 上 learned-region 相对 KalmanPrevious 有明显提升。

下一步实验重点：

1. 在 `bonn_crowd / crowd2 / person_tracking` 上验证泛化。
2. 汇总每个序列的 `experiment_baseline_kalman_learned.csv`。
3. 统计平均 `motion_nn_m_mean`、`motion_vs_static_gain_percent` 和 method-to-method gain。
4. 若 crowd 类序列失败，优先检查 CUDA BA 扩展签名、显存和 `video.npz` 是否完整。
5. 在论文/报告中强调：本方法主要提升动态物体 4D 建图，不一定显著提升相机 ATE/RMSE。
