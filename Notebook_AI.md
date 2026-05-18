# DROID-W 动态预测系统设计文档

> 基于 D4RT + Kalman Filter 的逐点运动预测与近实时可视化

---

## 目录

1. [范式转变：从 Outlier Rejection 到 Dynamic Geometry Modeling](#一范式转变)
2. [Dynamic Motion Field 的本质](#二dynamic-motion-field-的本质)
3. [系统核心创新：Motion-Compensated SLAM](#三motion-compensated-slam)
4. [动静分离是否必要](#四动静分离是否必要)
5. [系统目标重新定义](#五系统目标重新定义)
6. [系统总体架构](#六系统总体架构)
7. [详细模块设计](#七详细模块设计)
8. [完整运行流程](#八完整运行流程)
9. [评估指标](#九评估指标)
10. [项目时间线](#十项目时间线)
11. [代码实现](#十一代码实现)

---

## 一、范式转变

### 传统 SLAM 的假设

传统 Bundle Adjustment（BA）的核心假设：**世界是静止的，只有相机在运动。**

$$X_j = T_j T_i^{-1} X_i$$

当动态物体出现时：

$$X_j \neq T_j T_i^{-1} X_i$$

reprojection residual 变大，BA 认为这是一个"错误的点"，于是：

```
feature mismatch → wrong correspondence → pose drift → map corruption
```

### 传统 Dynamic SLAM 的做法（粗暴）

```
detect moving object
    ↓
mask out
    ↓
只优化 static 区域
```

代表方法：DynaSLAM、DS-SLAM、Semantic-SLAM、DROID-W。

本质全部相同：**动态 = 错误 = 丢弃。**

### DROID-W 的进步

DROID-W 不再依赖语义类别判断动态，而是：

> 如果 multi-view feature 不一致 → 提高 uncertainty → 降低 BA 权重

数学表示：

$$\Sigma_{ij} = w_{ij} \cdot \frac{1}{u_i}$$

动态点 $u_i \uparrow$，权重下降。

**但本质问题没有解决：动态点依然被边缘化，不是被理解。**

### 真正的范式转变

| | 传统 SLAM | DROID-W | 新范式 |
|---|---|---|---|
| 动态点的处理 | 丢弃 | 降权 | 理解运动 |
| 动态点的语义 | 错误 | 不确定 | 有效的运动几何 |
| 信息利用 | 丢失 | 部分丢失 | 完整保留 |

> **从 "Outlier Rejection" 到 "Dynamic Geometry Modeling" 的范式转变。**

---

## 二、Dynamic Motion Field 的本质

### 传统运动模型

只有相机运动：

$$X_j = T_j T_i^{-1} X_i$$

### 动态世界的运动模型

相机运动 + 物体运动：

$$X_j = M_{i \to j}(X_i)$$

$M_{i \to j}$ 即 **Dynamic Motion Field**，表示空间中每个点如何随时间运动。

### 直观理解

以人走路为例：

| 身体部位 | 运动 |
|---|---|
| 左肩膀 | 向右移动 5cm |
| 手 | 向上挥动 20cm |
| 腿 | 向前运动 |

不同 3D 点有不同运动 → Motion Field 本质是一个 **3D vector field**。

### Scene Flow vs Optical Flow

| | Optical Flow | Scene Flow |
|---|---|---|
| 空间 | 2D 图像运动 | 3D 世界运动 |
| 示例（人朝相机走） | 图像向外扩散 | 沿 Z 轴前进 |
| 理解层次 | 像素级 | 物理世界级 |

### DPM / D4RT 的作用

DPM / D4RT 第一次真正提供 **dense 3D temporal correspondence**：

$$\Delta X = P_1(t_2, \pi_1) - P_1(t_1, \pi_1)$$

这就是真实的 3D motion vector。

| | DROID-W 知道 | DPM 知道 |
|---|---|---|
| 信息 | feature 不像 | 这是同一个物理点在 $t_1$ 和 $t_2$ 的位置 |

---

## 三、Motion-Compensated SLAM

### 核心创新

> **不是 Dynamic-aware SLAM，而是 Motion-Compensated SLAM。**

对于 static 点：

$$X_j = T_j T_i^{-1} X_i$$

对于 dynamic 点：

$$X_j = T_j \cdot M_{i \to j}(X_i)$$

### 世界模型的改变

| | 传统 | 新范式 |
|---|---|---|
| 相机 | rigid | rigid |
| 世界 | rigid | **non-rigid** |

### Motion-Compensated BA

**原始 BA residual：**

$$r_{ij} = p_{ij}^{obs} - \Pi(T_j T_i^{-1} X_i)$$

**Dynamic-aware residual：**

$$E = \underbrace{\sum_{k \in \mathcal{S}} \left\| \pi(T_j T_i^{-1} X_k) - u_{jk} \right\|^2}_{\text{Static term}} + \underbrace{\sum_{k \in \mathcal{D}} \left\| \pi(T_j \cdot M_{i\to j}(X_k)) - u_{jk} \right\|^2}_{\text{Dynamic term}}$$

**关键变化：**

```
以前：residual 大 → remove
现在：residual 大 → estimate motion
```

---

## 四、动静分离是否必要

### 结论：不需要硬分离，只需软区分

传统方法必须硬分离，因为模型里没有位置给"运动"，是一个二值决策。

在新框架中：

- static 点：$M_{i\to j}(X_k) = X_k$，位移为零，退化为传统 BA
- dynamic 点：$M_{i\to j}(X_k) = X_k + \Delta X$，补偿运动后参与优化

**所有点都进入同一个代价函数，区别只是 Motion Field 给它的 $\Delta X$ 是不是零。**

### 为什么仍需软区分

1. **相机位姿估计**：主要由 static 点约束，避免 Motion Field 误差污染 $T_j$
2. **Motion Field 监督信号**：static 点约束 $\Delta X = 0$，dynamic 点提供真实位移

### 本质

```
每个点 X_k 都有一个连续的 Motion Vector ΔX_k

ΔX_k ≈ 0  →  static 点
ΔX_k ≠ 0  →  dynamic 点

分离是结果，不是前提。
```

---

## 五、系统目标重新定义

### 从定位+建图 → 定位+建图+行为预测

```
传统 SLAM：   我在哪里？地图长什么样？
Dynamic SLAM：我在哪里？哪些点是动态的？
新系统：      我在哪里？动态物体现在在哪？它下一步去哪？
```

### 系统定位

> **一个以 4D 重建为后端的 SLAM 系统，它不仅知道世界现在的样子，还知道动态物体过去怎么运动，并能预测它们下一步去哪里。**

这是一个 **Spatial-Temporal World Model**。

### 设计决策

| 决策点 | 选择 |
|---|---|
| 4D Reconstruction | D4RT |
| 预测粒度 | 逐点预测 |
| 预测结果用途 | 可视化 |
| 预测帧数 | 未来 1-2 帧 |
| 实时性 | 近实时（滑动窗口） |

---

## 六、系统总体架构

```
输入视频序列
      ↓
┌─────────────────────────────────────────────┐
│           DROID-W Frontend                   │
│  特征提取 → 相关性计算 → 光流估计             │
│  输出：相机位姿 T_i，静态点云                 │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│           D4RT 4D Reconstruction             │
│  输出：动态点的时空轨迹 X_k(t_1...t_n)       │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│         SlidingWindowPredictor               │
│  Kalman Filter 逐点维护运动状态              │
│  滑动窗口更新 → 预测未来 1-2 帧              │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│             Visualizer4D                     │
│  当前点云（白）+ 历史轨迹（蓝）+ 预测（红）   │
└─────────────────────────────────────────────┘
```

---

## 七、详细模块设计

### Module 1：D4RT 输出格式

```
输入：连续视频帧 I_1, I_2, ..., I_N
输出：[N_points × N_frames × 3] 张量

Trajectory_k = {
  t_1: X_k(t_1) = [x, y, z]
  t_2: X_k(t_2) = [x, y, z]
  ...
  t_N: X_k(t_N) = [x, y, z]
}
```

### Module 2：逐点预测器

#### 方案 A：匀速模型

$$\hat{X}_k(t+1) = X_k(t) + \frac{1}{N-1}\sum_{i=1}^{N-1}(X_k(t_{i+1}) - X_k(t_i))$$

适用场景：快速验证，直线匀速运动。

#### 方案 B：Kalman Filter（推荐 MVP）

状态向量：$[x, y, z, v_x, v_y, v_z]$

| | 匀速模型 | Kalman Filter |
|---|---|---|
| 突然加速/转弯 | 预测失效 | 自动适应 |
| 不确定性 | 无 | 给出置信椭球 |
| 计算复杂度 | 极低 | 低 |

#### 方案 C：LSTM（后期升级）

```
输入：过去 N 帧位置序列 [N, 3]
输出：未来 2 帧位置 [2, 3]
适用：复杂周期性运动、多物体交互
```

### Module 3：滑动窗口管理器

**近实时逻辑：**

```
t=1       积累
t=2       积累
...
t=N       第一次预测 → 输出 t=N+1, N+2
t=N+1     滑动更新  → 输出 t=N+2, N+3
t=N+2     滑动更新  → 输出 t=N+3, N+4

延迟 = N 帧处理时间
预测超前 = 1-2 帧
```

### Module 4：可视化

| 元素 | 颜色 | 含义 |
|---|---|---|
| 点云 | 白色 | 当前帧动态点 |
| 线段 | 蓝色 | 历史轨迹 |
| 线段 | 红色虚线 | 预测轨迹 |

---

## 八、完整运行流程

```
初始化：
  window_size   = 8   # 用过去 8 帧
  predict_steps = 2   # 预测未来 2 帧

逐帧处理：

Frame t:
  1. DROID-W 输出相机位姿 T_t
  2. D4RT 输出动态点云 {X_k(t)}
  3. 坐标对齐到世界坐标系（camera → world）
  4. SlidingWindowPredictor.add_frame()

  if t >= window_size:
      5. predictions = predictor.predict()
         → 得到 t+1, t+2 帧的预测位置
      6. Visualizer.update(current, history, predictions)
```

---

## 九、评估指标

### ADE（Average Displacement Error）

$$\text{ADE} = \frac{1}{N} \sum_k \| X_k^{pred}(t) - X_k^{true}(t) \|$$

### FDE（Final Displacement Error）

$$\text{FDE} = \frac{1}{N} \sum_k \| X_k^{pred}(t_{final}) - X_k^{true}(t_{final}) \|$$

---

## 十、项目时间线

| 周次 | 任务 | 目标 |
|---|---|---|
| Week 1 | 跑通 D4RT，理解输出格式；实现匀速预测 | 验证可视化 |
| Week 2 | 替换为 Kalman Filter；实现滑动窗口管理器 | 验证预测质量 |
| Week 3 | 与 DROID-W 集成；坐标系对齐 | 端到端可视化 |
| Week 4 | 评估预测误差（ADE/FDE）；可选升级为 LSTM | 量化结果 |

---

## 十一、代码实现

### 项目文件结构

```
DROID-W/
├── dynamic_prediction/
│   ├── __init__.py           # 模块入口
│   ├── point_predictor.py    # 匀速模型 + Kalman Filter
│   ├── sliding_window.py     # 滑动窗口管理器
│   ├── visualizer_4d.py      # Open3D / Matplotlib 可视化
│   ├── d4rt_bridge.py        # D4RT 输出接口
│   └── evaluator.py          # ADE / FDE 评估
└── run_dynamic_prediction.py # 主运行脚本
```

### 安装依赖

```bash
pip install numpy open3d filterpy matplotlib
```

### 运行命令

```bash
# 合成数据演示（无需 D4RT）
python run_dynamic_prediction.py --mode demo --n_points 80 --n_frames 60

# 真实 D4RT 数据
python run_dynamic_prediction.py --mode real --d4rt_path /path/to/d4rt_output.npz
```

### 核心参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--window_size` | 8 | 滑动窗口大小（帧数） |
| `--predict_steps` | 2 | 预测未来帧数 |
| `--fps` | 10.0 | 目标帧率（0=不限速） |
| `--n_points` | 100 | demo 模式：点数量 |
| `--n_frames` | 50 | demo 模式：总帧数 |

### D4RT 数据接口

D4RT 输出的 `.npz` 文件应包含以下字段：

| 字段 | 形状 | 说明 |
|---|---|---|
| `tracks` | `[N, T, 3]` | 3D 轨迹（必须） |
| `visibility` | `[N, T]` | 可见性 mask（可选） |
| `camera_poses` | `[T, 4, 4]` | 相机位姿（可选，用于坐标系对齐） |

---

## 总结

> **用 D4RT 给每个点建立轨迹历史，用 Kalman Filter 维护运动状态，每帧滑动更新，预测超前 1-2 帧——这就是近实时行为预测系统的最小可行版本。**

### 系统能力对比

| 能力 | 传统 SLAM | DROID-W | 本系统 |
|---|---|---|---|
| 相机定位 | ✅ | ✅ | ✅ |
| 静态建图 | ✅ | ✅ | ✅ |
| 动态点处理 | ❌ 丢弃 | ⚠️ 降权 | ✅ 理解 |
| 物体轨迹 | ❌ | ❌ | ✅ |
| 行为预测 | ❌ | ❌ | ✅ |

---

## 十二、实现记录（2026-05-18）

### 当前实现总结

根据本笔记的系统设计，已将文档描述转化为可运行代码。

#### 新增文件结构

```
dynamic_prediction/
├── __init__.py           # 模块入口，导出核心类
├── point_predictor.py    # ConstantVelocityPredictor + KalmanPointPredictor
├── sliding_window.py     # SlidingWindowPredictor 滑动窗口管理器
├── visualizer_4d.py      # Open3D 实时可视化（附 Matplotlib 后备）
├── d4rt_bridge.py        # D4RTLoader / DROIDWBridge / SyntheticDataGenerator
└── evaluator.py          # ADE / FDE 评估工具

src/
└── dynamic_bridge.py     # DROID-W ↔ dynamic_prediction 集成桥接层

run_dynamic_prediction.py # 主运行脚本（demo / real / droidw 三种模式）
```

#### 对现有代码的修改

| 文件 | 修改内容 |
|---|---|
| `src/slam.py` | `terminate()` 末尾添加动态预测钩子；新增 `run_dynamic_prediction()` 方法 |
| `configs/droid_w.yaml` | 新增 `dynamic_prediction:` 配置段（默认 `enable: False`） |
| `requirements.txt` | 新增 `filterpy>=1.4.5` |

#### 数据流

```
DROID-W video.npz（poses + disps + uncertainties）
        ↓
DROIDWBridge.extract_dynamic_points()   ← 高不确定性像素反投影至世界坐标
        ↓ per-frame [N, 3] 点云
SlidingWindowPredictor.add_frame()      ← 逐帧喂入，维护 Kalman Filter
        ↓ 窗口满后
predictor.predict()                     ← 输出未来 1-2 帧预测位置
        ↓
Visualizer4D.update()                   ← 白/蓝/红 三层可视化
```

#### 立即可运行

```bash
pip install filterpy
python run_dynamic_prediction.py --mode demo --n_points 80 --n_frames 60
```

---

### 为什么不需要训练模型

**Kalman Filter 没有参数需要学习**，它是一个纯数学滤波器。

每来一个新观测值，立即更新状态估计：

$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - H\hat{x}_{t|t-1})$$

不需要"历史数据集"来学习运动规律。只需在初始化时确定：
- 运动模型（匀速：$F$ 矩阵写死）
- 过程噪声 $Q$ 和测量噪声 $R$（手设超参数）

**没有任何权重需要梯度下降优化。**

#### 三种预测方案对比（训练需求）

| 方案 | 需要训练？ | 原因 |
|---|---|---|
| 匀速模型 | ❌ | 纯几何计算，求均值速度 |
| Kalman Filter（当前实现） | ❌ | 数学滤波器，参数是手设超参数 |
| LSTM（Week 4 升级方向） | ✅ 需要 | 神经网络，需要标注轨迹数据训练 |

Kalman Filter 被选为 MVP 的核心理由：**零训练成本，开箱即用，在线实时更新。**