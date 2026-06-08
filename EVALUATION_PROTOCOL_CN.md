# 论文口径评估方案

本文档用于将 `D4RT_paper.pdf` 与 `droid_w_paper.pdf` 的评估方法迁移到本项目，对你的动态 SLAM / 4D 方法进行公平评估，并组织成能突出方法优势的实验结果。

## 1. 核心结论先行

为了让结果具有说服力，**主实验表必须严格使用已有文献指标**。本项目内部设计的指标可以保留，但只能作为补充分析或 ablation，不能作为“优于其他方法”的主要证据。

建议把指标分成三层：

1. **主指标：相机轨迹精度**
   - 沿用 DROID-W：`ATE RMSE`，所有方法都用 `Sim(3) Umeyama` 与 GT 对齐。
   - 在 TUM / Bonn 中用 `cm`；DROID-W 室外长序列用 `m`；DyCheck 可补充归一化 ATE。

2. **主指标：D4RT / TAPVid-3D 动态 3D tracking**
   - 严格沿用 D4RT：`APD3D`、`OA`、`AJ`。
   - 如果评估 world-coordinate tracking，则沿用 D4RT：`APD3D` 和 `L1`。
   - 这些指标来自已有 3D tracking benchmark，比自定义最近邻误差更有说服力。

3. **主指标：几何重建 / 深度 / 位姿稳定性**
   - D4RT pose：`ATE`、`RPE-T`、`RPE-R`、`Pose AUC@30`。
   - D4RT point cloud reconstruction：mean-shift 对齐后 `mean L1`。
   - D4RT depth：`AbsRel (S)` 和 `AbsRel (SS)`。

4. **效率与消融**
   - 沿用 DROID-W：`FPS` 与 ablation。
   - 消融建议：去掉动态反馈、去掉运动预测、去掉不确定性/动态 mask、使用原始 DROID-SLAM。

5. **补充指标：本项目动态 4D 诊断**
   - `motion_nn_mean_m`、`motion_nn_median_m`、`motion_inlier_ratio`、`motion_vs_static_gain_percent`、`axis_bias_l2_m` 只能作为 supplementary。
   - 这些指标可以解释你的模块为什么有效，但不能替代文献指标。

## 2. 数据集与实验矩阵

优先使用与 DROID-W 论文一致的数据集：

| Dataset | 推荐序列 | 主指标 | 单位 | 用途 |
|---|---:|---|---|---|
| Bonn RGB-D Dynamic | 论文 8 个动态序列 | ATE RMSE | cm | 强动态 RGB-D，对 SLAM 轨迹最有说服力 |
| TUM RGB-D Dynamic | `freiburg3_walking_*`, `freiburg3_sitting_*`, `freiburg2_desk_with_person` | ATE RMSE | cm | 标准动态 SLAM 对比 |
| DyCheck | 12 个动态场景 | normalized ATE / ATE | dataset scale | 验证非刚体/复杂运动鲁棒性 |
| DROID-W | Downtown 1-7 | ATE RMSE | m | 长轨迹真实户外场景 |
| YouTube | 无 GT | qualitative + reconstruction consistency | - | 展示真实 in-the-wild 效果 |

如果时间有限，最小实验集建议：

| 目的 | Dataset |
|---|---|
| 证明轨迹精度 | Bonn + TUM |
| 证明动态 4D 建模 | TUM walking / Bonn crowd 类动态序列 |
| 证明真实复杂场景鲁棒性 | DROID-W Downtown 或 YouTube |

## 3. Baseline 设置

必须包含：

| Baseline | 作用 |
|---|---|
| DROID-SLAM | 静态 SLAM 强基线，证明动态场景中失效点 |
| DROID-W | 论文方法主 baseline，证明你的改进不是只赢传统方法 |
| 本项目 `switch_baseline` | 如果它代表“无你的模块”的版本，可作为最直接 ablation |

可选包含：

| Baseline | 作用 |
|---|---|
| ORB-SLAM2 / DSO | 传统 SLAM 对比，但复现实验成本较高 |
| WildGS-SLAM | 动态不确定性 / Gaussian SLAM 对比 |
| MonST3R / MegaSaM / D4RT | 若你要强调 4D reconstruction，可作为 feed-forward/重建类对比 |

## 4. 严格文献指标定义

### 4.1 Camera Tracking

**ATE RMSE**

- 论文口径：估计轨迹与 GT 轨迹做 `Sim(3)` 对齐，然后计算 translation APE 的 RMSE。
- 本项目位置：[src/utils/eval_traj.py](/home/youran/文档/DROID-W/TUM-Cambridge_onSLAM/src/utils/eval_traj.py)
- 已实现：`traj_est.align(traj_ref, correct_scale=True)` + `evo.core.metrics.APE(translation_part)`。

报告格式：

| Method | Bonn Avg ATE cm ↓ | TUM Avg ATE cm ↓ | DyCheck Avg ↓ | DROID-W Avg m ↓ |
|---|---:|---:|---:|---:|
| DROID-SLAM | | | | |
| DROID-W | | | | |
| Ours | | | | |

这是最重要的 DROID-W 主表。你的方法如果要证明 SLAM tracking 优势，必须至少在 Bonn / TUM / DyCheck 上复现这个表格。

### 4.2 D4RT Pose Metrics

D4RT 的 pose table 报告：

- `RPE-T`：相对位移误差。
- `RPE-R`：相对旋转误差。
- `ATE`：绝对平移误差。
- `Pose AUC@30`：姿态精度曲线在阈值 30 内的面积。

论文口径：

- 所有方法先与 GT 做 `Sim(3)` alignment。
- 然后报告 `ATE`、`RPE-T`、`RPE-R`。
- 如果要与 VGGT / MegaSaM / D4RT 等 feed-forward 方法对比，建议加入 `Pose AUC@30`。

建议表格：

| Method | ATE ↓ | RPE-T ↓ | RPE-R ↓ | Pose AUC@30 ↑ |
|---|---:|---:|---:|---:|
| DROID-W | | | | |
| Ours | | | | |

### 4.3 D4RT / TAPVid-3D Dynamic 3D Tracking

D4RT 在 4D reconstruction and tracking 中使用 TAPVid-3D 标准协议。若你的方法声称具备动态 4D 能力，应优先复现这组指标。

Local camera coordinate tracking：

| Metric | 越好方向 | 文献含义 |
|---|---|---|
| `APD3D` | ↑ | Average percent of points within 3D delta error |
| `OA` | ↑ | Occlusion Accuracy |
| `AJ` | ↑ | 3D Average Jaccard |

World coordinate tracking：

| Metric | 越好方向 | 文献含义 |
|---|---|---|
| `APD3D` | ↑ | world-coordinate tracks 的 3D 阈值准确率 |
| `L1` | ↓ | predicted tracks 与 GT tracks 的平均 L1 偏差 |

建议表格：

| Method | Local APD3D ↑ | Local OA ↑ | Local AJ ↑ | World APD3D ↑ | World L1 ↓ |
|---|---:|---:|---:|---:|---:|
| SpatialTrackerV2 | | | | | |
| D4RT | | | | | |
| Ours | | | | | |

注意：这组指标需要有 3D point tracks GT。如果当前数据没有 TAPVid-3D 或等价 GT tracks，就不能声称“严格复现 D4RT tracking 指标”，只能报告 DROID-W tracking 指标和补充几何诊断。

### 4.4 D4RT Point Cloud Reconstruction

D4RT 的 point cloud reconstruction 指标：

- Dataset：Sintel、ScanNet。
- Protocol：predicted point cloud 与 GT point cloud 先做 mean-shift alignment。
- Metric：`mean L1 distance`。

建议表格：

| Method | Sintel L1 ↓ | ScanNet L1 ↓ |
|---|---:|---:|
| MegaSaM | | |
| VGGT / pi3 | | |
| D4RT | | |
| Ours | | |

### 4.5 D4RT Depth Metrics

D4RT 的 depth 指标：

- `AbsRel (S)`：scale-only alignment 后的 AbsRel。
- `AbsRel (SS)`：scale-and-shift alignment 后的 AbsRel。

建议表格：

| Method | Sintel AbsRel(S) ↓ | Sintel AbsRel(SS) ↓ | ScanNet AbsRel(S) ↓ | Bonn AbsRel(S) ↓ |
|---|---:|---:|---:|---:|
| MegaSaM | | | | |
| D4RT | | | | |
| Ours | | | | |

如果你的方法不直接输出 dense depth，可以不做这张表；不要用不等价输出强行比较。

### 4.6 Supplementary: Dynamic 4D Diagnostics

本项目当前已有动态 4D 诊断脚本：

- [scripts_eval/evaluate_dynamic_method.py](/home/youran/文档/DROID-W/TUM-Cambridge_onSLAM/scripts_eval/evaluate_dynamic_method.py)
- [scripts_eval/compare_dynamic_4d_modeling.py](/home/youran/文档/DROID-W/TUM-Cambridge_onSLAM/scripts_eval/compare_dynamic_4d_modeling.py)

这些指标不是 D4RT 或 DROID-W 原文主指标，因此只建议用于 supplementary：

| Metric | 越好方向 | 含义 |
|---|---|---|
| `motion_nn_mean_m` | ↓ | 预测动态点到下一帧动态点云的平均最近邻距离 |
| `motion_nn_median_m` | ↓ | 中位误差，抗 outlier |
| `motion_inlier_ratio` | ↑ | 预测点落入阈值内的比例 |
| `motion_vs_static_gain_percent` | ↑ | 相比“动态点不动”的提升百分比 |
| `axis_bias_l2_m` | ↓ | XYZ 平均偏差的 L2，衡量系统性漂移 |

运行示例：

```bash
python scripts_eval/compare_dynamic_4d_modeling.py \
  --baseline-model path/to/baseline/model_4d.npz \
  --learned-model path/to/ours/model_4d.npz \
  --baseline-name DROID-W \
  --learned-name Ours \
  --max-nn-dist 0.5 \
  --out-prefix outputs/summary/ours_vs_droidw_dynamic4d
```

补充表格：

| Method | Motion NN mean m ↓ | Motion NN median m ↓ | Inlier ratio ↑ | Gain over static ↑ | Axis bias m ↓ |
|---|---:|---:|---:|---:|---:|
| DROID-W / baseline | | | | | |
| Ours | | | | | |

### 4.7 Runtime

沿用 DROID-W 表格：

| Method | Bonn FPS ↑ | TUM FPS ↑ | DyCheck FPS ↑ |
|---|---:|---:|---:|
| DROID-SLAM | | | |
| DROID-W | | | |
| Ours | | | |

注意：FPS 需要统一硬件、分辨率、帧数和是否启用 mapping。

## 5. 如何突出你的方法优势

建议论文/报告中的论证顺序：

1. **轨迹不退化或更好**
   - 先用 `ATE RMSE` 对齐 DROID-W 论文主指标。
   - 结论句模板：`Ours achieves lower average ATE than DROID-W on dynamic sequences, indicating that explicit dynamic prediction improves camera tracking under moving objects.`

2. **动态 4D 能力用文献指标证明**
   - 如果有 TAPVid-3D / 等价 GT tracks，用 `APD3D`、`OA`、`AJ`、`World L1` 作为核心证据。
   - 如果没有 3D track GT，就不要把 `motion_nn_*` 写成主指标；应表述为“supplementary diagnostic metrics”。

3. **消融证明每个模块必要**
   - `Ours w/o motion prediction`
   - `Ours w/o dynamic feedback`
   - `Ours w/o dynamic BA weighting`
   - `Ours full`

4. **真实视频定性对比**
   - 展示 DROID-SLAM 的动态拖影/重复结构/尺度漂移。
   - 展示你的方法过滤动态区域、预测运动、静态地图更一致。

## 6. 建议最终实验表

### Table A: Camera Tracking

| Method | Bonn Avg ATE cm ↓ | TUM Avg ATE cm ↓ | DyCheck Avg ↓ |
|---|---:|---:|---:|
| DROID-SLAM | | | |
| DROID-W | | | |
| Ours | | | |

### Table B: Per-sequence ATE

| Dataset | Sequence | DROID-SLAM | DROID-W | Ours |
|---|---|---:|---:|---:|
| TUM | freiburg3_walking_xyz | | | |
| TUM | freiburg3_walking_halfsphere | | | |
| Bonn | crowd | | | |

### Table C: Dynamic 3D Tracking, D4RT / TAPVid-3D Protocol

| Method | Local APD3D ↑ | Local OA ↑ | Local AJ ↑ | World APD3D ↑ | World L1 ↓ |
|---|---:|---:|---:|---:|---:|
| SpatialTrackerV2 | | | | | |
| D4RT | | | | | |
| Ours | | | | | |

### Table D: Dynamic 4D Diagnostics, Supplementary Only

| Sequence | Method | Motion NN mean m ↓ | Motion NN median m ↓ | Inlier ratio ↑ | Gain over static ↑ |
|---|---|---:|---:|---:|---:|
| sequence-1 | baseline | | | | |
| sequence-1 | ours | | | | |

### Table E: Ablation

| Variant | ATE RMSE ↓ | Motion NN mean m ↓ | Inlier ratio ↑ | FPS ↑ |
|---|---:|---:|---:|---:|
| Full model | | | | |
| w/o motion prediction | | | | |
| w/o dynamic feedback | | | | |
| w/o uncertainty/mask weighting | | | | |

## 7. 已实现脚本

当前项目已补充以下严格文献指标脚本：

1. `scripts_eval/evaluate_pose_paper_metrics.py`
   - 输入估计轨迹和 GT。
   - 输出 `ATE`, `RPE-T`, `RPE-R`, `Pose AUC@30`。
   - 与 D4RT 的 pose table 对齐。
   - 也可复现 DROID-W 的 `ATE RMSE + Sim(3) Umeyama alignment`。

运行示例：

```bash
python scripts_eval/evaluate_pose_paper_metrics.py \
  --est outputs/your_run/traj/est_poses_full.txt \
  --gt datasets/TUM_RGBD/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt \
  --associate-by-index \
  --out-prefix outputs/summary/your_run_pose_paper_metrics
```

如果估计轨迹和 GT 都使用真实时间戳，去掉 `--associate-by-index`，脚本会按 `--max-time-diff` 做 timestamp association。

DyCheck 按 DROID-W 论文做轨迹长度归一化：

```bash
python scripts_eval/evaluate_pose_paper_metrics.py \
  --est outputs/your_run/traj/est_poses_full.txt \
  --gt path/to/dycheck_groundtruth.txt \
  --normalize-gt-length \
  --out-prefix outputs/summary/your_run_dycheck_pose_metrics
```

2. `scripts_eval/evaluate_tapvid3d_metrics.py`
   - 输入 predicted 3D tracks、GT 3D tracks、occlusion labels。
   - 输出 `APD3D`, `OA`, `AJ`, `World L1`。
   - 与 D4RT 的 4D reconstruction and tracking table 对齐。

输入 `.npz` 需要包含：

- `pred_tracks`: `[N,T,3]`
- `gt_tracks`: `[N,T,3]`
- `visible` 或 `gt_visible`: `[N,T]`
- 可选 `pred_visible`: `[N,T]`

运行示例：

```bash
python scripts_eval/evaluate_tapvid3d_metrics.py \
  --tracks outputs/your_run/tapvid3d_tracks.npz \
  --focal-length 525.0 \
  --out-prefix outputs/summary/your_run_tapvid3d
```

如果没有相机焦距，只能退化为绝对 3D 阈值模式。此时应在论文中明确说明不是完整 TAPVid-3D depth-adaptive protocol：

```bash
python scripts_eval/evaluate_tapvid3d_metrics.py \
  --tracks outputs/your_run/tapvid3d_tracks.npz \
  --absolute-thresholds 0.01,0.02,0.04,0.08,0.16 \
  --out-prefix outputs/summary/your_run_tapvid3d_absolute
```

3. `scripts_eval/evaluate_reconstruction_l1.py`
   - 输入 predicted point cloud 与 GT point cloud。
   - 按 D4RT：mean-shift alignment 后输出 `mean L1`。

运行示例：

```bash
python scripts_eval/evaluate_reconstruction_l1.py \
  --pred outputs/your_run/pred_points.npz \
  --gt path/to/gt_points.npz \
  --out-prefix outputs/summary/your_run_reconstruction_l1
```

4. `scripts_eval/evaluate_depth_absrel.py`
   - 输入 predicted depth 与 GT depth。
   - 输出 D4RT depth metrics：`AbsRel(S)` 和 `AbsRel(SS)`。

运行示例：

```bash
python scripts_eval/evaluate_depth_absrel.py \
  --pred outputs/your_run/pred_depth.npz \
  --gt path/to/gt_depth.npz \
  --out-prefix outputs/summary/your_run_depth_absrel
```

这样最后可以直接生成论文式结果表，并且每个数字都能追溯到脚本和输出文件。


python scripts_eval/evaluate_pose_paper_metrics.py \
  --est outputs/Bonn/bonn_crowd/traj/est_poses_full.txt \
  --gt datasets/Bonn/rgbd_bonn_crowd/groundtruth.txt \
  --associate-by-index \
  --out-prefix eval/summary/bonn_crowd

python scripts_eval/evaluate_pose_paper_metrics.py \
  --est Outputs/Bonn/bonn_crowd/traj/est_poses_full.txt \
  --gt datasets/Bonn/rgbd_bonn_crowd/groundtruth.txt \
  --associate-by-index \
  --out-prefix eval/summary/bonn_crowd_baseline