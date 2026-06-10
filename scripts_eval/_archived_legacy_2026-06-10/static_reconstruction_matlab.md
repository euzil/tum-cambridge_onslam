# MATLAB 静态 3D/4D 重建说明

本文档说明如何在 MATLAB 中查看和重建 DROID-W 导出的静态环境点云。

## 1. 基础路径

先在 MATLAB 中进入项目根目录，并加入脚本路径：

```matlab
addpath('scripts_eval')
```

下面示例默认使用：

```matlab
modelPath = 'Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat';
```

如果你的结果在其他序列目录下，请把 `modelPath` 改成对应的 `model_4d.mat` 路径。

## 2. 直接累积显示静态点

如果你想尽量接近原始导出的静态像素点，不做体素降采样，可以运行：

```matlab
view_4d_map_accum_matlab( ...
    modelPath, ...
    inf, ...    % 累积全部帧
    false, ...  % 不显示动态点
    0);         % voxelSize=0，不做 voxel downsample
```

注意：这仍然不是“所有像素点”，而是 `model_4d.mat` 中已经导出的点，并且只显示 `dynamic == false` 的静态点。

## 3. 优化后的静态地图

如果想得到更干净的静态地图，可以运行：

```matlab
[mapPts, mapRgb] = optimize_static_map_matlab( ...
    modelPath, ...
    0.01, ...   % voxelSize，越小点越多
    1, ...      % minVoxelCount=1，避免删掉太多静态点
    16, ...     % KNN 邻居数
    1.0, ...    % 离群点阈值
    true);      % 显示结果
```

这个版本会经过过滤和融合：

- 只保留 `dynamic == false` 的静态点。
- 去掉最高 2% 不确定性点。
- 按 `voxelSize` 做体素融合。
- `minVoxelCount=1` 时，不会因为体素观测次数少而删点。
- 使用 KNN 做离群点过滤。

因此，`optimize_static_map_matlab` 输出的不是所有像素点，而是过滤和融合后的静态点。

## 4. 保存为 PLY

如果已经通过 `optimize_static_map_matlab` 得到 `mapPts` 和 `mapRgb`，可以保存为 PLY：

```matlab
pc = pointCloud(mapPts, 'Color', uint8(mapRgb * 255));
pcwrite(pc, 'Outputs/Bonn/bonn_balloon/static_map_matlab.ply');
```

## 5. 点数太少时的建议

如果你觉得静态环境缺少很多点，优先尝试：

```matlab
view_4d_map_accum_matlab(modelPath, inf, false, 0);
```

如果需要优化版本，则降低过滤强度：

```matlab
[mapPts, mapRgb] = optimize_static_map_matlab(modelPath, 0.005, 1, 16, 1.5, true);
```

其中：

- `voxelSize=0.005` 会比 `0.01` 保留更多细节。
- `minVoxelCount=1` 避免删除只出现一次的真实静态表面。
- `knnStd=1.5` 比 `1.0` 更宽松，会删掉更少离群点。
