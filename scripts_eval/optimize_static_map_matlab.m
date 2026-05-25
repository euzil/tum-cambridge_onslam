function [mapPts, mapRgb] = optimize_static_map_matlab(modelPath, voxelSize, minVoxelCount, knnK, knnStd, showFigure)
%OPTIMIZE_STATIC_MAP_MATLAB Build a cleaner static map from model_4d.mat.
%
% Usage:
%   [pts, rgb] = optimize_static_map_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat');
%   [pts, rgb] = optimize_static_map_matlab('.../model_4d.mat', 0.02, 3, 16, 1.0, true);
%
% Inputs:
%   modelPath      : path to model_4d.mat
%   voxelSize      : voxel size in meters (default 0.02)
%   minVoxelCount  : keep voxels with at least this many points (default 3)
%   knnK           : K for KNN outlier pruning (default 16)
%   knnStd         : outlier threshold = mean + knnStd*std (default 1.0)
%   showFigure     : plot map in a figure (default true)
%
% Outputs:
%   mapPts         : Mx3 fused static points
%   mapRgb         : Mx3 fused colors in [0,1]

if nargin < 2 || isempty(voxelSize)
    voxelSize = 0.02;
end
if nargin < 3 || isempty(minVoxelCount)
    minVoxelCount = 3;
end
if nargin < 4 || isempty(knnK)
    knnK = 16;
end
if nargin < 5 || isempty(knnStd)
    knnStd = 1.0;
end
if nargin < 6 || isempty(showFigure)
    showFigure = true;
end

S = load(modelPath);
pts = double(S.points);
rgb = double(S.colors) / 255.0;

if isfield(S, 'dynamic')
    dyn = logical(S.dynamic(:));
else
    dyn = false(size(pts, 1), 1);
end

if isfield(S, 'uncertainty')
    uncer = double(S.uncertainty(:));
    good = uncer < quantile(uncer, 0.98);
else
    good = true(size(pts, 1), 1);
end

keep = (~dyn) & good;
pts = pts(keep, :);
rgb = rgb(keep, :);

fprintf('[static-opt] input static points: %d\n', size(pts, 1));

% 1) Voxel fusion with averaged geometry/color.
[pts, rgb, voxelCount] = voxel_fuse_xyzrgb(pts, rgb, voxelSize);
fprintf('[static-opt] after voxel fusion: %d\n', size(pts, 1));

% 2) Remove weak voxels (likely unstable/noisy observations).
strong = voxelCount >= minVoxelCount;
pts = pts(strong, :);
rgb = rgb(strong, :);
fprintf('[static-opt] after minVoxelCount=%d: %d\n', minVoxelCount, size(pts, 1));

% 3) KNN statistical outlier removal.
if size(pts, 1) > (knnK + 5)
    try
        [~, dist] = knnsearch(pts, pts, 'K', knnK + 1);
    catch
        dist = squareform(pdist(pts));
        dist = sort(dist, 2, 'ascend');
        dist = dist(:, 1:(knnK + 1));
    end
    meanNbrDist = mean(dist(:, 2:end), 2);
    thr = mean(meanNbrDist) + knnStd * std(meanNbrDist);
    inlier = meanNbrDist <= thr;
    pts = pts(inlier, :);
    rgb = rgb(inlier, :);
    fprintf('[static-opt] after KNN outlier remove: %d\n', size(pts, 1));
end

mapPts = pts;
mapRgb = rgb;

if showFigure
    fig = figure('Name', 'Optimized Static Map', 'Color', 'k');
    ax = axes(fig);
    scatter3(ax, mapPts(:,1), mapPts(:,2), mapPts(:,3), 5, mapRgb, 'filled');
    axis(ax, 'equal');
    grid(ax, 'on');
    view(ax, 3);
    xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
    set(ax, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    title(ax, sprintf('Optimized static map | points=%d | voxel=%.3fm', ...
        size(mapPts, 1), voxelSize), 'Color', 'w');
end
end

function [ptsOut, rgbOut, counts] = voxel_fuse_xyzrgb(pts, rgb, voxel)
ijk = floor(pts ./ voxel);
[~, ~, g] = unique(ijk, 'rows');
n = max(g);
counts = accumarray(g, 1, [n, 1]);

sumP = zeros(n, 3);
sumC = zeros(n, 3);
for d = 1:3
    sumP(:, d) = accumarray(g, pts(:, d), [n, 1]);
    sumC(:, d) = accumarray(g, rgb(:, d), [n, 1]);
end

ptsOut = sumP ./ counts;
rgbOut = sumC ./ counts;
end
