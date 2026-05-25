function view_4d_map_accum_matlab(modelPath, windowSize, showDynamic, voxelSize)
%VIEW_4D_MAP_ACCUM_MATLAB Visualize an accumulated 4D map in MATLAB.
%
% Usage:
%   view_4d_map_accum_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat')
%   view_4d_map_accum_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', 53, false, 0.03)
%
% Inputs:
%   modelPath   : path to model_4d.mat (from export_4d_model.py)
%   windowSize  : number of frames to accumulate (default: all frames)
%   showDynamic : whether to overlay dynamic points in red (default: false)
%   voxelSize   : voxel size in meters for downsampling (default: 0.03)

if nargin < 2 || isempty(windowSize)
    windowSize = inf;
end
if nargin < 3 || isempty(showDynamic)
    showDynamic = false;
end
if nargin < 4 || isempty(voxelSize)
    voxelSize = 0.03;
end

S = load(modelPath);
nFrames = numel(S.frame_start_1);

if isinf(windowSize)
    windowSize = nFrames;
else
    windowSize = max(1, min(nFrames, round(windowSize)));
end

ptsStatic = [];
rgbStatic = [];
ptsDynamic = [];

startFrame = max(1, nFrames - windowSize + 1);
for t = startFrame:nFrames
    lo = double(S.frame_start_1(t));
    hi = double(S.frame_end_1(t));
    if hi < lo
        continue;
    end

    pts = double(S.points(lo:hi, :));
    rgb = double(S.colors(lo:hi, :)) / 255.0;
    dyn = logical(S.dynamic(lo:hi));

    ptsStatic = [ptsStatic; pts(~dyn, :)]; %#ok<AGROW>
    rgbStatic = [rgbStatic; rgb(~dyn, :)]; %#ok<AGROW>
    if showDynamic
        ptsDynamic = [ptsDynamic; pts(dyn, :)]; %#ok<AGROW>
    end
end

if ~isempty(ptsStatic) && voxelSize > 0
    [ptsStatic, rgbStatic] = voxel_downsample_xyzrgb(ptsStatic, rgbStatic, voxelSize);
end
if ~isempty(ptsDynamic) && voxelSize > 0
    ptsDynamic = voxel_downsample_xyz(ptsDynamic, voxelSize);
end

fig = figure('Name', 'Accumulated static 4D map', 'Color', 'k');
ax = axes(fig);
axis(ax, 'equal');
grid(ax, 'on');
view(ax, 3);
xlabel(ax, 'X');
ylabel(ax, 'Y');
zlabel(ax, 'Z');
set(ax, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

if ~isempty(ptsStatic)
    scatter3(ax, ptsStatic(:,1), ptsStatic(:,2), ptsStatic(:,3), 4, rgbStatic, 'filled');
    hold(ax, 'on');
end
if showDynamic && ~isempty(ptsDynamic)
    scatter3(ax, ptsDynamic(:,1), ptsDynamic(:,2), ptsDynamic(:,3), 5, ...
        repmat([1.0, 0.1, 0.05], size(ptsDynamic, 1), 1), 'filled');
end
hold(ax, 'off');

title(ax, sprintf(['Accumulated map | frames=%d..%d | static=%d | dynamic=%d | ', ...
    'voxel=%.3fm'], startFrame, nFrames, size(ptsStatic, 1), size(ptsDynamic, 1), voxelSize), 'Color', 'w');
axis(ax, 'equal');
grid(ax, 'on');
end

function [ptsOut, rgbOut] = voxel_downsample_xyzrgb(pts, rgb, voxel)
ijk = floor(pts ./ voxel);
[~, ~, g] = unique(ijk, 'rows');

n = max(g);
sumP = zeros(n, 3);
sumC = zeros(n, 3);
cnt = accumarray(g, 1, [n, 1]);
for d = 1:3
    sumP(:, d) = accumarray(g, pts(:, d), [n, 1]);
    sumC(:, d) = accumarray(g, rgb(:, d), [n, 1]);
end

ptsOut = sumP ./ cnt;
rgbOut = sumC ./ cnt;
end

function ptsOut = voxel_downsample_xyz(pts, voxel)
ijk = floor(pts ./ voxel);
[~, ia, ~] = unique(ijk, 'rows', 'stable');
ptsOut = pts(ia, :);
end
