function view_static_env_dynamic_timeline_matlab(globalDir, pointSizeStatic, pointSizeDynamic)
%VIEW_STATIC_ENV_DYNAMIC_TIMELINE_MATLAB
% Show fixed global static map + frame-varying dynamic points.
%
% Required files in globalDir:
%   - global_static_map.mat      (mapPts, mapRgb)
%   - global_dynamic_points.mat  (dynPts, dynRgb, dynFid)
%
% Usage:
%   view_static_env_dynamic_timeline_matlab('Outputs/Bonn/bonn_balloon/slam_global')
%   view_static_env_dynamic_timeline_matlab('Outputs/Bonn/bonn_balloon/slam_global', 2, 8)

if nargin < 2 || isempty(pointSizeStatic)
    pointSizeStatic = 2;
end
if nargin < 3 || isempty(pointSizeDynamic)
    pointSizeDynamic = 8;
end

staticPath = fullfile(globalDir, 'global_static_map.mat');
dynPath = fullfile(globalDir, 'global_dynamic_points.mat');

if ~exist(staticPath, 'file')
    error('Missing file: %s', staticPath);
end
if ~exist(dynPath, 'file')
    error('Missing file: %s', dynPath);
end

S = load(staticPath);
D = load(dynPath);

if ~isfield(S, 'mapPts') || ~isfield(S, 'mapRgb')
    error('global_static_map.mat must contain mapPts and mapRgb.');
end
if ~isfield(D, 'dynPts') || ~isfield(D, 'dynFid')
    error('global_dynamic_points.mat must contain dynPts and dynFid.');
end

mapPts = double(S.mapPts);
mapRgb = double(S.mapRgb);
if isfield(S, 'mapFirstFid')
    mapFirstFid = double(S.mapFirstFid(:));
else
    mapFirstFid = ones(size(mapPts, 1), 1);
end
dynPts = double(D.dynPts);
dynFid = double(D.dynFid(:));

if isfield(D, 'dynRgb')
    dynRgb = double(D.dynRgb);
else
    dynRgb = repmat([1.0, 0.1, 0.03], size(dynPts, 1), 1);
end

nFrames = max(1, round(max([dynFid; mapFirstFid])));

fig = figure('Name', 'Static Environment + Dynamic Timeline', 'Color', 'k');
ax = axes('Parent', fig, 'Position', [0.06, 0.14, 0.90, 0.82]);
hold(ax, 'on');
hSta = scatter3(ax, nan, nan, nan, pointSizeStatic, [0.6, 0.6, 0.6], 'filled');
hDyn = scatter3(ax, nan, nan, nan, pointSizeDynamic, [1.0, 0.1, 0.03], 'filled');
axis(ax, 'equal');
grid(ax, 'on');
view(ax, 3);
xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
set(ax, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

slider = uicontrol( ...
    'Parent', fig, ...
    'Style', 'slider', ...
    'Units', 'normalized', ...
    'Position', [0.06, 0.04, 0.78, 0.05], ...
    'Min', 1, ...
    'Max', nFrames, ...
    'Value', 1, ...
    'SliderStep', [1 / max(nFrames - 1, 1), 10 / max(nFrames - 1, 1)]);

txt = uicontrol( ...
    'Parent', fig, ...
    'Style', 'text', ...
    'Units', 'normalized', ...
    'Position', [0.86, 0.04, 0.10, 0.05], ...
    'String', 'Frame 1', ...
    'BackgroundColor', 'k', ...
    'ForegroundColor', 'w', ...
    'FontWeight', 'bold');

update_frame(1);
slider.Callback = @(src, ~) update_frame(round(src.Value));

    function update_frame(f)
        f = max(1, min(nFrames, f));
        idxSta = (round(mapFirstFid) <= f);
        if any(idxSta)
            ps = mapPts(idxSta, :);
            cs = mapRgb(idxSta, :);
            set(hSta, 'XData', ps(:,1), 'YData', ps(:,2), 'ZData', ps(:,3), 'CData', cs);
        else
            set(hSta, 'XData', nan, 'YData', nan, 'ZData', nan);
        end
        idx = (round(dynFid) == f);
        if any(idx)
            p = dynPts(idx, :);
            c = dynRgb(idx, :);
            set(hDyn, 'XData', p(:,1), 'YData', p(:,2), 'ZData', p(:,3), 'CData', c);
        else
            set(hDyn, 'XData', nan, 'YData', nan, 'ZData', nan);
        end
        title(ax, sprintf('Frame %d / %d | static seen %d/%d | dynamic %d', ...
            f, nFrames, nnz(idxSta), size(mapPts, 1), nnz(idx)), 'Color', 'w');
        txt.String = sprintf('Frame %d', f);
        slider.Value = f;
        drawnow;
    end
end
