function view_static_env_dynamic_timeline_matlab(globalDir, pointSizeStatic, pointSizeDynamic, modelPath, pointSizePred)
%VIEW_STATIC_ENV_DYNAMIC_TIMELINE_MATLAB
% Show fixed global static map + frame-varying dynamic points.
%
% Required files in globalDir:
%   - global_static_map.mat      (mapPts, mapRgb)
%   - global_dynamic_points.mat  (dynPts, dynRgb, dynFid)
%
% Optional:
%   - model_4d.mat (for per-frame predicted_next_points overlay)
%   - Next-frame dynamic points are shown in green for prediction comparison
%
% Usage:
%   view_static_env_dynamic_timeline_matlab('Outputs/Bonn/bonn_balloon/slam_global')
%   view_static_env_dynamic_timeline_matlab('Outputs/Bonn/bonn_balloon/slam_global', 2, 8)
%   view_static_env_dynamic_timeline_matlab('.../slam_global', 2, 8, ...
%       'Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', 10)

if nargin < 2 || isempty(pointSizeStatic)
    pointSizeStatic = 2;
end
if nargin < 3 || isempty(pointSizeDynamic)
    pointSizeDynamic = 8;
end
if nargin < 4
    modelPath = '';
end
if nargin < 5 || isempty(pointSizePred)
    pointSizePred = 12;
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
M = [];
hasPred = false;
hasMotionValid = false;
if ~isempty(modelPath) && exist(modelPath, 'file')
    M = load(modelPath);
    hasPred = isfield(M, 'predicted_next_points') && isfield(M, 'frame_start_1') && isfield(M, 'frame_end_1');
    hasMotionValid = isfield(M, 'motion_valid');
    if hasPred
        nFrames = max(nFrames, numel(M.frame_start_1));
    end
end

fig = figure('Name', 'Static Environment + Dynamic Timeline', 'Color', 'k');
ax = axes('Parent', fig, 'Position', [0.06, 0.14, 0.90, 0.82]);
hold(ax, 'on');
hSta = scatter3(ax, nan, nan, nan, pointSizeStatic, [0.6, 0.6, 0.6], 'filled');
hDyn = scatter3(ax, nan, nan, nan, pointSizeDynamic, [1.0, 0.1, 0.03], 'filled');
hPred = scatter3(ax, nan, nan, nan, pointSizePred, [0.05, 0.95, 1.0], 'filled');
hNextDyn = scatter3(ax, nan, nan, nan, pointSizeDynamic, [0.1, 1.0, 0.2], 'filled');
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
        idxNext = (round(dynFid) == (f + 1));
        if any(idx)
            p = dynPts(idx, :);
            c = dynRgb(idx, :);
            set(hDyn, 'XData', p(:,1), 'YData', p(:,2), 'ZData', p(:,3), 'CData', c);
        else
            set(hDyn, 'XData', nan, 'YData', nan, 'ZData', nan);
        end
        if any(idxNext)
            pNext = dynPts(idxNext, :);
            set(hNextDyn, 'XData', pNext(:,1), 'YData', pNext(:,2), 'ZData', pNext(:,3), ...
                'CData', repmat([0.1, 1.0, 0.2], size(pNext, 1), 1));
        else
            pNext = zeros(0, 3);
            set(hNextDyn, 'XData', nan, 'YData', nan, 'ZData', nan);
        end

        nPred = 0;
        predErrMean = NaN;
        predErrMedian = NaN;
        if hasPred && f <= numel(M.frame_start_1)
            lo = double(M.frame_start_1(f));
            hi = double(M.frame_end_1(f));
            if hi >= lo
                pred = double(M.predicted_next_points(lo:hi, :));
                if isfield(M, 'dynamic')
                    dmask = logical(M.dynamic(lo:hi));
                    pred = pred(dmask, :); % show predicted points for dynamic subset
                    if hasMotionValid
                        mvalid = logical(M.motion_valid(lo:hi));
                        pred = pred(mvalid(dmask), :); % only keep points with real predicted motion
                    end
                elseif hasMotionValid
                    mvalid = logical(M.motion_valid(lo:hi));
                    pred = pred(mvalid, :); % conservative fallback
                end
                if ~isempty(pred)
                    set(hPred, 'XData', pred(:,1), 'YData', pred(:,2), 'ZData', pred(:,3), ...
                        'CData', repmat([0.05, 0.95, 1.0], size(pred, 1), 1));
                    nPred = size(pred, 1);
                    if ~isempty(pNext)
                        d = nearest_distances(pred, pNext);
                        predErrMean = mean(d);
                        predErrMedian = median(d);
                    end
                else
                    set(hPred, 'XData', nan, 'YData', nan, 'ZData', nan);
                end
            else
                set(hPred, 'XData', nan, 'YData', nan, 'ZData', nan);
            end
        else
            set(hPred, 'XData', nan, 'YData', nan, 'ZData', nan);
        end
        if isfinite(predErrMean)
            title(ax, sprintf(['Frame %d / %d | static seen %d/%d | dynamic(t) %d | ', ...
                'pred(t->t+1) %d | dynamic(t+1) %d | pred->next mean %.3fm med %.3fm'], ...
                f, nFrames, nnz(idxSta), size(mapPts, 1), nnz(idx), nPred, nnz(idxNext), ...
                predErrMean, predErrMedian), 'Color', 'w');
        else
            title(ax, sprintf(['Frame %d / %d | static seen %d/%d | dynamic(t) %d | ', ...
                'pred(t->t+1) %d | dynamic(t+1) %d'], ...
                f, nFrames, nnz(idxSta), size(mapPts, 1), nnz(idx), nPred, nnz(idxNext)), 'Color', 'w');
        end
        txt.String = sprintf('Frame %d', f);
        slider.Value = f;
        drawnow;
    end
end

function d = nearest_distances(srcPts, dstPts)
if isempty(srcPts) || isempty(dstPts)
    d = [];
    return;
end

n = size(srcPts, 1);
d = inf(n, 1);
chunk = 512;
for i = 1:chunk:n
    j = min(i + chunk - 1, n);
    a = srcPts(i:j, :);
    dx = a(:, 1) - dstPts(:, 1).';
    dy = a(:, 2) - dstPts(:, 2).';
    dz = a(:, 3) - dstPts(:, 3).';
    dist2 = dx.^2 + dy.^2 + dz.^2;
    d(i:j) = sqrt(min(dist2, [], 2));
end
end
