function view_4d_comparison_matlab(rootDir, fps, axisMode, showPrediction)
%VIEW_4D_COMPARISON_MATLAB Synchronized MATLAB viewer for three 4D models.
%
% Usage:
%   addpath('scripts_eval');
%   view_4d_comparison_matlab('Outputs/Bonn/bonn_balloon/matlab_4d_comparison')
%   view_4d_comparison_matlab('Outputs/Bonn/bonn_balloon/matlab_4d_comparison', 8, 'none', true)

if nargin < 1 || isempty(rootDir)
    rootDir = 'Outputs/Bonn/bonn_balloon/matlab_4d_comparison';
end
if nargin < 2 || isempty(fps)
    fps = 8;
end
if nargin < 3 || isempty(axisMode)
    axisMode = 'none';
end
if nargin < 4 || isempty(showPrediction)
    showPrediction = true;
end

names = {'Baseline static', 'Kalman previous', 'Learned region'};
files = {
    fullfile(rootDir, 'baseline_static', 'model_4d.mat')
    fullfile(rootDir, 'kalman_previous', 'model_4d.mat')
    fullfile(rootDir, 'learned_region', 'model_4d.mat')
};

S = cell(1, 3);
for i = 1:3
    if ~isfile(files{i})
        error('Missing model file: %s', files{i});
    end
    S{i} = load(files{i});
end

nFrames = min([numel(S{1}.frame_start_1), numel(S{2}.frame_start_1), numel(S{3}.frame_start_1)]);
delay = 1 / max(fps, eps);

fig = figure('Name', 'Baseline vs Kalman vs Learned 4D models', 'Color', 'k');
tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
axesList = gobjects(1, 3);
for i = 1:3
    axesList(i) = nexttile;
    axis(axesList(i), 'equal');
    grid(axesList(i), 'on');
    view(axesList(i), 3);
    xlabel(axesList(i), 'X');
    ylabel(axesList(i), 'Y');
    zlabel(axesList(i), 'Z');
    set(axesList(i), 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
end

for t = 1:nFrames
    if ~isvalid(fig)
        break;
    end

    for i = 1:3
        ax = axesList(i);
        cla(ax);

        lo = double(S{i}.frame_start_1(t));
        hi = double(S{i}.frame_end_1(t));
        if hi < lo
            continue;
        end

        pts = apply_axis_mode(double(S{i}.points(lo:hi, :)), axisMode);
        rgb = double(S{i}.colors(lo:hi, :)) / 255.0;
        dyn = logical(S{i}.dynamic(lo:hi));
        rgb(dyn, :) = repmat([1.0, 0.05, 0.02], nnz(dyn), 1);

        stepAll = max(1, floor(size(pts, 1) / 8000));
        scatter3(ax, pts(1:stepAll:end, 1), pts(1:stepAll:end, 2), pts(1:stepAll:end, 3), ...
            4, rgb(1:stepAll:end, :), 'filled');
        hold(ax, 'on');

        predCount = 0;
        if showPrediction && isfield(S{i}, 'predicted_next_points') && isfield(S{i}, 'motion_valid')
            mv = logical(S{i}.motion_valid(lo:hi));
            mv = mv(:) & dyn(:);
            if any(mv)
                pred = apply_axis_mode(double(S{i}.predicted_next_points(lo:hi, :)), axisMode);
                pred = pred(mv, :);
                src = pts(mv, :);
                predCount = size(pred, 1);

                stepPred = max(1, floor(predCount / 1000));
                scatter3(ax, pred(1:stepPred:end, 1), pred(1:stepPred:end, 2), pred(1:stepPred:end, 3), ...
                    12, repmat([0.05, 0.75, 1.0], numel(1:stepPred:predCount), 1), 'filled');

                stepVec = max(1, floor(predCount / 200));
                q0 = src(1:stepVec:end, :);
                q1 = pred(1:stepVec:end, :) - q0;
                quiver3(ax, q0(:,1), q0(:,2), q0(:,3), q1(:,1), q1(:,2), q1(:,3), ...
                    0, 'Color', [0.9, 0.9, 1.0], 'LineWidth', 0.65);
            end
        end

        if t < nFrames
            nlo = double(S{i}.frame_start_1(t + 1));
            nhi = double(S{i}.frame_end_1(t + 1));
            nextDyn = logical(S{i}.dynamic(nlo:nhi));
            if any(nextDyn)
                nextPts = apply_axis_mode(double(S{i}.points(nlo:nhi, :)), axisMode);
                nextPts = nextPts(nextDyn, :);
                stepNext = max(1, floor(size(nextPts, 1) / 1000));
                scatter3(ax, nextPts(1:stepNext:end, 1), nextPts(1:stepNext:end, 2), nextPts(1:stepNext:end, 3), ...
                    8, repmat([0.1, 1.0, 0.2], numel(1:stepNext:size(nextPts, 1)), 1), 'filled');
            end
        end

        hold(ax, 'off');
        title(ax, sprintf('%s | frame %d/%d | dyn %d | pred %d', ...
            names{i}, t, nFrames, nnz(dyn), predCount), 'Color', 'w');
        axis(ax, 'equal');
        grid(ax, 'on');
    end

    drawnow;
    pause(delay);
end
end

function pts = apply_axis_mode(pts, mode)
switch lower(string(mode))
    case "none"
        return;
    case "swap_xy"
        pts = pts(:, [2, 1, 3]);
    case "swap_xz"
        pts = pts(:, [3, 2, 1]);
    case "swap_yz"
        pts = pts(:, [1, 3, 2]);
    case "flip_x"
        pts(:, 1) = -pts(:, 1);
    case "flip_y"
        pts(:, 2) = -pts(:, 2);
    case "flip_z"
        pts(:, 3) = -pts(:, 3);
    case "flip_xy"
        pts(:, 1:2) = -pts(:, 1:2);
    case "flip_xz"
        pts(:, [1, 3]) = -pts(:, [1, 3]);
    case "flip_yz"
        pts(:, 2:3) = -pts(:, 2:3);
    case "flip_xyz"
        pts = -pts;
    otherwise
        error('Unknown axisMode: %s', mode);
end
end
