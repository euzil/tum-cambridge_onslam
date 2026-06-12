function view_4d_model_matlab(modelPath, fps, dynamicRed, axisMode, showPrediction)
%VIEW_4D_MODEL_MATLAB Play an exported DROID-W 4D model in MATLAB.
%
% Usage:
%   view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat')
%   view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat', 8, true)
%   view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat', 8, true, 'swap_yz')
%   view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat', 8, true, 'none', true)
%
% Expected variables in model_4d.mat:
%   points          Nx3 single, world coordinates
%   colors          Nx3 uint8, RGB
%   dynamic         Nx1 logical
%   frame_start_1   Tx1 int, 1-based inclusive start index
%   frame_end_1     Tx1 int, 1-based inclusive end index
%   predicted_next_points Nx3 single, predicted next-frame positions

if nargin < 2 || isempty(fps)
    fps = 8;
end
if nargin < 3 || isempty(dynamicRed)
    dynamicRed = true;
end
if nargin < 4 || isempty(axisMode)
    axisMode = 'none';
end
if nargin < 5 || isempty(showPrediction)
    showPrediction = true;
end

S = load(modelPath);
nFrames = numel(S.frame_start_1);
hasPrediction = isfield(S, 'predicted_next_points') && isfield(S, 'motion_valid');

fig = figure('Name', 'DROID-W 4D MATLAB model', 'Color', 'k');
ax = axes(fig);
axis(ax, 'equal');
grid(ax, 'on');
view(ax, 3);
xlabel(ax, 'X');
ylabel(ax, 'Y');
zlabel(ax, 'Z');
set(ax, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

delay = 1 / max(fps, eps);

for t = 1:nFrames
    if ~isvalid(fig)
        break;
    end

    lo = double(S.frame_start_1(t));
    hi = double(S.frame_end_1(t));
    if hi < lo
        continue;
    end

    pts_raw = double(S.points(lo:hi, :));
    pts = apply_axis_mode(pts_raw, axisMode);
    rgb = double(S.colors(lo:hi, :)) / 255.0;
    dyn = logical(S.dynamic(lo:hi));

    if dynamicRed
        rgb(dyn, :) = repmat([1.0, 0.05, 0.02], nnz(dyn), 1);
    end

    cla(ax);
    scatter3(ax, pts(:,1), pts(:,2), pts(:,3), 5, rgb, 'filled');
    hold(ax, 'on');

    predCount = 0;
    if showPrediction && hasPrediction
        pred_raw = double(S.predicted_next_points(lo:hi, :));
        mv = logical(S.motion_valid(lo:hi));
        mv = mv(:) & dyn(:);
        if any(mv)
            pred = apply_axis_mode(pred_raw(mv, :), axisMode);
            src = pts(mv, :);
            predCount = size(pred, 1);
            scatter3(ax, pred(:,1), pred(:,2), pred(:,3), 14, ...
                repmat([0.05, 0.75, 1.0], predCount, 1), 'filled');

            step = max(1, floor(predCount / 250));
            q0 = src(1:step:end, :);
            q1 = pred(1:step:end, :) - q0;
            quiver3(ax, q0(:,1), q0(:,2), q0(:,3), q1(:,1), q1(:,2), q1(:,3), ...
                0, 'Color', [0.9, 0.9, 1.0], 'LineWidth', 0.75);
        end

        if t < nFrames
            nlo = double(S.frame_start_1(t + 1));
            nhi = double(S.frame_end_1(t + 1));
            if nhi >= nlo
                nextDyn = logical(S.dynamic(nlo:nhi));
                if any(nextDyn)
                    nextPts = apply_axis_mode(double(S.points(nlo:nhi, :)), axisMode);
                    nextPts = nextPts(nextDyn, :);
                    stepNext = max(1, floor(size(nextPts, 1) / 1200));
                    nextPts = nextPts(1:stepNext:end, :);
                    scatter3(ax, nextPts(:,1), nextPts(:,2), nextPts(:,3), 10, ...
                        repmat([0.1, 1.0, 0.2], size(nextPts, 1), 1), 'filled');
                end
            end
        end
    end
    hold(ax, 'off');

    title(ax, sprintf('Frame %d / %d | points %d | dynamic %d | predicted %d | axis %s', ...
        t, nFrames, size(pts, 1), nnz(dyn), predCount, char(axisMode)), 'Color', 'w');
    axis(ax, 'equal');
    grid(ax, 'on');
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
