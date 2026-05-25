function view_4d_model_matlab(modelPath, fps, dynamicRed, axisMode)
%VIEW_4D_MODEL_MATLAB Play an exported DROID-W 4D model in MATLAB.
%
% Usage:
%   view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat')
%   view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat', 8, true)
%   view_4d_model_matlab('bonn_balloon/4d_model/model_4d.mat', 8, true, 'swap_yz')
%
% Expected variables in model_4d.mat:
%   points          Nx3 single, world coordinates
%   colors          Nx3 uint8, RGB
%   dynamic         Nx1 logical
%   frame_start_1   Tx1 int, 1-based inclusive start index
%   frame_end_1     Tx1 int, 1-based inclusive end index

if nargin < 2 || isempty(fps)
    fps = 8;
end
if nargin < 3 || isempty(dynamicRed)
    dynamicRed = true;
end
if nargin < 4 || isempty(axisMode)
    axisMode = 'none';
end

S = load(modelPath);
nFrames = numel(S.frame_start_1);

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

    pts = double(S.points(lo:hi, :));
    pts = apply_axis_mode(pts, axisMode);
    rgb = double(S.colors(lo:hi, :)) / 255.0;
    dyn = logical(S.dynamic(lo:hi));

    if dynamicRed
        rgb(dyn, :) = repmat([1.0, 0.05, 0.02], nnz(dyn), 1);
    end

    cla(ax);
    scatter3(ax, pts(:,1), pts(:,2), pts(:,3), 5, rgb, 'filled');
    title(ax, sprintf('Frame %d / %d | points %d | dynamic %d', ...
        t, nFrames, size(pts, 1), nnz(dyn)), 'Color', 'w');
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
