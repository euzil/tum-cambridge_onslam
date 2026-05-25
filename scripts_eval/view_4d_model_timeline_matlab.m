function view_4d_model_timeline_matlab(modelPath, dynamicRed, axisMode)
%VIEW_4D_MODEL_TIMELINE_MATLAB Interactive 4D viewer with a frame slider.
%
% Usage:
%   view_4d_model_timeline_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat')
%   view_4d_model_timeline_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', true, 'flip_z')

if nargin < 2 || isempty(dynamicRed)
    dynamicRed = false;
end
if nargin < 3 || isempty(axisMode)
    axisMode = 'none';
end

S = load(modelPath);
nFrames = numel(S.frame_start_1);
if nFrames < 1
    error('No frames in model file.');
end

fig = figure('Name', '4D Timeline Viewer', 'Color', 'k');
ax = axes('Parent', fig, 'Position', [0.06, 0.14, 0.90, 0.82]);
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

    function update_frame(t)
        t = max(1, min(nFrames, t));
        lo = double(S.frame_start_1(t));
        hi = double(S.frame_end_1(t));
        if hi < lo
            return;
        end

        pts = double(S.points(lo:hi, :));
        pts = apply_axis_mode(pts, axisMode);
        rgb = double(S.colors(lo:hi, :)) / 255.0;

        if isfield(S, 'dynamic')
            dyn = logical(S.dynamic(lo:hi));
        else
            dyn = false(size(pts, 1), 1);
        end
        if dynamicRed
            rgb(dyn, :) = repmat([1.0, 0.05, 0.02], nnz(dyn), 1);
        end

        cla(ax);
        scatter3(ax, pts(:,1), pts(:,2), pts(:,3), 5, rgb, 'filled');
        title(ax, sprintf('Frame %d / %d | points %d | dynamic %d', ...
            t, nFrames, size(pts, 1), nnz(dyn)), 'Color', 'w');
        axis(ax, 'equal');
        grid(ax, 'on');
        txt.String = sprintf('Frame %d', t);
        slider.Value = t;
        drawnow;
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
