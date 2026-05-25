function view_static_map_matlab(modelPath, pointSize, axisMode)
%VIEW_STATIC_MAP_MATLAB Visualize fused static map exported by export_4d_model.py.
%
% Usage:
%   view_static_map_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat')
%   view_static_map_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', 6)
%   view_static_map_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', 6, 'swap_yz')

if nargin < 2 || isempty(pointSize)
    pointSize = 5;
end
if nargin < 3 || isempty(axisMode)
    axisMode = 'none';
end

S = load(modelPath);
if ~isfield(S, 'static_map_points') || isempty(S.static_map_points)
    error(['No static_map_points in this model. Re-export with updated ', ...
        'scripts_eval/export_4d_model.py.']);
end

pts = double(S.static_map_points);
pts = apply_axis_mode(pts, axisMode);
rgb = double(S.static_map_colors) / 255.0;

fig = figure('Name', 'Fused Static Map', 'Color', 'k');
ax = axes(fig);
scatter3(ax, pts(:,1), pts(:,2), pts(:,3), pointSize, rgb, 'filled');
axis(ax, 'equal');
grid(ax, 'on');
view(ax, 3);
xlabel(ax, 'X');
ylabel(ax, 'Y');
zlabel(ax, 'Z');
set(ax, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

if isfield(S, 'static_map_obs')
    obs = double(S.static_map_obs(:));
    title(ax, sprintf('Static Map | points=%d | mean obs=%.2f', ...
        size(pts, 1), mean(obs)), 'Color', 'w');
else
    title(ax, sprintf('Static Map | points=%d', size(pts, 1)), 'Color', 'w');
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
