function view_object_4d_model_matlab(modelPath, fps, showPredicted)
%VIEW_OBJECT_4D_MODEL_MATLAB Play object-level dynamic 4D model in MATLAB.
%
% Usage:
%   view_object_4d_model_matlab('Outputs/Bonn/bonn_balloon/object_4d_model/object_4d_model.mat')
%   view_object_4d_model_matlab('Outputs/Bonn/bonn_balloon/object_4d_model/object_4d_model.mat', 8, true)

if nargin < 2 || isempty(fps)
    fps = 8;
end
if nargin < 3 || isempty(showPredicted)
    showPredicted = true;
end

S = load(modelPath);
nFrames = numel(S.frame_start_1);
objectIds = unique(double(S.object_ids(:)));
objectIds(objectIds <= 0) = [];
palette = lines(max(numel(objectIds), 1));

fig = figure('Name', 'Object-level DROID-W 4D model', 'Color', 'k');
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
    pred = double(S.predicted_next_points(lo:hi, :));
    ids = double(S.object_ids(lo:hi));

    rgb = zeros(numel(ids), 3);
    for k = 1:numel(ids)
        idx = find(objectIds == ids(k), 1);
        if isempty(idx)
            rgb(k, :) = [0.8, 0.8, 0.8];
        else
            rgb(k, :) = palette(idx, :);
        end
    end

    cla(ax);
    scatter3(ax, pts(:,1), pts(:,2), pts(:,3), 8, rgb, 'filled');
    hold(ax, 'on');
    if showPredicted
        step = max(1, floor(size(pts, 1) / 250));
        q0 = pts(1:step:end, :);
        q1 = pred(1:step:end, :) - q0;
        quiver3(ax, q0(:,1), q0(:,2), q0(:,3), q1(:,1), q1(:,2), q1(:,3), ...
            0, 'Color', [1.0, 1.0, 1.0], 'LineWidth', 0.75);
    end
    hold(ax, 'off');

    title(ax, sprintf('Frame %d / %d | points %d | objects %d', ...
        t, nFrames, size(pts, 1), numel(unique(ids))), 'Color', 'w');
    axis(ax, 'equal');
    grid(ax, 'on');
    drawnow;
    pause(delay);
end
end
