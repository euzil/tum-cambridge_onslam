function [pts, rgb] = view_static_reconstruction_matlab(modelPath, pointSize, savePlyPath)
%VIEW_STATIC_RECONSTRUCTION_MATLAB Show dense static reconstruction exported from video.npz.
%
% Usage:
%   view_static_reconstruction_matlab('Outputs/Bonn/bonn_balloon/static_3d/static_reconstruction.mat')
%   view_static_reconstruction_matlab('.../static_reconstruction.mat', 4, 'static_reconstruction_matlab.ply')
%
% Expected variables:
%   points      Nx3 single world coordinates
%   colors      Nx3 uint8 RGB, or colors01 Nx3 single colors in [0,1]

if nargin < 2 || isempty(pointSize)
    pointSize = 4;
end
if nargin < 3
    savePlyPath = '';
end

S = load(modelPath);
pts = double(S.points);
if isfield(S, 'colors01')
    rgb = double(S.colors01);
else
    rgb = double(S.colors) / 255.0;
end
rgb = min(max(rgb, 0), 1);

fprintf('[static-view] points: %d\n', size(pts, 1));

if ~isempty(savePlyPath)
    write_ply(savePlyPath, pts, rgb);
    fprintf('[static-view] wrote PLY: %s\n', savePlyPath);
end

fig = figure('Name', 'Static 3D Reconstruction', 'Color', 'k');
ax = axes(fig);
scatter3(ax, pts(:,1), pts(:,2), pts(:,3), pointSize, rgb, 'filled');
axis(ax, 'equal');
grid(ax, 'on');
view(ax, 3);
xlabel(ax, 'X');
ylabel(ax, 'Y');
zlabel(ax, 'Z');
set(ax, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
title(ax, sprintf('Static 3D reconstruction | points=%d', size(pts, 1)), 'Color', 'w');

if isfield(S, 'poses_c2w') && ~isempty(S.poses_c2w)
    hold(ax, 'on');
    poses = double(S.poses_c2w);
    camPts = squeeze(poses(:, 1:3, 4));
    plot3(ax, camPts(:,1), camPts(:,2), camPts(:,3), ...
        '-', 'Color', [0.1 0.7 1.0], 'LineWidth', 1.5);
    scatter3(ax, camPts(:,1), camPts(:,2), camPts(:,3), ...
        12, repmat([0.1 0.7 1.0], size(camPts, 1), 1), 'filled');
    hold(ax, 'off');
end
end

function write_ply(path, pts, rgb)
rgb8 = uint8(round(min(max(rgb, 0), 1) * 255));
fid = fopen(path, 'w');
if fid < 0
    error('Cannot open %s for writing.', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'ply\n');
fprintf(fid, 'format ascii 1.0\n');
fprintf(fid, 'element vertex %d\n', size(pts, 1));
fprintf(fid, 'property float x\n');
fprintf(fid, 'property float y\n');
fprintf(fid, 'property float z\n');
fprintf(fid, 'property uchar red\n');
fprintf(fid, 'property uchar green\n');
fprintf(fid, 'property uchar blue\n');
fprintf(fid, 'end_header\n');
fprintf(fid, '%.6f %.6f %.6f %d %d %d\n', [pts, double(rgb8)]');
end
