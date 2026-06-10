function stats = build_global_slam_from_4d_matlab(modelPath, outDir, voxelSize, minObs, maxColorStd, axisMode, includeDynamic)
%BUILD_GLOBAL_SLAM_FROM_4D_MATLAB Build a global static SLAM map from model_4d.mat.
%
% Outputs:
%   - global_static_map.ply       (global static point cloud)
%   - global_static_map.mat       (points/colors/obs)
%   - trajectory_tum.txt          (camera trajectory: timestamp tx ty tz qx qy qz qw)
%   - global_dynamic_points.mat   (optional; dynamic points + frame ids)
%   - global_scene_static_dynamic.ply (optional; static + dynamic overlay)
%
% Usage:
%   stats = build_global_slam_from_4d_matlab('Outputs/Bonn/bonn_balloon/4d_model/model_4d.mat', ...
%       'Outputs/Bonn/bonn_balloon/slam_global');
%
%   stats = build_global_slam_from_4d_matlab('.../model_4d.mat', '.../slam_global', 0.02, 3, 0.20, 'flip_z');

if nargin < 3 || isempty(voxelSize)
    voxelSize = 0.02;
end
if nargin < 4 || isempty(minObs)
    minObs = 3;
end
if nargin < 5 || isempty(maxColorStd)
    maxColorStd = 0.20;
end
if nargin < 6 || isempty(axisMode)
    axisMode = 'none';
end
if nargin < 7 || isempty(includeDynamic)
    includeDynamic = true;
end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

S = load(modelPath);
nFrames = numel(S.frame_start_1);

pts = double(S.points);
rgb = double(S.colors) / 255.0;

if isfield(S, 'dynamic')
    dyn = logical(S.dynamic(:));
else
    dyn = false(size(pts, 1), 1);
end

% Build frame ids for temporal observation counting.
fid = zeros(size(pts, 1), 1);
for t = 1:nFrames
    lo = double(S.frame_start_1(t));
    hi = double(S.frame_end_1(t));
    if hi >= lo
        fid(lo:hi) = t;
    end
end

% Keep static points only.
keep = ~dyn;
pts = pts(keep, :);
rgb = rgb(keep, :);
fid = fid(keep);

% Apply axis transform before fusion so map+trajectory are consistent.
pts = apply_axis_mode(pts, axisMode);

% Voxel fuse + temporal/color stability filtering.
[mapPts, mapRgb, mapObs, mapFirstFid] = fuse_static_global(pts, rgb, fid, voxelSize, minObs, maxColorStd);

% Save MAT.
save(fullfile(outDir, 'global_static_map.mat'), 'mapPts', 'mapRgb', 'mapObs', 'mapFirstFid', ...
    'voxelSize', 'minObs', 'maxColorStd', 'axisMode');

% Save PLY.
write_ply_xyzrgb(fullfile(outDir, 'global_static_map.ply'), mapPts, mapRgb);

% Optional dynamic export in global frame.
dynPts = zeros(0, 3);
dynRgb = zeros(0, 3);
dynFid = zeros(0, 1);
if includeDynamic
    keepDyn = dyn;
    dynPts = double(S.points(keepDyn, :));
    dynPts = apply_axis_mode(dynPts, axisMode);
    dynRgb = double(S.colors(keepDyn, :)) / 255.0;
    dynFid = fid_all_from_ranges(S.frame_start_1, S.frame_end_1, size(S.points, 1));
    dynFid = dynFid(keepDyn);
    save(fullfile(outDir, 'global_dynamic_points.mat'), 'dynPts', 'dynRgb', 'dynFid', 'axisMode');

    % Downsample dynamic overlay for lighter global visualization.
    [dynPtsSmall, dynRgbSmall] = voxel_downsample_xyzrgb(dynPts, dynRgb, max(voxelSize, 0.03));
    mixPts = [mapPts; dynPtsSmall];
    mixRgb = [mapRgb; repmat([1.0, 0.08, 0.03], size(dynPtsSmall, 1), 1)];
    write_ply_xyzrgb(fullfile(outDir, 'global_scene_static_dynamic.ply'), mixPts, mixRgb);
end

% Save trajectory in TUM format if poses exist.
if isfield(S, 'poses_c2w') && ~isempty(S.poses_c2w)
    poses = double(S.poses_c2w);
    poses = apply_axis_mode_to_poses(poses, axisMode);

    if isfield(S, 'timestamps') && numel(S.timestamps) == size(poses, 1)
        ts = double(S.timestamps(:));
    else
        ts = (0:size(poses, 1)-1).';
    end

    traj = zeros(size(poses, 1), 8);
    traj(:, 1) = ts;
    for i = 1:size(poses, 1)
        T = poses(i, :, :);
        T = squeeze(T);
        R = T(1:3, 1:3);
        t = T(1:3, 4);
        q = rotm2quat_local(R); % [qw qx qy qz]
        traj(i, :) = [ts(i), t(:).', q(2), q(3), q(4), q(1)];
    end
    writematrix(traj, fullfile(outDir, 'trajectory_tum.txt'), 'Delimiter', 'space');
end

stats = struct();
stats.n_frames = nFrames;
stats.n_static_input = size(pts, 1);
stats.n_global_map_points = size(mapPts, 1);
stats.voxel_size = voxelSize;
stats.min_obs = minObs;
stats.max_color_std = maxColorStd;
stats.axis_mode = axisMode;
stats.include_dynamic = includeDynamic;
stats.n_dynamic_points = size(dynPts, 1);

fprintf('[global-slam] frames=%d static_input=%d map_points=%d\n', ...
    stats.n_frames, stats.n_static_input, stats.n_global_map_points);
if includeDynamic
    fprintf('[global-slam] dynamic_points=%d\n', stats.n_dynamic_points);
end
fprintf('[global-slam] saved: %s\n', outDir);
end

function fid = fid_all_from_ranges(frameStart, frameEnd, nPoints)
fid = zeros(nPoints, 1);
nFrames = numel(frameStart);
for t = 1:nFrames
    lo = double(frameStart(t));
    hi = double(frameEnd(t));
    if hi >= lo
        fid(lo:hi) = t;
    end
end
end

function [mapPts, mapRgb, mapObs, mapFirstFid] = fuse_static_global(pts, rgb, fid, voxel, minObs, maxColorStd)
ijk = floor(pts ./ voxel);
[~, ~, g] = unique(ijk, 'rows');
n = max(g);

cnt = accumarray(g, 1, [n, 1]);
sumP = zeros(n, 3);
sumC = zeros(n, 3);
sumC2 = zeros(n, 3);

for d = 1:3
    sumP(:, d) = accumarray(g, pts(:, d), [n, 1]);
    sumC(:, d) = accumarray(g, rgb(:, d), [n, 1]);
    sumC2(:, d) = accumarray(g, rgb(:, d).^2, [n, 1]);
end

meanP = sumP ./ cnt;
meanC = sumC ./ cnt;
varC = max(sumC2 ./ cnt - meanC.^2, 0);
stdC = sqrt(varC);
stdMean = mean(stdC, 2);

pair = unique([g, fid], 'rows');
obs = accumarray(pair(:, 1), 1, [n, 1]);
firstFid = accumarray(g, fid, [n, 1], @min, 0);

stable = obs >= max(1, minObs);
if maxColorStd > 0
    stable = stable & (stdMean <= maxColorStd);
end

mapPts = meanP(stable, :);
mapRgb = meanC(stable, :);
mapObs = obs(stable);
mapFirstFid = firstFid(stable);
end

function [ptsOut, rgbOut] = voxel_downsample_xyzrgb(pts, rgb, voxel)
if isempty(pts)
    ptsOut = pts;
    rgbOut = rgb;
    return;
end
ijk = floor(pts ./ voxel);
[~, ~, g] = unique(ijk, 'rows');
n = max(g);
cnt = accumarray(g, 1, [n, 1]);
sumP = zeros(n, 3);
sumC = zeros(n, 3);
for d = 1:3
    sumP(:, d) = accumarray(g, pts(:, d), [n, 1]);
    sumC(:, d) = accumarray(g, rgb(:, d), [n, 1]);
end
ptsOut = sumP ./ cnt;
rgbOut = sumC ./ cnt;
end

function write_ply_xyzrgb(path, pts, rgb)
rgb8 = uint8(max(0, min(255, round(rgb * 255))));
fid = fopen(path, 'w');
if fid < 0
    error('Failed to open PLY for writing: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'ply\nformat ascii 1.0\n');
fprintf(fid, 'element vertex %d\n', size(pts, 1));
fprintf(fid, 'property float x\nproperty float y\nproperty float z\n');
fprintf(fid, 'property uchar red\nproperty uchar green\nproperty uchar blue\n');
fprintf(fid, 'end_header\n');
for i = 1:size(pts, 1)
    fprintf(fid, '%.6f %.6f %.6f %d %d %d\n', ...
        pts(i, 1), pts(i, 2), pts(i, 3), ...
        rgb8(i, 1), rgb8(i, 2), rgb8(i, 3));
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

function posesOut = apply_axis_mode_to_poses(posesIn, mode)
posesOut = posesIn;
A = eye(4);
switch lower(string(mode))
    case "none"
        return;
    case "swap_xy"
        A(1:3, 1:3) = [0 1 0; 1 0 0; 0 0 1];
    case "swap_xz"
        A(1:3, 1:3) = [0 0 1; 0 1 0; 1 0 0];
    case "swap_yz"
        A(1:3, 1:3) = [1 0 0; 0 0 1; 0 1 0];
    case "flip_x"
        A(1, 1) = -1;
    case "flip_y"
        A(2, 2) = -1;
    case "flip_z"
        A(3, 3) = -1;
    case "flip_xy"
        A(1, 1) = -1; A(2, 2) = -1;
    case "flip_xz"
        A(1, 1) = -1; A(3, 3) = -1;
    case "flip_yz"
        A(2, 2) = -1; A(3, 3) = -1;
    case "flip_xyz"
        A(1, 1) = -1; A(2, 2) = -1; A(3, 3) = -1;
    otherwise
        error('Unknown axisMode: %s', mode);
end

for i = 1:size(posesIn, 1)
    T = squeeze(posesIn(i, :, :));
    posesOut(i, :, :) = A * T * A;
end
end

function q = rotm2quat_local(R)
% Return [qw qx qy qz]
tr = trace(R);
if tr > 0
    S = sqrt(tr + 1.0) * 2;
    qw = 0.25 * S;
    qx = (R(3,2) - R(2,3)) / S;
    qy = (R(1,3) - R(3,1)) / S;
    qz = (R(2,1) - R(1,2)) / S;
elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
    S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2;
    qw = (R(3,2) - R(2,3)) / S;
    qx = 0.25 * S;
    qy = (R(1,2) + R(2,1)) / S;
    qz = (R(1,3) + R(3,1)) / S;
elseif R(2,2) > R(3,3)
    S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2;
    qw = (R(1,3) - R(3,1)) / S;
    qx = (R(1,2) + R(2,1)) / S;
    qy = 0.25 * S;
    qz = (R(2,3) + R(3,2)) / S;
else
    S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2;
    qw = (R(2,1) - R(1,2)) / S;
    qx = (R(1,3) + R(3,1)) / S;
    qy = (R(2,3) + R(3,2)) / S;
    qz = 0.25 * S;
end
q = [qw, qx, qy, qz];
q = q / norm(q + eps);
end
