export TORCH_CUDA_ARCH_LIST="8.9"
export CC=gcc-11
export CXX=g++-11
export CUDA_HOME=/home/youran/miniconda3/envs/droid-w


view_static_env_dynamic_timeline_matlab( ...
    'outputs/Bonn/bonn_person_tracking/slam_global_learned_region', ...
    2, ...
    4, ...
    'outputs/Bonn/bonn_person_tracking/4d_model_learned_motion_region/model_4d.mat', ...
    6);

view_static_env_dynamic_timeline_matlab( ...
    'output/Bonn/bonn_person_tracking/slam_global', ...
    2, ...
    4, ...
    'output/Bonn/bonn_person_tracking/4d_model_previous_motion/model_4d.mat', ...
    6);


    conda activate d4rt

python scripts_d4rt/build_d4rt_slam_cache.py \
  --slam-config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml \
  --opend4rt-root Open-d4rt \
  --model-config Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml \
  --ckpt-path Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --output output/d4rt_cache/freiburg3_walking_xyz_smoke16.npz \
  --device cuda \
  --max-frames 16 \
  --query-chunk-size 1024 \
  --source-batch-size 1

python scripts_d4rt/build_d4rt_slam_cache.py \
  --slam-config configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml \
  --opend4rt-root Open-d4rt \
  --model-config Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --ckpt-path Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --output output/d4rt_cache/freiburg3_walking_xyz_smoke16.npz \
  --device cuda \
  --max-frames 16 \
  --query-chunk-size 1024 \
  --source-batch-size 1


python scripts_d4rt/build_d4rt_slam_cache.py 
  --slam-config configs/Dynamic/Bonn/bonn_balloon.yaml 
  --opend4rt-root Open-d4rt 
  --model-config Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml 
  --ckpt-path Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt 
  --output output/d4rt_cache/bonn_balloon_smoke16_g4.npz 
  --device cuda 
  --max-frames 16 
  --grid-stride 4 
  --query-chunk-size 128 
  --source-batch-size 1


python run.py --config configs/Dynamic/Bonn/bonn_balloon_d4rt_smoke.yaml

python scripts_eval/evaluate_pose_paper_metrics.py \
  --est Outputs/Bonn/bonn_person_tracking2_d4rt_smoke/traj/est_poses_full.txt \
  --gt datasets/Bonn/rgbd_bonn_person_tracking2/groundtruth.txt \
  --associate-by-index \
  --out-prefix eval/AB_2/frame_8/bonn_person_tracking2/bonn_person_tracking2

