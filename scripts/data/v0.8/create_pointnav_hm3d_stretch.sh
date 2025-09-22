#!/bin/bash
#SBATCH --job-name=create_pointnav_hm3d_stretch
#SBATCH --output=slurm_logs/data/v0.8/%x-%j.out
#SBATCH --error=slurm_logs/data/v0.8/%x-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --partition=overcap

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

conda activate embodied_splat

srun python -u mesh_nav/utils/create_pointnav_stretch.py \
    --scene_dataset hm3d \
    --episode_data_path "./data/datasets/pointnav/pointnav_hm3d_stretch/v0.8" \
    --scene_files_path "./data/scene_datasets/hm3d_semantic_v0.2" \
    --force_recreate_navmesh \
    --config_file "config/experiments/ddppo_imagenav_inr_collision_v2.yaml" \
