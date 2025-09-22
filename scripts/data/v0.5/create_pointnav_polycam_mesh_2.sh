#!/bin/bash
#SBATCH --job-name=create_pointnav_polycam_mesh_2
#SBATCH --output=slurm_logs/data/v0.5/%x-%j.out
#SBATCH --error=slurm_logs/data/v0.5/%x-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --partition=kira-lab
#SBATCH --qos=short

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

conda activate embodied_splat

srun python -u mesh_nav/utils/create_pointnav_stretch.py \
    --is_gen_shortest_path \
    --episode_data_path "./data/datasets/pointnav/pointnav_polycam_mesh_2/v0.5" \
    --scene_files_path "./data/scene_datasets/polycam_data/polycam_mesh_2" \
    --force_recreate_navmesh \
    --config_file "config/experiments/ddppo_imagenav_inr_collision_v2.yaml" \