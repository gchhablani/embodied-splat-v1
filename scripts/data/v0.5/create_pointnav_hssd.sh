#!/bin/bash
#SBATCH --job-name=create_pointnav_hssd
#SBATCH --output=slurm_logs/data/v0.5/%x-%j.out
#SBATCH --error=slurm_logs/data/v0.5/%x-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --exclude=robby,chappie,voltron
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --partition=overcap

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/flash1/gchhablani3/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate mesh_nav_gauss

srun python -u mesh_nav/utils/create_pointnav_stretch.py \
    --scene_dataset hssd \
    --episode_data_path "./data/datasets/pointnav/pointnav_hssd/v0.5" \
    --scene_files_path "./data/scene_datasets/hssd-hab/stages/" \
    --scene_dataset_config_file "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json" \
    --force_recreate_navmesh \
    --config_file "config/experiments/ddppo_imagenav_inr_collision_v2.yaml" \
