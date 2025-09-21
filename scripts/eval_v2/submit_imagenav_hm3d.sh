#!/bin/bash

TASK="imagenav"
MAIN_DATASET="hm3d_semantic_v0.2"
DATA_PATH="data/datasets/pointnav/pointnav_hm3d_stretch/v0.8"
CONFIG_FILE="config/experiments/ddppo_${TASK}_inr_collision_v2.yaml"

SCENE_DATASET="default"
SIMULATOR_TYPE="CustomSim-v0"

# Path to the sbatch script to be used
SBATCH_SCRIPT_PATH="./scripts/eval_v2/templates/eval_hm3d.sh"

SCENES_DIR="data/scene_datasets/${MAIN_DATASET}/"

CONTENT_SCENES="['*']"

existing_job_name=$(grep '^#SBATCH --job-name=' "$SBATCH_SCRIPT_PATH" | cut -d'=' -f2)

updated_job_name=$(echo "$existing_job_name" | sed "s/\${TASK}/${TASK}/g" | sed "s/\${MAIN_DATASET}/${MAIN_DATASET}/g" | sed "s/\${MESH_TYPE}\///g" | sed "s/\${SCENE_NAME}\///g")

# Submit the script using sbatch with the updated job name and export the variables
sbatch --gpus a40:1 \
    --nodes 1 \
    --cpus-per-task 10 \
    --ntasks-per-node 1 \
    --exclude=${EXCLUDED_NODES} \
    --signal=USR1@100 \
    --requeue \
    --partition=overcap \
    --export=ALL,TASK=${TASK},MAIN_DATASET=${MAIN_DATASET},MESH_TYPE=${MESH_TYPE},SCENE_NAME=${SCENE_NAME},DATA_PATH=${DATA_PATH},SCENES_DIR=${SCENES_DIR},CONFIG_FILE=${CONFIG_FILE},CONTENT_SCENES=${CONTENT_SCENES},SCENE_DATASET=${SCENE_DATASET},SIMULATOR_TYPE=${SIMULATOR_TYPE} \
    --job-name="$updated_job_name" \
    "$SBATCH_SCRIPT_PATH"

echo "Submitted $SBATCH_SCRIPT_PATH with job-name $updated_job_name"
