#!/bin/bash

TASK="imagenav"
MAIN_DATASET="hssd-hab"
DATA_PATH="data/datasets/pointnav/pointnav_hssd/v0.5"
CONFIG_FILE="config/experiments/ddppo_${TASK}_hssd_inr_collision_v2.yaml"
TOTAL_NUM_STEPS=12e8
SCENE_NAME=null
MESH_TYPE=null
SCENE_DATASET="default"
SIMULATOR_TYPE="CustomSim-v0"
NUM_CHECKPOINTS=400

SBATCH_SCRIPT_PATH="./scripts/train_v2/templates/train_hssd_high_lr.sh"

SCENES_DIR="data/scene_datasets/${MAIN_DATASET}/"

CONTENT_SCENES="['*']"

existing_job_name=$(grep '^#SBATCH --job-name=' "$SBATCH_SCRIPT_PATH" | cut -d'=' -f2)

updated_job_name=$(echo "$existing_job_name" | sed "s/\${TASK}/${TASK}/g" | sed "s/\${MAIN_DATASET}/${MAIN_DATASET}/g" | sed "s/\${MESH_TYPE}\///g" | sed "s/\${SCENE_NAME}\///g")

sbatch --gpus a40:16 \
    --nodes 2 \
    --cpus-per-task 16 \
    --ntasks-per-node 8 \
    --signal=USR1@100 \
    --requeue \
    --partition=kira-lab \
    --qos=long \
    --export=ALL,TASK=${TASK},MAIN_DATASET=${MAIN_DATASET},MESH_TYPE=${MESH_TYPE},SCENE_NAME=${SCENE_NAME},DATA_PATH=${DATA_PATH},SCENES_DIR=${SCENES_DIR},CONFIG_FILE=${CONFIG_FILE},CONTENT_SCENES=${CONTENT_SCENES},TOTAL_NUM_STEPS=${TOTAL_NUM_STEPS},SCENE_DATASET=${SCENE_DATASET},SIMULATOR_TYPE=${SIMULATOR_TYPE},NUM_CHECKPOINTS=${NUM_CHECKPOINTS} \
    --job-name="$updated_job_name" \
    "$SBATCH_SCRIPT_PATH"

echo "Submitted $SBATCH_SCRIPT_PATH with job-name $updated_job_name"