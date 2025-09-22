#!/bin/bash

TASK="imagenav"
MAIN_DATASET="polycam_data"
DATA_PATH="data/datasets/pointnav/pointnav_polycam_mesh_2/v0.5"
CONFIG_FILE="config/experiments/finetune_ddppo_${TASK}_inr_collision_v2.yaml"
MESH_TYPES=("polycam_mesh_2")
SCENE_NAMES=("coda_conference_room")
TOTAL_NUM_STEPS=2e7
SCENE_DATASET="default"
SIMULATOR_TYPE="CustomSim-v0"
NUM_CHECKPOINTS=21 # 25
EXCLUDED_NODES=$(cat scripts/excluded_nodes.txt)

SBATCH_SCRIPT_PATH="./scripts/fine_train_v2/templates/train.sh"

for MESH_TYPE in "${MESH_TYPES[@]}"; do
    for SCENE_NAME in "${SCENE_NAMES[@]}"; do
        SCENES_DIR="data/scene_datasets/${MAIN_DATASET}/${MESH_TYPE}/"

        CONTENT_SCENES="['${SCENE_NAME}']"
        existing_job_name=$(grep '^#SBATCH --job-name=' "$SBATCH_SCRIPT_PATH" | cut -d'=' -f2)

        updated_job_name=$(echo "$existing_job_name" | sed "s/\${TASK}/${TASK}/g" | sed "s/\${MAIN_DATASET}/${MAIN_DATASET}/g" | sed "s/\${MESH_TYPE}/${MESH_TYPE}/g" | sed "s/\${SCENE_NAME}/${SCENE_NAME}/g")

        sbatch --gpus a40:16 \
            --nodes 2 \
            --cpus-per-task 16 \
            --ntasks-per-node 8 \
            --exclude=${EXCLUDED_NODES} \
            --signal=USR1@100 \
            --requeue \
            --partition=kira-lab \
            --qos=short \
            --export=ALL,TASK=${TASK},MAIN_DATASET=${MAIN_DATASET},MESH_TYPE=${MESH_TYPE},SCENE_NAME=${SCENE_NAME},DATA_PATH=${DATA_PATH},SCENES_DIR=${SCENES_DIR},CONFIG_FILE=${CONFIG_FILE},CONTENT_SCENES=${CONTENT_SCENES},TOTAL_NUM_STEPS=${TOTAL_NUM_STEPS},SCENE_DATASET=${SCENE_DATASET},SIMULATOR_TYPE=${SIMULATOR_TYPE},NUM_CHECKPOINTS=${NUM_CHECKPOINTS} \
            --job-name="$updated_job_name" \
            "$SBATCH_SCRIPT_PATH"

        echo "Submitted $SBATCH_SCRIPT_PATH with job-name $updated_job_name"
    done
done

echo "All sbatch jobs have been submitted."
