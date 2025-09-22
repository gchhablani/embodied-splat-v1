#!/bin/bash

# Set the static variables
TASK="imagenav"
MAIN_DATASET="polycam_data"
DATA_PATH="data/datasets/pointnav/pointnav_hssd/v0.5"
CONFIG_FILE="config/experiments/finetune_ddppo_${TASK}_inr_collision_v2.yaml"

# Set the lists for MESH_TYPE and SCENE_NAME
MESH_TYPES=("polycam_mesh")
SCENE_NAMES=("castleberry" "piedmont" "clough_classroom" "grad_lounge")
SCENE_DATASET="default"
SIMULATOR_TYPE="CustomSim-v0"

# Path to the sbatch script to be used
SBATCH_SCRIPT_PATH="./scripts/fine_train_eval_v2/templates/eval_hssd_on_hssd.sh"

for MESH_TYPE in "${MESH_TYPES[@]}"; do
    for SCENE_NAME in "${SCENE_NAMES[@]}"; do
        SCENES_DIR="data/scene_datasets/hssd-hab/"
        CONTENT_SCENES="['*']"
        # Extract the existing job name from the sbatch script
        existing_job_name=$(grep '^#SBATCH --job-name=' "$SBATCH_SCRIPT_PATH" | cut -d'=' -f2)

        updated_job_name=$(echo "$existing_job_name" | sed "s/\${TASK}/${TASK}/g" | sed "s/\${MAIN_DATASET}/${MAIN_DATASET}/g" | sed "s/\${MESH_TYPE}/${MESH_TYPE}/g" | sed "s/\${SCENE_NAME}/${SCENE_NAME}/g")

        # Submit the script using sbatch with the updated job name and export the variables
        sbatch --gpus a40:1 \
            --nodes 1 \
            --cpus-per-task 10 \
            --ntasks-per-node 1 \
            --signal=USR1@100 \
            --requeue \
            --partition=overcap \
            --export=ALL,TASK=${TASK},MAIN_DATASET=${MAIN_DATASET},MESH_TYPE=${MESH_TYPE},SCENE_NAME=${SCENE_NAME},DATA_PATH=${DATA_PATH},SCENES_DIR=${SCENES_DIR},CONFIG_FILE=${CONFIG_FILE},CONTENT_SCENES=${CONTENT_SCENES},SCENE_DATASET=${SCENE_DATASET},SIMULATOR_TYPE=${SIMULATOR_TYPE} \
            --job-name="$updated_job_name" \
            "$SBATCH_SCRIPT_PATH"

        echo "Submitted $SBATCH_SCRIPT_PATH with job-name $updated_job_name"
    done
done

echo "All sbatch jobs have been submitted."
