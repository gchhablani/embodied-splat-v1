#!/bin/bash

TASK="imagenav"
MAIN_DATASET="hssd-hab"
DATA_PATH="data/datasets/pointnav/pointnav_hssd/v0.5"
CONFIG_FILE="config/experiments/ddppo_${TASK}_hssd_inr_collision_v2.yaml"

SCENE_DATASET="default"
SIMULATOR_TYPE="CustomSim-v0"

# Path to the sbatch script to be used
SBATCH_SCRIPT_PATH="./scripts/eval_v2/templates/eval_hssd_single_ckpt_single_scene.sh"

SCENES_DIR="data/scene_datasets/${MAIN_DATASET}/"

ckpt_dir="data/new_checkpoints_v2/${TASK}/${MAIN_DATASET}/ddppo_imagenav_high_lr/vc1/exp_inr_collision/steps_1000/seed_100/1200_m"

ckpt_interval=1

num_ckpt_files=$(ls ${ckpt_dir}/*ckpt.*.pth | wc -l)
count=$((num_ckpt_files - 1))
echo "Num ckpt files: $num_ckpt_files, Interval: $ckpt_interval"

uuid="ckpt_${count}"

SCENES_PATH="data/datasets/pointnav/pointnav_hssd/v0.5/val/content"
for scene_file in ${SCENES_PATH}/*.json.gz; do
    SCENE=$(basename "${scene_file}" .json.gz)
    CONTENT_SCENES="['${SCENE}']"
    echo "Scene: $SCENE"
    CHECKPOINT_DIR="${ckpt_dir}/ckpt.${count}.pth"
    CHECKPOINT_ID=${count}

    TENSORBOARD_DIR="tb/${TASK}/${MAIN_DATASET}/ddppo_imagenav_high_lr/vc1/exp_inr_collision/steps_1000/seed_100/1200_m/test_eval_ckpt_${CHECKPOINT_ID}_${SCENE}"

    echo "Ckpt id: $uuid - $count, ${TENSORBOARD_DIR}, ${CHECKPOINT_DIR}"

    existing_job_name=$(grep '^#SBATCH --job-name=' "$SBATCH_SCRIPT_PATH" | cut -d'=' -f2)
    updated_job_name=$(echo "$existing_job_name" | sed "s/\${TASK}/${TASK}/g" | sed "s/\${MAIN_DATASET}/${MAIN_DATASET}/g" | sed "s/\${MESH_TYPE}\///g" | sed "s/\${SCENE_NAME}\///g" | sed "s/\${CKPT_ID}/${CHECKPOINT_ID}/g" | sed "s/\${SCENE}/${SCENE}/g")

    # Submit the script using sbatch with the updated job name and export the variables
    sbatch --gpus a40:1 \
            --nodes 1 \
            --cpus-per-task 10 \
            --ntasks-per-node 1 \
            --signal=USR1@100 \
            --requeue \
            --partition=overcap \
            --export=ALL,TASK=${TASK},MAIN_DATASET=${MAIN_DATASET},MESH_TYPE=${MESH_TYPE},SCENE_NAME=${SCENE_NAME},DATA_PATH=${DATA_PATH},SCENES_DIR=${SCENES_DIR},CONFIG_FILE=${CONFIG_FILE},CONTENT_SCENES=${CONTENT_SCENES},SCENE_DATASET=${SCENE_DATASET},SIMULATOR_TYPE=${SIMULATOR_TYPE},CHECKPOINT_DIR=${CHECKPOINT_DIR},TENSORBOARD_DIR=${TENSORBOARD_DIR} \
            --job-name="$updated_job_name" \
            "$SBATCH_SCRIPT_PATH"

    echo "Submitted $SBATCH_SCRIPT_PATH with job-name $updated_job_name"
done
