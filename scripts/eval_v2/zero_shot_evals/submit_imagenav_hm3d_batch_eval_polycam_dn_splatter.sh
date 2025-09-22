#!/bin/bash

TASK="imagenav"
MAIN_DATASET="hm3d_semantic_v0.2"
CONFIG_FILE="config/experiments/ddppo_${TASK}_inr_collision_v2.yaml"
SIMULATOR_TYPE="CustomSim-v0"

SBATCH_SCRIPT_PATH="./scripts/eval_v2/templates/eval_hm3d_single_ckpt.sh"

SCENES_DIR="data/scene_datasets/polycam_data/dn_splatter/"
DATA_PATH="data/datasets/pointnav/pointnav_dn_splatter/v0.5"
SCENE_DATASET="data/scene_datasets/polycam_data/dn_splatter/polycam_data_dn_splatter.scene_dataset_config.json"
CONTENT_SCENES="['*']"

ckpt_dir="data/new_checkpoints_v2/${TASK}/${MAIN_DATASET}/ddppo_imagenav/vc1/exp_inr_collision/steps_1000/seed_100/1200_m"


ckpt_interval=20
count=0

num_ckpt_files=$(ls ${ckpt_dir}/*ckpt.*.pth | wc -l)
echo "Num ckpt files: $num_ckpt_files, Interval: $ckpt_interval"

for (( i=$count; i<$num_ckpt_files; i+=$ckpt_interval ));
do
    uuid="ckpt_${count}"

    CHECKPOINT_DIR="${ckpt_dir}/ckpt.${i}.pth"
    CHECKPOINT_ID=${i}
    TENSORBOARD_DIR="tb/${TASK}/${MAIN_DATASET}/ddppo_imagenav/vc1/exp_inr_collision/steps_1000/seed_100/1200_m/dn_splatter/ablation_eval_ckpt_${CHECKPOINT_ID}"

    # echo "Ckpt id: $uuid - $i, ${tensorboard_dir}, ${current_ckpt_dir}"
    echo "Ckpt id: $uuid - $i, ${TENSORBOARD_DIR}, ${CHECKPOINT_DIR}"

    existing_job_name=$(grep '^#SBATCH --job-name=' "$SBATCH_SCRIPT_PATH" | cut -d'=' -f2)
    updated_job_name=$(echo "$existing_job_name" | sed "s/\${TASK}/${TASK}/g" | sed "s/\${MAIN_DATASET}/${MAIN_DATASET}/g" | sed "s/\${MESH_TYPE}\///g" | sed "s/\${SCENE_NAME}\///g" | sed "s/\${CKPT_ID}/${CHECKPOINT_ID}/g")

    # Submit the script using sbatch with the updated job name and export the variables
    sbatch --gpus a40:1 \
            --nodes 1 \
            --cpus-per-task 10 \
            --ntasks-per-node 1 \
            --signal=USR1@100 \
            --requeue \
            --partition=kira-lab \
            --qos=debug \
            --export=ALL,TASK=${TASK},MAIN_DATASET=${MAIN_DATASET},MESH_TYPE=${MESH_TYPE},SCENE_NAME=${SCENE_NAME},DATA_PATH=${DATA_PATH},SCENES_DIR=${SCENES_DIR},CONFIG_FILE=${CONFIG_FILE},CONTENT_SCENES=${CONTENT_SCENES},SCENE_DATASET=${SCENE_DATASET},SIMULATOR_TYPE=${SIMULATOR_TYPE},CHECKPOINT_DIR=${CHECKPOINT_DIR},TENSORBOARD_DIR=${TENSORBOARD_DIR},NUM_ENVIRONMENTS=1 \
            --job-name="$updated_job_name" \
            "$SBATCH_SCRIPT_PATH"

    echo "Submitted $SBATCH_SCRIPT_PATH with job-name $updated_job_name"

    count=$((count + $ckpt_interval))
done
