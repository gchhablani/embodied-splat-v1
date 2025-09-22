#!/bin/bash
#SBATCH --job-name=train_v2/v0.5/${TASK}/${MAIN_DATASET}/${MESH_TYPE}/${SCENE_NAME}/ddppo_imagenav_high_lr/vc1/exp_inr_collision/steps_1000/seed_100
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --gpus a40:16
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 16
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --partition=kira-lab
#SBATCH --qos=short

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
# export HABITAT_ENV_DEBUG=1

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

conda activate embodied_splat


WB_ENTITY="user"
PROJECT_NAME="embodied_splat"

if [[ "${SCENE_NAME}" == "null" && "${MESH_TYPE}" == "null" ]]; then  
  TENSORBOARD_DIR="tb/${TASK}/${MAIN_DATASET}/ddppo_imagenav_high_lr/vc1/exp_inr_collision/steps_1000/seed_100/1200_m"
  CHECKPOINT_DIR="data/new_checkpoints_v2/${TASK}/${MAIN_DATASET}/ddppo_imagenav_high_lr/vc1/exp_inr_collision/steps_1000/seed_100/1200_m"
else
  TENSORBOARD_DIR="tb/${TASK}/${MAIN_DATASET}/${MESH_TYPE}/${SCENE_NAME}/ddppo_imagenav_high_lr/vc1/exp_inr_collision/steps_1000/seed_100/1200_m"
  CHECKPOINT_DIR="data/new_checkpoints_v2/${TASK}/${MAIN_DATASET}/${MESH_TYPE}/${SCENE_NAME}/ddppo_imagenav_high_lr/vc1/exp_inr_collision/steps_1000/seed_100/1200_m"
fi

srun python -um embodied_splat.run \
  --run-type train \
  --exp-config ${CONFIG_FILE} \
  habitat_baselines.trainer_name="ddppo" \
  habitat_baselines.num_environments=10 \
  habitat_baselines.num_updates=-1 \
  habitat.simulator.type=${SIMULATOR_TYPE} \
  habitat.simulator.scene_dataset=${SCENE_DATASET} \
  habitat_baselines.total_num_steps=${TOTAL_NUM_STEPS} \
  habitat.seed=100 \
  habitat.environment.max_episode_steps=1000 \
  habitat_baselines.num_checkpoints=${NUM_CHECKPOINTS} \
  habitat.dataset.scenes_dir=${SCENES_DIR} \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.dataset.content_scenes=${CONTENT_SCENES} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  habitat_baselines.writer_type=wb \
  habitat_baselines.wb.entity=$WB_ENTITY \
  habitat_baselines.wb.run_name=$TENSORBOARD_DIR \
  habitat_baselines.wb.project_name=$PROJECT_NAME \
  habitat_baselines.updater_name=ImageNavPPO \
  habitat_baselines.distrib_updater_name=DistributedImageNavPPO \
  habitat_baselines.rl.ppo.optimizer_name="adamw" \
  habitat_baselines.rl.ppo.adamw_weight_decay=1e-6 \
  habitat_baselines.rl.ppo.encoder_lr=1.5e-5 \