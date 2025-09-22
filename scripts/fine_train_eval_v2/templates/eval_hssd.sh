#!/bin/bash
#SBATCH --job-name=fine_train_eval_v2/v0.5/${TASK}/${MAIN_DATASET}/${MESH_TYPE}/${SCENE_NAME}/ddppo_imagenav/vc1/exp_inr_collision/steps_1000/seed_100
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --gpus a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --exclude=xaea-12,nestor,shakey,dave,megabot,omgwth
#SBATCH --signal=USR1@100
#SBATCH --requeue
#SBATCH --partition=kira-lab
#SBATCH --qos=short

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

conda activate embodied_splat

WB_ENTITY="gchhablani3-gt"
PROJECT_NAME="3dgs"


if [[ "${SCENE_NAME}" == "null" && "${MESH_TYPE}" == "null" ]]; then  
  TENSORBOARD_DIR="tb/finetune_hssd_ckpt_332/${TASK}/${MAIN_DATASET}/ddppo_imagenav/vc1/exp_inr_collision/steps_1000/seed_100/20_m/eval"
  CHECKPOINT_DIR="data/new_checkpoints_v2/finetune_hssd_ckpt_332/${TASK}/${MAIN_DATASET}/ddppo_imagenav/vc1/exp_inr_collision/steps_1000/seed_100/20_m"
else
  TENSORBOARD_DIR="tb/finetune_hssd_ckpt_332/${TASK}/${MAIN_DATASET}/${MESH_TYPE}/${SCENE_NAME}/ddppo_imagenav/vc1/exp_inr_collision/steps_1000/seed_100/20_m/eval"
  CHECKPOINT_DIR="data/new_checkpoints_v2/finetune_hssd_ckpt_332/${TASK}/${MAIN_DATASET}/${MESH_TYPE}/${SCENE_NAME}/ddppo_imagenav/vc1/exp_inr_collision/steps_1000/seed_100/20_m"
fi

NUM_ENVIRONMENTS=1

srun python -um embodied_splat.run \
  --run-type eval \
  --exp-config ${CONFIG_FILE} \
  habitat_baselines.trainer_name="finetune_ddppo" \
  habitat_baselines.num_environments=${NUM_ENVIRONMENTS} \
  habitat_baselines.eval.split=val \
  habitat_baselines.evaluate=true \
  habitat.simulator.type=${SIMULATOR_TYPE} \
  habitat.simulator.scene_dataset=${SCENE_DATASET} \
  habitat.dataset.scenes_dir=${SCENES_DIR} \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.eval_ckpt_path_dir=${CHECKPOINT_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat.environment.max_episode_steps=1000 \
  habitat.dataset.content_scenes=${CONTENT_SCENES} \
  habitat.dataset.data_path=${DATA_PATH}/val/val.json.gz \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.eval.use_ckpt_config=False \
  habitat_baselines.writer_type=wb \
  habitat_baselines.wb.entity=$WB_ENTITY \
  habitat_baselines.wb.run_name=$TENSORBOARD_DIR \
  habitat_baselines.wb.project_name=$PROJECT_NAME \
  habitat_baselines.updater_name=ImageNavPPO \
  habitat_baselines.distrib_updater_name=DistributedImageNavPPO \
  habitat_baselines.rl.ppo.optimizer_name="adamw" \
  habitat_baselines.rl.ppo.adamw_weight_decay=1e-6 \
  habitat.task.lab_sensors.image_goal_rotation_sensor.sample_angle=false \
  habitat_baselines.start_from_checkpoint="ckpts/hssd_ckpt_332.pth" \