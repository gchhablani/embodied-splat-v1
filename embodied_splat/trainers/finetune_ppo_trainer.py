from typing import TYPE_CHECKING

import habitat_baselines.rl.multi_agent  # noqa: F401.
import hydra
from habitat import logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer


@baseline_registry.register_trainer(name="finetune_ddppo")
@baseline_registry.register_trainer(name="finetune_ppo")
class FinetunePPOTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.start_from_checkpoint = (
            config.habitat_baselines.start_from_checkpoint
        )

    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """
        Sets up the AgentAccessMgr. You still must call `agent.post_init` after
        this call. This only constructs the object.
        """

        self._create_obs_transforms()
        agent = baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type
        )(
            config=self.config,
            env_spec=self._env_spec,
            is_distrib=self._is_distributed,
            device=self.device,
            resume_state=resume_state,
            num_envs=self.envs.num_envs,
            percent_done_fn=self.percent_done,
            **kwargs,
        )

        if resume_state is None:
            logger.info(
                "Found no resume state. Assuming training starting from pre-trained checkpoint."
            )
            assert (
                self.start_from_checkpoint is not None
            ), "start_from_checkpoint must not be None if no resume state is specified."
            agent.load_ckpt_state_dict(
                self.load_checkpoint(
                    self.start_from_checkpoint, map_location="cpu"
                )
            )
        else:
            logger.info(
                "Found resume state. Assuming training is resuming from an intermediate point."
            )
        return agent

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"Loaded checkpoint trained for {step_id} steps")
        else:
            ckpt_dict = {"config": None}

        if "config" not in ckpt_dict:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            n_agents = len(config.habitat.simulator.agents)
            for agent_i in range(n_agents):
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agent_config = get_agent_config(
                    config.habitat.simulator, agent_i
                )

                agent_sensors = agent_config.sim_sensors
                extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
                with read_write(agent_sensors):
                    agent_sensors.update(extra_sensors)
                with read_write(config):
                    if config.habitat.gym.obs_keys is not None:
                        for render_view in extra_sensors.values():
                            if (
                                render_view.uuid
                                not in config.habitat.gym.obs_keys
                            ):
                                if n_agents > 1:
                                    config.habitat.gym.obs_keys.append(
                                        f"{agent_name}_{render_view.uuid}"
                                    )
                                else:
                                    config.habitat.gym.obs_keys.append(
                                        render_view.uuid
                                    )

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        if (
            self._agent.actor_critic.should_load_agent_state
            and self.config.habitat_baselines.eval.should_load_ckpt
        ):
            self._agent.load_state_dict(ckpt_dict)

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        evaluator = hydra.utils.instantiate(config.habitat_baselines.evaluator)
        assert isinstance(evaluator, Evaluator)
        evaluator.evaluate_agent(
            self._agent,
            self.envs,
            self.config,
            checkpoint_index,
            step_id,
            writer,
            self.device,
            self.obs_transforms,
            self._env_spec,
            self._rank0_keys,
        )

        self.envs.close()
