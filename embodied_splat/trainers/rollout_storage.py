from typing import List

from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.utils.timing import g_timer


@baseline_registry.register_storage
class CustomRolloutStorage(RolloutStorage):
    """
    Filters out the specified keys from the observations before storing them.
    TODO: Right now the arguments are hard-coded, but they should be made configurable
    """

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        actor_critic,
        is_double_buffered: bool = False,
        filter_obs: bool = True,
        obs_filter_keys: List = ["rgb", "depth", "semantic"],
    ):
        super().__init__(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            actor_critic,
            is_double_buffered,
        )
        self.filter_obs = filter_obs
        self.obs_filter_keys = obs_filter_keys
        if self.filter_obs:
            logger.info("Filtering is enabled for rollout storage.")
            for k in self.obs_filter_keys:
                if k in self.buffers["observations"]:
                    logger.info(f"Deleting the key {k} in rollout storage.")
                    del self.buffers["observations"][k]

    @g_timer.avg_time("rollout_storage.insert", level=1)
    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        if next_observations is not None and self.filter_obs:
            for k in self.obs_filter_keys:
                if k in next_observations:
                    del next_observations[k]

        super().insert(
            next_observations,
            next_recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            next_masks,
            buffer_index,
            **kwargs,
        )

    def insert_first_observations(self, batch):
        if self.filter_obs:
            for k in self.obs_filter_keys:
                if k in batch:
                    del batch[k]
        self.buffers["observations"][0] = batch  # type: ignore
