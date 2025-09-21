#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
from gym import spaces
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from torch import nn as nn

from embodied_splat.models.imagenav_visual_encoder import VisualEncoder


class ImageNavNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        backbone_config,
        hidden_size: int,
        rnn_type: str,
        num_recurrent_layers: int,
        use_augmentations: bool,
        use_augmentations_test_time: bool,
        eval_run: str,
        freeze_backbone: bool,
    ):
        super().__init__()

        rnn_input_size = 0

        # visual encoder
        assert (
            "rgb" in observation_space.spaces
        ), "ImageNavNet requires rgb in the observation space"
        # check for goal
        assert (
            "image_goal_rotation" in observation_space.spaces
        ), "ImageNavNet requires image_goal_rotation in the observation space"
        if (use_augmentations and not eval_run) or (
            use_augmentations_test_time and eval_run
        ):
            use_augmentations = True

        self.visual_encoder = VisualEncoder(
            backbone_config=backbone_config,
            use_augmentations=use_augmentations,
        )

        # 2x because we have goal images
        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_encoder.output_size, hidden_size * 2),
            nn.ReLU(True),
        )

        rnn_input_size += hidden_size * 2

        # previous action embedding
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        rnn_input_size += 32

        # state encoder
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        # TODO: move this to the model files
        # freeze backbone
        if freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

        # save configuration
        self._hidden_size = hidden_size

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_loss_state = {}

        # number of environments
        N = rnn_hidden_states.size(0)

        # visual encoder
        perception_embed = self.visual_encoder(observations, N)
        # aux_loss_state["perception_embed"] = perception_embed
        perception_embed = self.visual_fc(perception_embed)

        if torch.isnan(perception_embed).any():
            logger.info("NaN detected in the output of visual FC")

        x = [perception_embed]

        # previous action embedding
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )
        x.append(prev_actions)

        # state encoder
        x_out = torch.cat(x, dim=1)

        if torch.isnan(x_out).any():
            logger.info("NaN detected in the input of RNN")

        x_out, rnn_hidden_states = self.state_encoder(
            x_out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        # aux_loss_state["rnn_output"] = x_out

        if torch.isnan(x_out).any():
            logger.info("NaN detected in the output of RNN")

        return x_out, rnn_hidden_states, aux_loss_state


@baseline_registry.register_policy
class ImageNavPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        backbone_config,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        num_recurrent_layers: int = 1,
        use_augmentations: bool = False,
        use_augmentations_test_time: bool = False,
        eval_run: bool = False,
        freeze_backbone: bool = False,
        aux_loss_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(
            ImageNavNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                backbone_config=backbone_config,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                use_augmentations=use_augmentations,
                use_augmentations_test_time=use_augmentations_test_time,
                eval_run=eval_run,
                freeze_backbone=freeze_backbone,
            ),
            action_space=action_space,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        agent_name,
        **kwargs
    ):
        return cls(
            observation_space=observation_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            backbone_config=config.habitat_baselines.rl.policy[
                agent_name
            ].backbone_config,
            rnn_type=config.habitat_baselines.rl.policy[agent_name].rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.policy[
                agent_name
            ].num_recurrent_layers,
            use_augmentations=config.habitat_baselines.rl.policy[
                agent_name
            ].use_augmentations,
            use_augmentations_test_time=config.habitat_baselines.rl.policy[
                agent_name
            ].use_augmentations_test_time,
            eval_run=config.habitat_baselines.evaluate,
            freeze_backbone=config.habitat_baselines.rl.policy[
                agent_name
            ].freeze_backbone,
            action_space=action_space,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
        )
