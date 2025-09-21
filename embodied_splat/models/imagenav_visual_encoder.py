#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import numpy as np
import torch
import torch.nn as nn
from habitat import logger
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)

from embodied_splat.task.sensors import ImageGoalRotationSensor


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone_config: str,
        input_channels: int = 3,
        normalize_visual_inputs: bool = True,
        use_augmentations: bool = False,
    ):
        super().__init__()

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        if use_augmentations is False:
            backbone_config.transform.jitter = False
            backbone_config.transform.shift = False

        (
            self.backbone,
            self.embed_dim,
            self.visual_transform,
            _,
        ) = hydra.utils.call(backbone_config)

        self.compression, _, self.output_size = create_compression_layer(
            self.embed_dim * 2,
            self.backbone.final_spatial[0],
            self.backbone.final_spatial[1],
            after_compression_flat_size=2 * 2048,
            kernel_size=3,
        )

    def transform_images(self, observations, num_envs):
        images = None
        goal_images = None
        # RGB observation
        for sensor_uuid in [
            "rgb",
        ]:
            if sensor_uuid in observations:
                images = observations[sensor_uuid]

        for sensor_uuid in [
            ImageGoalRotationSensor.cls_uuid,
        ]:
            if sensor_uuid in observations:
                goal_images = observations[sensor_uuid]
        x = torch.cat([images, goal_images], dim=0)

        x = (
            x.permute(0, 3, 1, 2).float() / 255
        )  # convert channels-last to channels-first

        if torch.isnan(x).any():
            logger.info("NaN detected in the input to transform_images")

        x = self.visual_transform(x, num_envs)

        return x

    def forward(self, observations, num_envs) -> torch.Tensor:  # type: ignore
        # Run the `transform_images` function
        x = self.transform_images(observations, num_envs)

        # Check for NaNs in the output of `transform_images`
        if torch.isnan(x).any():
            logger.info("NaN detected in the output of transform_images")

        x = self.running_mean_and_var(x)
        if torch.isnan(x).any():
            logger.info("NaN detected in the output of running mean and var")

        check_for_nans_in_parameters(self.backbone)
        x = self.backbone(x)
        if torch.isnan(x).any():
            logger.info("NaN detected in the output of backbone")

        # combine the image and goal image features
        x1, x2 = x.split(x.shape[0] // 2, dim=0)
        x = torch.cat([x1, x2], dim=1)

        x = self.compression(x)
        if torch.isnan(x).any():
            logger.info("NaN detected in the output of compression")

        if torch.isnan(x).any():
            logger.info("NaN detected in the output of VC-1")

        return x


def check_for_nans_in_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.info(f"NaN detected in the parameter: {name}")
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.info(f"NaN detected in the gradient of parameter: {name}")


def create_compression_layer(
    embed_dim,
    final_spatial_x,
    final_spatial_y,
    after_compression_flat_size=2048,
    kernel_size=3,
):
    if kernel_size > 0:
        num_compression_channels = int(
            round(
                after_compression_flat_size
                / (final_spatial_x * final_spatial_y)
            )
        )

        if kernel_size == 1:
            padding = 0
        else:
            padding = 1
        compression = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                num_compression_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
            nn.Flatten(),
        )
    elif kernel_size == 0:
        num_compression_channels = embed_dim
        compression = nn.Flatten()
    else:
        raise ValueError(
            "Invalid input for kernel size: {}".format(kernel_size)
        )

    output_shape = (
        num_compression_channels,
        final_spatial_x,
        final_spatial_y,
    )
    output_size = np.prod(output_shape)

    return compression, output_shape, output_size
