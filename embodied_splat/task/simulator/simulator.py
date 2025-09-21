#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import habitat_sim
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from omegaconf import DictConfig


@registry.register_simulator(name="CustomSim-v0")
class CustomHabitatSim(HabitatSim):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.navmesh_settings = self.load_navmesh_settings()
        logger.info(
            f"Navmesh settings: radius: {self.navmesh_settings.agent_radius}, height: {self.navmesh_settings.agent_height}, max_climb: {self.navmesh_settings.agent_max_climb}, cell_height: {self.navmesh_settings.cell_height}"
        )
        self.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
        )

    def reconfigure(
        self,
        habitat_config: DictConfig,
        should_close_on_new_scene: bool = True,
    ) -> None:
        is_same_scene = habitat_config.scene == self._current_scene
        super().reconfigure(habitat_config, should_close_on_new_scene)
        if not is_same_scene:
            self.recompute_navmesh(
                self.pathfinder,
                self.navmesh_settings,
            )

    def load_navmesh_settings(self):
        agent_cfg = self.habitat_config.agents.main_agent
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = agent_cfg.height
        navmesh_settings.agent_radius = agent_cfg.radius
        navmesh_settings.agent_max_climb = agent_cfg.max_climb
        navmesh_settings.cell_height = 0.1  # TODO: Remove this hardcoding
        return navmesh_settings

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self.get_sensor_observations()
            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None
