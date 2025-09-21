from typing import TYPE_CHECKING, Any

from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class CustomCollisions(Measure):
    cls_uuid: str = "custom_collisions"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__()
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(self, episode: EmbodiedTask, *args: Any, **kwargs: Any):
        self._metric = 0
        if self._sim.previous_step_collided:
            self._metric = 1


@registry.register_measure
class CollisionPenalty(Measure):
    """
    Returns a penalty value if the robot has collided.
    """

    cls_uuid: str = "collision_penalty"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._collision_penalty = config.collision_penalty
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        # print(task.measurements.measures)
        task.measurements.check_measure_dependencies(
            self.uuid, [CustomCollisions.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        collisions = task.measurements.measures[
            CustomCollisions.cls_uuid
        ].get_metric()
        collided = collisions > 0
        if collided:
            self._metric = -self._collision_penalty
        else:
            self._metric = 0
