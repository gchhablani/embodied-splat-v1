from dataclasses import dataclass, field
from typing import Any, Dict

from habitat.config.default_structured_configs import (
    LabSensorConfig,
    MeasurementConfig,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesRLConfig,
    PolicyConfig,
    PPOConfig,
    RLConfig,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

cs = ConfigStore.instance()


@dataclass
class ImageGoalRotationSensorConfig(LabSensorConfig):
    type: str = "ImageGoalRotationSensor"
    sample_angle: bool = True


@dataclass
class ImageNavRewardMeasurementConfig(MeasurementConfig):
    type: str = "ImageNavReward"
    success_reward: float = 2.5
    angle_success_reward: float = 2.5
    slack_penalty: float = -0.01
    use_atg_reward: bool = True
    use_dtg_reward: bool = True
    use_atg_fix: bool = True
    atg_reward_distance: float = 1.0


@dataclass
class CollisionPenaltyMeasurementConfig(MeasurementConfig):
    type: str = "CollisionPenalty"
    collision_penalty: float = 0.03


@dataclass
class CustomCollisionsMeasurementConfig(MeasurementConfig):
    type: str = "CustomCollisions"


@dataclass
class ImageNavRewardCollisionPenaltyMeasurementConfig(MeasurementConfig):
    type: str = "ImageNavRewardCollisionPenalty"
    success_reward: float = 5.0
    angle_success_reward: float = 5.0
    slack_penalty: float = -0.01
    use_atg_reward: bool = True
    use_dtg_reward: bool = True
    use_atg_fix: bool = True
    atg_reward_distance: float = 1.0


@dataclass
class AngleSuccessMeasurementConfig(MeasurementConfig):
    type: str = "AngleSuccess"
    success_angle: float = 25.0


@dataclass
class AngleToGoalMeasurementConfig(MeasurementConfig):
    type: str = "AngleToGoal"


@dataclass
class CustomPPOConfig(PPOConfig):
    optimizer_name: str = "adam"
    adamw_weight_decay: float = 1e-6


@dataclass
class CustomRLConfig(RLConfig):
    ppo: CustomPPOConfig = CustomPPOConfig()


@dataclass
class CustomBaselinesRLConfig(HabitatBaselinesRLConfig):
    rl: CustomRLConfig = CustomRLConfig()


######################
# ImageNav Config
######################


@dataclass
class VisualEncoderTransformsConfig:
    _target_: str = (
        "embodied_splat.trainers.image_transforms.transform_augment"
    )
    resize_size: list = field(default_factory=lambda: [160, 120])
    output_size: list = field(default_factory=lambda: [160, 120])
    jitter: bool = True
    jitter_prob: float = 1.0
    jitter_brightness: float = 0.3
    jitter_contrast: float = 0.3
    jitter_saturation: float = 0.3
    jitter_hue: float = 0.3
    shift: bool = True
    shift_pad: int = 4
    randomize_environments: bool = False


@dataclass
class VisualEncoderModelConfig:
    _target_: str = "embodied_splat.trainers.vit.vit_base_patch16"
    img_size: list = field(default_factory=lambda: [160, 120])
    use_cls: bool = False
    global_pool: bool = False
    drop_path_rate: float = 0.0


@dataclass
class VisualEncoderConfig:
    _target_: str = "vc_models.models.vit.vit.load_mae_encoder"
    checkpoint_path: str = "model_ckpts/vc1_vitb.pth"
    model: VisualEncoderModelConfig = VisualEncoderModelConfig()


@dataclass
class VisualEncoderMetadataConfig:
    algo: str = "mae"
    model: str = "vit_base_patch16"
    data: list = field(default_factory=lambda: ["ego", "imagenet", "inav"])
    comment: str = "182_epochs"


@dataclass
class BackboneConfig:
    _target_: str = "vc_models.models.load_model"
    model: VisualEncoderConfig = VisualEncoderConfig()
    transform: VisualEncoderTransformsConfig = VisualEncoderTransformsConfig()
    metadata: VisualEncoderMetadataConfig = VisualEncoderMetadataConfig()


@dataclass
class ImageNavPolicyConfig(PolicyConfig):
    name: str = "ImageNavPolicy"
    backbone_config: BackboneConfig = BackboneConfig()
    rnn_type: str = "GRU"
    num_recurrent_layers: int = 1
    use_augmentations: bool = False
    use_augmentations_test_time: bool = False
    normalize_visual_inputs: bool = False
    freeze_backbone: bool = False


@dataclass
class ImageNavPPOConfig(PPOConfig):
    optimizer_name: str = "adamw"
    adamw_weight_decay: float = 1e-6
    encoder_lr: float = 1.5e-6


@dataclass
class ImageNavRLConfig(RLConfig):
    ppo: ImageNavPPOConfig = ImageNavPPOConfig()
    policy: Dict[str, Any] = field(
        default_factory=lambda: {"main_agent": ImageNavPolicyConfig()}
    )


@dataclass
class ImageNavBaselinesRLConfig(HabitatBaselinesRLConfig):
    rl: ImageNavRLConfig = ImageNavRLConfig()


######################
# Finetune Baselines Config
######################


@dataclass
class FinetuneImageNavBaselinesRLConfig(ImageNavBaselinesRLConfig):
    start_from_checkpoint: str = "ckpts/hm3d_ckpt_204.pth"


cs.store(
    package="habitat.task.lab_sensors.image_goal_rotation_sensor",
    group="habitat/task/lab_sensors",
    name="image_goal_rotation_sensor",
    node=ImageGoalRotationSensorConfig,
)


cs.store(
    package="habitat.task.measurements.imagenav_reward",
    group="habitat/task/measurements",
    name="imagenav_reward",
    node=ImageNavRewardMeasurementConfig,
)


cs.store(
    package="habitat.task.measurements.imagenav_reward_collision_penalty",
    group="habitat/task/measurements",
    name="imagenav_reward_collision_penalty",
    node=ImageNavRewardCollisionPenaltyMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.angle_success",
    group="habitat/task/measurements",
    name="angle_success",
    node=AngleSuccessMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.angle_to_goal",
    group="habitat/task/measurements",
    name="angle_to_goal",
    node=AngleToGoalMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.custom_collisions",
    group="habitat/task/measurements",
    name="custom_collisions",
    node=CustomCollisionsMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.collision_penalty",
    group="habitat/task/measurements",
    name="collision_penalty",
    node=CollisionPenaltyMeasurementConfig,
)

cs.store(
    group="habitat_baselines",
    name="habitat_baselines_rl_config_base",
    node=CustomBaselinesRLConfig,
)

# ImageNav Config
cs.store(
    package="habitat_baselines.rl.policy",
    name="imagenav_policy_base",
    node=ImageNavPolicyConfig,
)
cs.store(
    group="habitat_baselines",
    name="imagenav_baselines_rl_config_base",
    node=ImageNavBaselinesRLConfig,
)

# Finetune Baselines Config
cs.store(
    group="habitat_baselines",
    name="finetune_imagenav_baselines_rl_config_base",
    node=FinetuneImageNavBaselinesRLConfig,
)


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="embodied_splat",
            path="pkg://config/",
        )
        search_path.append(
            provider="embodied_splat",
            path="pkg://config/tasks/",
        )
        search_path.append(
            provider="habitat_baselines",
            path="pkg://habitat_baselines/config/",
        )
