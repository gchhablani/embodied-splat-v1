#!/usr/bin/env python3


import inspect
from typing import Optional

import torch.optim as optim
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.ppo import PPO


@baseline_registry.register_updater
class ImageNavPPO(PPO):
    """
    Custom PPO implementation for AdamW optimizer with visual encoder fine-tuning
    """

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config):
        return cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
            optimizer_name=config.get("optimizer_name", "adam"),
            adamw_weight_decay=config.get("adamw_weight_decay", 1e-6),
            encoder_lr=config.get("encoder_lr", 1.5e-6),
        )

    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
        optimizer_name: str = "adamw",
        adamw_weight_decay: float = 1e-6,
        encoder_lr: float = 1.5e-6,
    ) -> None:

        self.optimizer_name = optimizer_name
        self.adamw_weight_decay = adamw_weight_decay
        self.encoder_lr = encoder_lr

        super().__init__(
            actor_critic=actor_critic,
            clip_param=clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            use_normalized_advantage=use_normalized_advantage,
            entropy_target_factor=entropy_target_factor,
            use_adaptive_entropy_pen=use_adaptive_entropy_pen,
        )

    def _create_optimizer(self, lr, eps):

        # use different lr for visual encoder and other networks
        visual_encoder_params, other_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if (
                    "net.visual_encoder.backbone" in name
                    or "net.goal_visual_encoder.backbone" in name
                ):
                    visual_encoder_params.append(param)
                else:
                    other_params.append(param)
        logger.info(
            f"Number of visual encoder params to train: {sum(param.numel() for param in visual_encoder_params)}"
        )
        logger.info(
            f"Number of other params to train: {sum(param.numel() for param in other_params)}"
        )
        total_params = sum(
            param.numel() for param in visual_encoder_params
        ) + sum(param.numel() for param in other_params)
        logger.info(f"Total params to train: {total_params}")
        if total_params > 0:
            if self.optimizer_name == "adam":
                logger.info("Using Adam optimizer")
                optim_cls = optim.Adam
                optim_kwargs = dict(
                    params=[
                        {
                            "params": visual_encoder_params,
                            "lr": self.encoder_lr,
                        },
                        {"params": other_params, "lr": lr},
                    ],
                    lr=lr,
                    eps=eps,
                )
                signature = inspect.signature(optim_cls.__init__)
                if "foreach" in signature.parameters:
                    optim_kwargs["foreach"] = True
                else:
                    try:
                        import torch.optim._multi_tensor
                    except ImportError:
                        pass
                    else:
                        optim_cls = torch.optim._multi_tensor.Adam
            elif self.optimizer_name == "adamw":
                logger.info("Using AdamW optimizer with weight decay")
                optim_cls = optim.AdamW
                optim_kwargs = dict(
                    params=[
                        {
                            "params": visual_encoder_params,
                            "lr": self.encoder_lr,
                        },
                        {"params": other_params, "lr": lr},
                    ],
                    lr=lr,
                    eps=eps,
                    weight_decay=self.adamw_weight_decay,
                )
                signature = inspect.signature(optim_cls.__init__)
                if "foreach" in signature.parameters:
                    optim_kwargs["foreach"] = True
                else:
                    try:
                        import torch.optim._multi_tensor
                    except ImportError:
                        pass
                    else:
                        optim_cls = torch.optim._multi_tensor.AdamW
            else:
                raise NotImplementedError(
                    f"Optimizer {self.optimizer_name} not implemented"
                )
            return optim_cls(**optim_kwargs)
        else:
            return None


@baseline_registry.register_updater
class DistributedImageNavPPO(DecentralizedDistributedMixin, ImageNavPPO):
    pass
