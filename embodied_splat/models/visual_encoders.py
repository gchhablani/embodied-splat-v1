import torch
import torch.nn as nn
from habitat.core.logging import logger
from vc_models.models.compression_layer import create_compression_layer
from vc_models.models.vit import model_utils


class Vc1Wrapper(nn.Module):
    """
    Wrapper for the VC1 visual encoder. This will automatically download the model if it's not already.
    """

    def __init__(self, im_obs_space, model_id=None):
        super().__init__()

        if model_id is None:
            model_id = model_utils.VC1_BASE_NAME

        (
            self.net,
            self.embd_size,
            self.model_transforms,
            model_info,
        ) = model_utils.load_model(model_id)
        # These policies are not using depth sensors
        logger.info(f"Transforms: {self.model_transforms}")
        self._image_obs_keys = [
            k for k in im_obs_space.spaces.keys() if k != "depth"
        ]
        logger.info(f"Using image keys: {self._image_obs_keys}")
        # Count total # of channels
        self._n_input_channels = sum(
            im_obs_space.spaces[k].shape[2] for k in self._image_obs_keys
        )
        self.out_dim = 1

        for param in self.net.parameters():
            param.requires_grad = False
        for module in self.net.modules():
            if "BatchNorm" in type(module).__name__:
                module.momentum = 0.0
        if model_id.endswith("no_cls"):
            self.no_cls = True
            logger.info("Using VC1 with compression")
            self.compression, _, self.output_size = create_compression_layer(
                self.embd_size, self.net.final_spatial
            )
        else:
            self.no_cls = False
            logger.info("Using VC1 without compression, with cls token")
            self.compression = nn.Identity()
            self.output_size = self.embd_size
        self.net.eval()

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    @torch.autocast("cuda")
    def forward(self, obs):
        # Extract tensors that are shape [batch_size, img_width, img_height, img_channels]
        feats = []
        imgs = [v for k, v in obs.items() if k in self._image_obs_keys]

        # NOTE: We will only have a single image, so this does not really matter.
        for img in imgs:
            if img.shape[-1] != 3:
                img = torch.concat([img] * 3, dim=-1)
                scale_factor = 1.0
            else:
                scale_factor = 255.0

            img = self.model_transforms(
                img.permute(0, 3, 1, 2).contiguous() / scale_factor
            )
            feats.append(self.net(img))

        if len(feats) == 2:
            # feats = (feats[0] + feats[1])/2
            feats = torch.concat(feats, dim=-1)
        else:
            feats = feats[0]
        feats = self.compression(feats)
        return feats.flatten(1)

    @property
    def output_shape(self):
        return (self.output_size * len(self._image_obs_keys),)

    @property
    def feats_size(self):
        return self.output_size * len(self._image_obs_keys)
