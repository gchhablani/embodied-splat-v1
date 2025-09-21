import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from vc_models.transforms.randomize_env_transform import RandomizeEnvTransform
from vc_models.transforms.to_tensor_if_not import ToTensorIfNot


def transform_augment(
    # Resize/crop
    resize_size=256,
    output_size=224,
    # Jitter
    jitter=True,
    jitter_prob=1.0,
    jitter_brightness=0.3,
    jitter_contrast=0.3,
    jitter_saturation=0.3,
    jitter_hue=0.3,
    # Shift
    shift=True,
    shift_pad=4,
    # Randomize environments
    randomize_environments=False,
    normalize=False,
):
    if type(resize_size) is omegaconf.listconfig.ListConfig:
        resize_size = omegaconf.OmegaConf.to_object(resize_size)
    if type(output_size) is omegaconf.listconfig.ListConfig:
        output_size = omegaconf.OmegaConf.to_object(output_size)
    transforms = [
        ToTensorIfNot(),
        T.Resize(
            resize_size,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.CenterCrop(output_size),
    ]

    if jitter:
        transforms.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        jitter_brightness,
                        jitter_contrast,
                        jitter_saturation,
                        jitter_hue,
                    )
                ],
                p=jitter_prob,
            )
        )

    if shift:
        transforms.append(RandomShiftsAug(shift_pad))

    if normalize:
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transforms = T.Compose(transforms)

    return RandomizeEnvTransform(
        transforms, randomize_environments=randomize_environments
    )


# Implementation borrowed from here:
# https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if len(x.shape) == 3:
            single_frame = True
            x = x.unsqueeze(0)
        else:
            single_frame = False

        n, _, h, w = x.size()

        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")

        eps_h = 1.0 / (h + 2 * self.pad)
        arange_h = torch.linspace(
            -1.0 + eps_h,
            1.0 - eps_h,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]

        eps_w = 1.0 / (w + 2 * self.pad)
        arange_w = torch.linspace(
            -1.0 + eps_w,
            1.0 - eps_w,
            w + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:w]
        arange_h = arange_h.unsqueeze(1).repeat(1, w).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(h, 1).unsqueeze(2)

        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(1, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )

        shift[:, :, :, 0] *= 2.0 / (w + 2 * self.pad)
        shift[:, :, :, 1] *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        out = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        out = out.squeeze(0) if single_frame else out
        return out
