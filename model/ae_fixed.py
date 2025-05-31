import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Tuple

from model.layers import ResidualBlock, UpBlock
import numpy as np

import os
import scipy
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from torchvision.utils import make_grid, save_image
from skimage.metrics import structural_similarity as cal_ssim
from torch.utils.checkpoint import checkpoint


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 16):
        super().__init__()
        self.scale = scale
        self.out_channels = out_channels
        assert in_channels == out_channels, "Fixed encoder assumes in_channels == out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample by average pooling
        return F.avg_pool2d(x, kernel_size=self.scale)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 16):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        assert in_channels == out_channels, "Fixed decoder assumes in_channels == out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample by bilinear interpolation
        return F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)


class AE_fixed(nn.Module):
    def __init__(self, state_size=4, in_channels=4, out_channels=4, scale=16):
        super(AE_fixed, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels, out_channels=state_size, scale=scale)
        self.decoder = Decoder(in_channels=state_size, out_channels=out_channels, scale=scale)


    def forward(self, observations: torch.Tensor, option: int=1):
        """

        :param observations: [b, num_observations, num_channels, height, width]
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        assert num_observations > 2

        # Sample target snapshots and conditioning
        target_snapshots_indices = torch.randint(low=2, high=num_observations, size=[batch_size])
        target_snapshots = observations[torch.arange(batch_size), target_snapshots_indices]
        reference_snapshots_indices = target_snapshots_indices - 1
        reference_snapshots = observations[torch.arange(batch_size), reference_snapshots_indices]
        conditioning_snapshots_indices = torch.cat([torch.randint(low=0, high=s - 1, size=[1]) for s in target_snapshots_indices], dim=0)
        conditioning_snapshots = observations[torch.arange(batch_size), conditioning_snapshots_indices]
        
        # Encoder
        # input snapshots: [b, num_channels, height, width]
        input_snapshots = torch.stack([target_snapshots, reference_snapshots, conditioning_snapshots], dim=1)
        flat_input_snapshots = rearrange(input_snapshots, "b n c h w -> (b n) c h w")
        flat_latents = self.encoder(flat_input_snapshots)
        
        # Decode latents so that we can train the AE
        reconstructed_flat_snapshots = self.decoder(flat_latents)
        reconstructed_snapshots = rearrange(reconstructed_flat_snapshots, "(b n) c h w -> b n c h w", n=3)

        return input_snapshots, reconstructed_snapshots

    

