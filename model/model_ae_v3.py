import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedEncoder(nn.Module):
    def __init__(self, scale: int = 16):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=self.scale)


class LearnableDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 16):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class AE_v3(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, scale=16):
        super().__init__()
        self.encoder = FixedEncoder(scale=scale)
        self.decoder = LearnableDecoder(in_channels, out_channels, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)        # [B, C, H/scale, W/scale]
        out = self.decoder(z)      # [B, C, H, W]
        return x, out