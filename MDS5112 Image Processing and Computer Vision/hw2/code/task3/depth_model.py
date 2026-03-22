#!/usr/bin/env python
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet50_Weights, resnet50

    _HAS_WEIGHTS_ENUM = True
except ImportError:
    from torchvision.models import resnet50

    _HAS_WEIGHTS_ENUM = False


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet50DepthModel(nn.Module):
    """
    Monocular depth model:
      - Encoder: ResNet50 backbone
      - Decoder: U-Net style upsampling with skip connections
      - Output: positive relative depth (scale/shift ambiguous)
    """

    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        if _HAS_WEIGHTS_ENUM:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(pretrained=pretrained_backbone)

        # ResNet stages
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 256, H/4
        self.layer2 = backbone.layer2  # 512, H/8
        self.layer3 = backbone.layer3  # 1024, H/16
        self.layer4 = backbone.layer4  # 2048, H/32

        # Decoder
        self.up3 = _UpBlock(2048, 1024, 512)
        self.up2 = _UpBlock(512, 512, 256)
        self.up1 = _UpBlock(256, 256, 128)
        self.up0 = _UpBlock(128, 64, 64)

        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64, H/2
        x1 = self.layer1(self.maxpool(x0))  # 256, H/4
        x2 = self.layer2(x1)  # 512, H/8
        x3 = self.layer3(x2)  # 1024, H/16
        x4 = self.layer4(x3)  # 2048, H/32
        return x0, x1, x2, x3, x4

    def decode(self, features: Tuple[torch.Tensor, ...], out_size: Tuple[int, int]) -> torch.Tensor:
        x0, x1, x2, x3, x4 = features
        d3 = self.up3(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)
        d0 = F.interpolate(d0, size=out_size, mode="bilinear", align_corners=False)
        depth_logits = self.head(d0)
        return F.softplus(depth_logits) + 1e-3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.decode(features, out_size=(x.shape[-2], x.shape[-1]))
