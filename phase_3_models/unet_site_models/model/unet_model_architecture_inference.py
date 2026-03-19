"""Lightweight U-Net architecture for inference.

This module intentionally avoids importing training-time configuration (e.g. config_param.py)
and other heavy dependencies.

It must remain state-dict compatible with the training architecture so checkpoints load
with keys like:
- downs.0.conv.0.weight
- final_conv.weight
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path
        current_in = int(in_channels)
        for feature in features:
            self.downs.append(DoubleConv(current_in, int(feature), dropout_prob))
            current_in = int(feature)

        # Up path
        for feature in reversed(features):
            feature = int(feature)
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature, dropout_prob))

        self.bottleneck = DoubleConv(int(features[-1]), int(features[-1]) * 2, dropout_prob)
        self.final_conv = nn.Conv2d(int(features[0]), int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

            x = self.ups[idx + 1](torch.cat((skip_connection, x), dim=1))

        x = self.final_conv(x)
        return F.softmax(x, dim=1)
