from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SimpleRestorationCNN(nn.Module):
    """Small residual CNN used only to validate the training pipeline."""

    def __init__(self, channels: int = 48, num_blocks: int = 4) -> None:
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(num_blocks)))
        self.exit = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        features = self.entry(degraded)
        residual = self.exit(self.body(features))
        return (degraded + residual).clamp(0.0, 1.0)
