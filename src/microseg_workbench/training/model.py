"""A small, readable U-Net for semantic segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int
    num_classes: int
    base_channels: int = 16
    depth: int = 3
    architecture: str = "unet"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.depth < 1 or cfg.base_channels < 4:
            raise ValueError("depth must be >= 1 and base_channels >= 4")
        self.cfg = cfg
        chans = [cfg.base_channels * 2**i for i in range(cfg.depth + 1)]
        self.down = nn.ModuleList([_block(cfg.in_channels, chans[0])] +
                                  [_block(chans[i], chans[i + 1]) for i in range(cfg.depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2)
                                 for i in reversed(range(cfg.depth))])
        self.merge = nn.ModuleList([_block(chans[i] * 2, chans[i]) for i in reversed(range(cfg.depth))])
        self.head = nn.Conv2d(chans[0], cfg.num_classes, 1)

    @property
    def multiple(self) -> int:
        return 2 ** self.cfg.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
