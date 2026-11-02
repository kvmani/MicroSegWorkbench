"""Network architectures: a registry so different networks can be trained on the same data and compared.

Every architecture is built from a :class:`ModelConfig` (input channels, number of classes, a width knob
``base_channels``, a depth knob ``depth``) and must return a ``torch.nn.Module`` that maps ``(N, C, H, W)`` to
``(N, classes, H, W)`` logits for inputs whose height and width are multiples of ``multiple(cfg)``.

Built in: ``unet`` (convolutional U-Net) and ``segformer`` (a light SegFormer-style transformer written in plain
PyTorch, no extra packages). More can be added by plugins (``MSW_PLUGINS``) with :func:`register_architecture`; the
checkpoint stores the architecture name, so a model only loads where its architecture is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .model import ModelConfig, UNet


@dataclass(frozen=True)
class Architecture:
    name: str
    title: str
    description: str
    builder: Callable[[ModelConfig], nn.Module]
    multiple: Callable[[ModelConfig], int]
    size_help: str
    module: str


ARCHITECTURES: dict[str, Architecture] = {}


def register_architecture(name: str, title: str, description: str, builder: Callable[[ModelConfig], nn.Module],
                          multiple: Callable[[ModelConfig], int], size_help: str = "") -> None:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Architecture name {name!r} must be letters, digits or _.")
    ARCHITECTURES[name] = Architecture(name, title, description, builder, multiple, size_help,
                                       getattr(builder, "__module__", "?"))


class UnknownArchitecture(ValueError):
    pass


def build_model(cfg: ModelConfig) -> nn.Module:
    """Build the network named by ``cfg.architecture``; the module gets ``.cfg`` and ``.multiple``."""
    arch = ARCHITECTURES.get(cfg.architecture)
    if arch is None:
        from ..plugins import load_plugins

        load_plugins()
        arch = ARCHITECTURES.get(cfg.architecture)
    if arch is None:
        raise UnknownArchitecture(f"Network {cfg.architecture!r} is not available here (known: "
                                  f"{sorted(ARCHITECTURES)}). Load the plugin that provides it (MSW_PLUGINS).")
    model = arch.builder(cfg)
    if not isinstance(model, UNet):  # UNet defines both itself
        model.cfg = cfg
        model.multiple = arch.multiple(cfg)
    return model


# ---- SegFormer-lite ------------------------------------------------------------------------------------------
class _OverlapPatchEmbed(nn.Module):
    def __init__(self, cin: int, dim: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, kernel, stride, kernel // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class _EfficientAttention(nn.Module):
    """Multi-head self-attention whose keys/values come from a spatially reduced map (SegFormer)."""

    def __init__(self, dim: int, heads: int, sr: int) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.sr = nn.Conv2d(dim, dim, sr, sr) if sr > 1 else None
        self.norm = nn.LayerNorm(dim) if sr > 1 else None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        if self.sr is not None:
            r = self.sr(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
            r = self.norm(r)
        else:
            r = x
        k, v = self.kv(r).reshape(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class _MixFFN(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        b, n, c = x.shape
        x = self.dw(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, sr: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _EfficientAttention(dim, heads, sr)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = _MixFFN(dim, 4 * dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.n1(x), h, w)
        return x + self.ffn(self.n2(x), h, w)


class SegFormerLite(nn.Module):
    """SegFormer-style network: hierarchical transformer encoder (strides 4, 8, 16, 32) and an all-MLP decoder.

    ``base_channels`` sets the width (stage widths 2b, 4b, 8b, 16b), ``depth`` the number of stages (1-4).
    """

    HEADS = (1, 2, 4, 8)
    SR = (8, 4, 2, 1)

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        stages = max(1, min(4, cfg.depth))
        dims = [cfg.base_channels * 2 * 2**i for i in range(stages)]
        self.embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        cin = cfg.in_channels
        for i, dim in enumerate(dims):
            self.embeds.append(_OverlapPatchEmbed(cin, dim, 7 if i == 0 else 3, 4 if i == 0 else 2))
            heads = min(self.HEADS[i], max(1, dim // 16))
            self.blocks.append(nn.ModuleList([_Block(dim, heads, self.SR[i]) for _ in range(2)]))
            self.norms.append(nn.LayerNorm(dim))
            cin = dim
        dec = cfg.base_channels * 8
        self.lin = nn.ModuleList([nn.Conv2d(d, dec, 1) for d in dims])
        self.fuse = nn.Sequential(nn.Conv2d(dec * stages, dec, 1, bias=False), nn.BatchNorm2d(dec), nn.ReLU(True))
        self.head = nn.Conv2d(dec, cfg.num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = []
        for embed, blocks, norm in zip(self.embeds, self.blocks, self.norms):
            t, h, w = embed(x)
            for blk in blocks:
                t = blk(t, h, w)
            x = norm(t).transpose(1, 2).reshape(t.shape[0], -1, h, w)
            feats.append(x)
        first = feats[0].shape[-2:]
        ups = [F.interpolate(lin(f), size=first, mode="bilinear", align_corners=False) if f.shape[-2:] != first
               else lin(f) for lin, f in zip(self.lin, feats)]
        out = self.head(self.fuse(torch.cat(ups, dim=1)))
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


register_architecture(
    "unet", "U-Net (convolutional)",
    "The classic encoder-decoder with skip connections. Fast, reliable on small datasets; a good first choice.",
    UNet, lambda cfg: 2 ** cfg.depth, "width = base channels; depth = number of down-sampling levels")
register_architecture(
    "segformer", "SegFormer-lite (transformer)",
    "A light vision transformer (efficient self-attention, MLP decoder). Sees wider context; usually needs more "
    "data and epochs than U-Net. Written in plain PyTorch, no extra packages.",
    SegFormerLite, lambda cfg: 4 * 2 ** (max(1, min(4, cfg.depth)) - 1),
    "width = base channels (stage widths 2x..16x); depth = number of transformer stages (1-4)")
