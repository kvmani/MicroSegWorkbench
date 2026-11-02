"""Rough, honest estimate of what a training run will cost, from one timed step on this computer."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from ..core.dataset import DatasetVersion
from ..core.workspace import Workspace
from .config import TrainConfig
from .data import SplitSource, count_batches
from .device import choose_device
from .architectures import ARCHITECTURES, build_model
from .model import ModelConfig, count_parameters
from .trainer import CPU_AUTO_BATCH


def estimate_run(cfg: TrainConfig, workspace: Workspace, timing_steps: int = 2) -> dict[str, Any]:
    """Epochs, steps, model size, memory for data and a time estimate (timed forward+backward on dummy patches).

    Batch size 0 (automatic) is estimated as the CPU default or 16 on a GPU; validation time is approximated from
    the measured step time. All figures are approximate and labelled as such in the UI.
    """
    version = DatasetVersion.open(workspace.datasets_dir / cfg.dataset_id)
    prep = cfg.preprocess()
    train = SplitSource(version, "train", prep, max_cached=0)
    val = SplitSource(version, "val", max_cached=0)
    channels = next(iter(train.channels()), 1)
    device, info, _note = choose_device(cfg.device)
    batch = cfg.batch_size or (CPU_AUTO_BATCH if device.type == "cpu" else 16)
    steps = count_batches(len(train), cfg.patches_per_image, batch)
    model = build_model(ModelConfig(channels, version.class_map.num_classes, cfg.base_channels, cfg.depth,
                                    cfg.architecture)).to(device)
    p = cfg.augment.patch_size
    x = torch.zeros(batch, channels, p, p, device=device)
    y = torch.zeros(batch, p, p, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    step()  # warm-up
    start = time.perf_counter()
    for _ in range(timing_steps):
        step()
    step_s = (time.perf_counter() - start) / timing_steps
    aug_s = 0.0
    if len(train):  # augmentation runs on the CPU for every patch: time it on a real training image
        import numpy as np

        from ..core.augment import augment

        sample = train.get(0)
        t0 = time.perf_counter()
        for i in range(8):
            augment(sample.image, sample.labels, cfg.augment, np.random.default_rng(i))
        aug_s = (time.perf_counter() - t0) / 8
    val_pixels = sum(r["height"] * r["width"] for r in val.records) * prep.resize_factor ** 2
    val_s = step_s / 3 * val_pixels / (batch * p * p)  # forward only, roughly a third of a training step
    overhead_s = 0.15  # checkpoints and preview images per epoch
    total_s = cfg.epochs * (steps * step_s + len(train) * cfg.patches_per_image * aug_s + val_s + overhead_s)
    return {
        "epochs": cfg.epochs, "steps_per_epoch": steps, "batch_size": batch, "auto_batch": cfg.batch_size == 0,
        "patches_per_epoch": len(train) * cfg.patches_per_image, "patch_size": p, "train_images": len(train),
        "parameters": count_parameters(model), "architecture": ARCHITECTURES[cfg.architecture].title, "device": info.label(), "step_seconds": step_s, "augment_seconds_per_patch": aug_s,
        "total_seconds": total_s, "data_memory_bytes": train.estimated_bytes() + val.estimated_bytes(),
    }
