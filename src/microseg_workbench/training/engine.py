"""Model construction, full-image inference and checkpoint I/O."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.augment import IGNORE_INDEX
from ..core.metrics import confusion_matrix, metrics_from_confusion
from .data import Sample
from .model import ModelConfig, UNet

CHECKPOINT_SCHEMA = "microseg-workbench.checkpoint/1"


LOSSES = ("cross_entropy", "cross_entropy_dice")


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean soft Dice over classes present in the batch (truth or prediction); ignore pixels excluded."""
    num_classes = logits.shape[1]
    valid = (target != IGNORE_INDEX).unsqueeze(1).to(logits.dtype)
    probs = logits.float().softmax(1) * valid
    onehot = F.one_hot(target.clamp(max=num_classes - 1), num_classes).permute(0, 3, 1, 2).to(probs.dtype) * valid
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    present = denom > 0
    dice = (2 * inter + 1.0) / (denom + 1.0)
    return 1.0 - dice[present].mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    if kind == "cross_entropy":
        return ce
    if kind == "cross_entropy_dice":
        return ce + soft_dice_loss(logits, target)
    raise ValueError(f"Unknown loss {kind!r}; choose one of {LOSSES}.")


@torch.no_grad()
def predict_logits(model: UNet, image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Full-image inference on float32 (H, W, C); reflect-pads to the U-Net multiple and crops back."""
    model.eval()
    h, w = image.shape[:2]
    m = model.multiple
    ph, pw = (-h) % m, (-w) % m
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to(device)
    if ph or pw:
        mode = "reflect" if ph < h and pw < w else "replicate"
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return model(x)[0, :, :h, :w]


def predict_labels(model: UNet, image: np.ndarray, device: torch.device) -> np.ndarray:
    return predict_logits(model, image, device).argmax(0).to(torch.uint8).cpu().numpy()


@torch.no_grad()
def evaluate(model: UNet, samples: list[Sample], device: torch.device, class_names: list[str]
             ) -> tuple[dict[str, Any], float | None, dict[str, np.ndarray]]:
    """Return (metrics, mean loss, predictions by stem) over full images."""
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    losses, preds = [], {}
    for s in samples:
        logits = predict_logits(model, s.image, device)
        target = torch.from_numpy(s.labels.astype(np.int64)).to(device)
        if (target != IGNORE_INDEX).any():
            losses.append(float(F.cross_entropy(logits[None], target[None], ignore_index=IGNORE_INDEX)))
        pred = logits.argmax(0).to(torch.uint8).cpu().numpy()
        preds[s.stem] = pred
        cm += confusion_matrix(s.labels, pred, k)
    metrics = metrics_from_confusion(cm, class_names)
    return metrics, (float(np.mean(losses)) if losses else None), preds


def save_checkpoint(path: Path, model: UNet, payload: dict[str, Any]) -> None:
    """Atomic checkpoint write (temp file + replace) so a crash never leaves a torn file."""
    data = dict(payload)
    data["schema"] = CHECKPOINT_SCHEMA
    data["model_config"] = model.cfg.to_dict()
    data["model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[UNet, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path} is not a MicroSegWorkbench checkpoint.")
    model = UNet(ModelConfig(**data["model_config"]))
    model.load_state_dict(data["model_state"])
    model.to(device).eval()
    return model, data
