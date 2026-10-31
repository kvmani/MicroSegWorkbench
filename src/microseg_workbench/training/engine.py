"""Model construction, full-image inference and checkpoint I/O."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ..core.atomic import replace_with_retry
from ..core.augment import IGNORE_INDEX
from ..core.metrics import confusion_matrix, metrics_from_confusion
from ..core.preprocess import Preprocess, normalize, resize_image
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


TILE = 512
TILE_OVERLAP = 64


@torch.no_grad()
def predict_probabilities(model: UNet, image: np.ndarray, device: torch.device, tile: int = TILE,
                          overlap: int = TILE_OVERLAP) -> torch.Tensor:
    """Class probabilities (K, H, W) on the CPU.

    Images larger than ``tile`` are processed in overlapping tiles (bounded GPU memory); overlapping
    probabilities are averaged. Small images are processed whole.
    """
    h, w = image.shape[:2]
    if h <= tile and w <= tile:
        return predict_logits(model, image, device).float().softmax(0).cpu()
    if overlap * 2 >= tile:
        raise ValueError("tile overlap must be less than half the tile size")
    stride = tile - overlap
    ys = sorted({min(y, max(0, h - tile)) for y in range(0, max(1, h - overlap), stride)})
    xs = sorted({min(x, max(0, w - tile)) for x in range(0, max(1, w - overlap), stride)})
    acc = torch.zeros(model.cfg.num_classes, h, w)
    count = torch.zeros(1, h, w)
    for y in ys:
        for x in xs:
            crop = image[y:y + tile, x:x + tile]
            probs = predict_logits(model, crop, device).float().softmax(0).cpu()
            acc[:, y:y + crop.shape[0], x:x + crop.shape[1]] += probs
            count[:, y:y + crop.shape[0], x:x + crop.shape[1]] += 1
    return acc / count


def segment_probabilities(model: UNet, image: np.ndarray, device: torch.device, prep: Preprocess,
                          tile: int = TILE) -> torch.Tensor:
    """Probabilities (K, H, W) at the ORIGINAL resolution for a [0, 1] image (resize, normalize, predict, map back)."""
    h, w = image.shape[:2]
    scaled = resize_image(image, prep.scaled_shape(h, w))
    probs = predict_probabilities(model, normalize(scaled, prep.normalization), device, tile=tile)
    if probs.shape[1:] != (h, w):
        probs = F.interpolate(probs[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    return probs


def predict_labels(model: UNet, image: np.ndarray, device: torch.device, prep: Preprocess | None = None,
                   tile: int = TILE) -> np.ndarray:
    """Model channel index per pixel (uint8) at the image's original resolution."""
    return segment_probabilities(model, image, device, prep or Preprocess(), tile=tile).argmax(0).to(torch.uint8).numpy()


@torch.no_grad()
def evaluate(model: UNet, samples: Iterable[Sample], device: torch.device, class_names: list[str],
             prep: Preprocess | None = None, keep: set[str] | None = None,
             on_prediction: Callable[[Sample, np.ndarray], None] | None = None,
             ) -> tuple[dict[str, Any], float | None, dict[str, np.ndarray]]:
    """Metrics, mean cross-entropy and predictions over the analysed region of each image, at original resolution.

    The excluded border (``prep.exclude_border``) is removed first and never scored; returned predictions and
    ``on_prediction`` receive the cropped sample and prediction. Images are processed one at a time. ``keep`` limits which predictions are returned (None = all);
    ``on_prediction`` receives every (sample, prediction) as it is produced.
    """
    prep = prep or Preprocess()
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    losses, preds = [], {}
    for s in samples:
        s = Sample(s.stem, prep.crop(s.image), prep.crop(s.labels), s.stats)
        probs = segment_probabilities(model, s.image, device, prep)
        target = torch.from_numpy(s.labels.astype(np.int64))
        if (target != IGNORE_INDEX).any():
            losses.append(float(F.nll_loss(probs.clamp_min(1e-12).log()[None], target[None],
                                           ignore_index=IGNORE_INDEX)))
        pred = probs.argmax(0).to(torch.uint8).numpy()
        if keep is None or s.stem in keep:
            preds[s.stem] = pred
        if on_prediction is not None:
            on_prediction(s, pred)
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
        replace_with_retry(tmp, path)
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
