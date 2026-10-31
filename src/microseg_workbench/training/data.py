"""Loading a dataset version into memory as model-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.augment import IGNORE_INDEX, AugmentConfig, augment
from ..core.classes import ClassMap
from ..core.dataset import DatasetVersion
from ..core.imageio import load_image, load_mask_values, to_unit_float
from ..core.preprocess import Preprocess, image_stats, normalize, resize_image, resize_labels


@dataclass(frozen=True)
class Sample:
    stem: str
    image: np.ndarray  # float32 (H, W, C) in [0, 1] (dtype-range scaled)
    labels: np.ndarray  # uint8 (H, W) model channel indices, IGNORE_INDEX for ignored
    stats: tuple[np.ndarray, np.ndarray] | None = None  # whole-image mean/std at model scale


def to_training_samples(samples: list[Sample], prep: Preprocess) -> list[Sample]:
    """Resize to model scale and record whole-image statistics for normalization."""
    out = []
    for s in samples:
        shape = prep.scaled_shape(*s.labels.shape)
        image = resize_image(s.image, shape)
        out.append(Sample(s.stem, image, resize_labels(s.labels, shape), image_stats(image)))
    return out


def value_to_channel_lut(class_map: ClassMap) -> np.ndarray:
    """Lookup table: mask value -> model channel index (undeclared values -> IGNORE)."""
    return class_map.channel_lut()


def load_split(version: DatasetVersion, split: str) -> list[Sample]:
    lut = value_to_channel_lut(version.class_map)
    out = []
    for s in sorted(version.samples(split), key=lambda s: s["stem"]):
        img = to_unit_float(load_image(version.image_path(s)).array)
        raw = load_mask_values(version.mask_path(s))
        labels = lut[raw]
        if (labels == IGNORE_INDEX).any():
            raise ValueError(f"Stored mask {s['mask_file']} has values outside the class map; dataset is corrupt.")
        out.append(Sample(stem=s["stem"], image=img, labels=labels))
    return out


def epoch_batches(samples: list[Sample], cfg: AugmentConfig, patches_per_image: int, batch_size: int,
                  seed: int, epoch: int, normalization: str = "dtype_range") -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic augmented batches for one epoch: (N, C, P, P) float32 and (N, P, P) int64."""
    order = [(i, r) for i in range(len(samples)) for r in range(patches_per_image)]
    perm = np.random.default_rng([seed, epoch, 0]).permutation(len(order))
    batches = []
    for start in range(0, len(perm), batch_size):
        imgs, labs = [], []
        for j in perm[start:start + batch_size]:
            i, r = order[j]
            rng = np.random.default_rng([seed, epoch, 1, i, r])
            img, lab, _ = augment(samples[i].image, samples[i].labels, cfg, rng)
            imgs.append(normalize(img, normalization, samples[i].stats).transpose(2, 0, 1))
            labs.append(lab)
        batches.append((np.stack(imgs).astype(np.float32), np.stack(labs).astype(np.int64)))
    return batches
