"""Synchronized geometric and photometric augmentation of image/label pairs.

Scientific rules enforced here:

* Geometric transforms move image and labels identically. Labels are only ever
  permuted or nearest-neighbour sampled, so no new class values appear.
* Pixels that come from outside the original image (rotation corners, padding)
  are labelled ``IGNORE_INDEX`` and excluded from the loss and metrics; they are
  never silently called background.
* Photometric transforms change the image only.
* All randomness comes from the ``numpy.random.Generator`` passed in, so a seed
  reproduces the exact same augmented sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from PIL import Image

IGNORE_INDEX = 255


class AugmentConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AugmentConfig:
    patch_size: int = 64
    hflip: float = 0.5
    vflip: float = 0.5
    rot90: float = 0.5
    rotate_prob: float = 0.0
    rotate_max_deg: float = 15.0
    photometric_prob: float = 0.8
    brightness: float = 0.08
    contrast: float = 0.15
    gamma: float = 0.0
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hflip", "vflip", "rot90", "rotate_prob", "photometric_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise AugmentConfigError(f"{name} is a probability and must be between 0 and 1 (got {value}).")
        for name in ("rotate_max_deg", "brightness", "contrast", "gamma", "noise_std"):
            if getattr(self, name) < 0:
                raise AugmentConfigError(f"{name} must not be negative.")
        if self.patch_size < 16 or self.patch_size % 8:
            raise AugmentConfigError("patch_size must be a multiple of 8 and at least 16.")
        if self.contrast >= 1.0:
            raise AugmentConfigError("contrast must be below 1 (it is a relative change).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AugmentConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise AugmentConfigError(f"Unknown augmentation options: {sorted(unknown)}")
        return cls(**payload)


NO_AUGMENT = dict(hflip=0.0, vflip=0.0, rot90=0.0, rotate_prob=0.0, photometric_prob=0.0)


def _rotate(image: np.ndarray, labels: np.ndarray, angle: float, image_resample: int) -> tuple[np.ndarray, np.ndarray]:
    channels = [
        np.asarray(Image.fromarray(np.ascontiguousarray(image[..., c]), mode="F")
                   .rotate(angle, resample=image_resample, fillcolor=0.0), dtype=np.float32)
        for c in range(image.shape[2])
    ]
    rotated_labels = np.asarray(Image.fromarray(labels, mode="L").rotate(
        angle, resample=Image.Resampling.NEAREST, fillcolor=IGNORE_INDEX), dtype=np.uint8)
    return np.stack(channels, axis=2), rotated_labels


def random_crop(image: np.ndarray, labels: np.ndarray, size: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
    """Crop ``size``x``size``; images smaller than that are padded (padding = ignore)."""
    h, w = labels.shape
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)), constant_values=0.0)
        labels = np.pad(labels, ((0, ph), (0, pw)), constant_values=IGNORE_INDEX)
        h, w = labels.shape
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return image[y:y + size, x:x + size], labels[y:y + size, x:x + size]


def augment(image: np.ndarray, labels: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator, *,
            crop: bool = True, image_resample: int = Image.Resampling.BILINEAR
            ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Augment a float32 ``(H, W, C)`` image in [0, 1] and uint8 ``(H, W)`` labels.

    Returns the new image, labels and a human-readable list of applied operations.
    ``image_resample`` exists so tests can verify alignment with nearest sampling.
    """
    if image.shape[:2] != labels.shape:
        raise ValueError(f"Image {image.shape[:2]} and labels {labels.shape} are not aligned.")
    ops: list[str] = []
    # Draw every random number unconditionally so the stream is stable when options change.
    u = rng.random(5)
    k = int(rng.integers(1, 4))
    angle = float(rng.uniform(-cfg.rotate_max_deg, cfg.rotate_max_deg))
    b, c, g = rng.uniform(-1.0, 1.0, size=3)
    noise_seed = int(rng.integers(0, 2**31 - 1))

    if cfg.rotate_prob and u[3] < cfg.rotate_prob and angle != 0.0:
        image, labels = _rotate(image, labels, angle, image_resample)
        ops.append(f"rotate {angle:+.1f} deg")
    if crop:
        image, labels = random_crop(image, labels, cfg.patch_size, rng)
    if u[0] < cfg.hflip:
        image, labels = image[:, ::-1], labels[:, ::-1]
        ops.append("flip left-right")
    if u[1] < cfg.vflip:
        image, labels = image[::-1], labels[::-1]
        ops.append("flip up-down")
    if u[2] < cfg.rot90:
        image, labels = np.rot90(image, k, axes=(0, 1)), np.rot90(labels, k, axes=(0, 1))
        ops.append(f"rotate {90 * k} deg")
    if u[4] < cfg.photometric_prob:
        if cfg.contrast:
            factor = 1.0 + c * cfg.contrast
            mean = float(image.mean())
            image = (image - mean) * factor + mean
            ops.append(f"contrast x{factor:.2f}")
        if cfg.brightness:
            image = image + b * cfg.brightness
            ops.append(f"brightness {b * cfg.brightness:+.3f}")
        if cfg.gamma:
            gamma = float(np.exp(g * cfg.gamma))
            image = np.clip(image, 0.0, 1.0) ** gamma
            ops.append(f"gamma {gamma:.2f}")
        if cfg.noise_std:
            image = image + np.random.default_rng(noise_seed).normal(0, cfg.noise_std, image.shape)
            ops.append(f"noise sd {cfg.noise_std:.3f}")
        image = np.clip(image, 0.0, 1.0)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(labels), ops
