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
from typing import Any, Callable

import numpy as np
from PIL import Image

IGNORE_INDEX = 255

PhotometricOp = Callable[[np.ndarray, np.random.Generator, dict], np.ndarray]
PHOTOMETRIC_OPS: dict[str, dict[str, Any]] = {}


def register_photometric_op(name: str, fn: PhotometricOp, defaults: dict[str, Any] | None = None,
                            description: str = "") -> None:
    """Register an extra image-only augmentation (for plugins).

    ``fn(image, rng, params)`` receives a float32 (H, W, C) image in [0, 1], a seeded generator and the parameters,
    and must return an image of the same shape. It never sees the labels, so it cannot misalign them; all
    randomness must come from ``rng`` so runs stay reproducible.
    """
    if not name.isidentifier():
        raise AugmentConfigError(f"Augmentation name {name!r} must be a simple identifier.")
    PHOTOMETRIC_OPS[name] = {"fn": fn, "defaults": dict(defaults or {}), "description": description,
                             "module": getattr(fn, "__module__", "?")}


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge replication (float32 (H, W, C))."""
    if sigma <= 0:
        return image
    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    out = np.pad(image, ((r, r), (0, 0), (0, 0)), mode="edge")
    out = sum(k[i] * out[i:i + image.shape[0]] for i in range(len(k)))
    out = np.pad(out, ((0, 0), (r, r), (0, 0)), mode="edge")
    out = sum(k[i] * out[:, i:i + image.shape[1]] for i in range(len(k)))
    return out.astype(np.float32)


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
    blur_max_sigma: float = 0.0
    extra_ops: tuple = ()  # ({"name": registered op, "probability": p, "params": {...}}, ...)

    def __post_init__(self) -> None:
        for name in ("hflip", "vflip", "rot90", "rotate_prob", "photometric_prob"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise AugmentConfigError(f"{name} is a probability and must be between 0 and 1 (got {value}).")
        ops = tuple(dict(o) for o in self.extra_ops)
        object.__setattr__(self, "extra_ops", ops)
        for op in ops:
            if op.get("name") not in PHOTOMETRIC_OPS:
                raise AugmentConfigError(f"Unknown augmentation {op.get('name')!r}; is its plugin loaded? "
                                         f"Available: {sorted(PHOTOMETRIC_OPS)}")
            if not 0.0 <= float(op.get("probability", 0)) <= 1.0:
                raise AugmentConfigError(f"Probability of {op['name']} must be between 0 and 1.")
        for name in ("rotate_max_deg", "brightness", "contrast", "gamma", "noise_std", "blur_max_sigma"):
            if getattr(self, name) < 0:
                raise AugmentConfigError(f"{name} must not be negative.")
        if self.patch_size < 16 or self.patch_size % 8:
            raise AugmentConfigError("patch_size must be a multiple of 8 and at least 16.")
        if self.contrast >= 1.0:
            raise AugmentConfigError("contrast must be below 1 (it is a relative change).")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["extra_ops"] = [dict(o) for o in self.extra_ops]
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AugmentConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise AugmentConfigError(f"Unknown augmentation options: {sorted(unknown)}")
        data = dict(payload)
        data["extra_ops"] = tuple(data.get("extra_ops") or ())
        return cls(**data)


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
            crop: bool = True, image_resample: int = Image.Resampling.BILINEAR,
            trace: list | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Augment a float32 ``(H, W, C)`` image in [0, 1] and uint8 ``(H, W)`` labels.

    Returns the new image, labels and a human-readable list of applied operations.
    ``image_resample`` exists so tests can verify alignment with nearest sampling.
    ``trace`` (a list) receives ``(step_id, description, image, labels)`` for the input and after every applied
    step - used by the visual pipeline builder. It never changes the random stream or the result.
    """

    def record(step: str, text: str) -> None:
        if trace is not None:
            trace.append((step, text, np.array(image, dtype=np.float32), np.array(labels)))
    if image.shape[:2] != labels.shape:
        raise ValueError(f"Image {image.shape[:2]} and labels {labels.shape} are not aligned.")
    ops: list[str] = []
    # Draw every random number unconditionally so the stream is stable when options change.
    u = rng.random(5)
    k = int(rng.integers(1, 4))
    angle = float(rng.uniform(-cfg.rotate_max_deg, cfg.rotate_max_deg))
    b, c, g = rng.uniform(-1.0, 1.0, size=3)
    noise_seed = int(rng.integers(0, 2**31 - 1))
    record("input", "input")

    rotate = bool(cfg.rotate_prob) and u[3] < cfg.rotate_prob and angle != 0.0
    if rotate and crop:
        # Rotate only a window that contains the rotated patch (side >= patch * sqrt(2)), then take its centre:
        # same geometry as rotating the whole image, ~20x less work on large micrographs, and no rotation
        # corners inside the patch.
        p = cfg.patch_size
        window = int(np.ceil(p * np.sqrt(2.0))) + 2
        image, labels = random_crop(image, labels, window, rng)
        image, labels = _rotate(image, labels, angle, image_resample)
        o = (window - p) // 2
        image, labels = image[o:o + p, o:o + p], labels[o:o + p, o:o + p]
        ops.append(f"rotate {angle:+.1f} deg")
        record("rotate", f"rotate {angle:+.1f} deg + crop {p} px")
    elif rotate:
        image, labels = _rotate(image, labels, angle, image_resample)
        ops.append(f"rotate {angle:+.1f} deg")
        record("rotate", f"rotate {angle:+.1f} deg")
    elif crop:
        image, labels = random_crop(image, labels, cfg.patch_size, rng)
        record("crop", f"crop {cfg.patch_size} px")
    if u[0] < cfg.hflip:
        image, labels = image[:, ::-1], labels[:, ::-1]
        ops.append("flip left-right")
        record("hflip", "flip left-right")
    if u[1] < cfg.vflip:
        image, labels = image[::-1], labels[::-1]
        ops.append("flip up-down")
        record("vflip", "flip up-down")
    if u[2] < cfg.rot90:
        image, labels = np.rot90(image, k, axes=(0, 1)), np.rot90(labels, k, axes=(0, 1))
        ops.append(f"rotate {90 * k} deg")
        record("rot90", f"rotate {90 * k} deg")
    if u[4] < cfg.photometric_prob:
        if cfg.contrast:
            factor = 1.0 + c * cfg.contrast
            mean = float(image.mean())
            image = (image - mean) * factor + mean
            ops.append(f"contrast x{factor:.2f}")
            record("contrast", ops[-1])
        if cfg.brightness:
            image = image + b * cfg.brightness
            ops.append(f"brightness {b * cfg.brightness:+.3f}")
            record("brightness", ops[-1])
        if cfg.gamma:
            gamma = float(np.exp(g * cfg.gamma))
            image = np.clip(image, 0.0, 1.0) ** gamma
            ops.append(f"gamma {gamma:.2f}")
            record("gamma", ops[-1])
        if cfg.noise_std:
            image = image + np.random.default_rng(noise_seed).normal(0, cfg.noise_std, image.shape)
            ops.append(f"noise sd {cfg.noise_std:.3f}")
            record("noise_std", ops[-1])
        if cfg.blur_max_sigma:
            sigma = float(np.random.default_rng([noise_seed, 1]).uniform(0, cfg.blur_max_sigma))
            image = gaussian_blur(image, sigma)
            ops.append(f"blur sigma {sigma:.2f}")
            record("blur_max_sigma", ops[-1])
        image = np.clip(image, 0.0, 1.0)
    for k, op in enumerate(cfg.extra_ops):  # extra draws only when extra ops are configured
        op_rng = np.random.default_rng([noise_seed, 100 + k])
        if op_rng.random() < float(op.get("probability", 0)):
            spec = PHOTOMETRIC_OPS[op["name"]]
            params = {**spec["defaults"], **(op.get("params") or {})}
            out = spec["fn"](image, op_rng, params)
            if out.shape != image.shape:
                raise AugmentConfigError(f"Augmentation {op['name']} changed the image shape.")
            image = np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)
            ops.append(op["name"])
            record(f"extra:{op['name']}", op["name"])
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(labels), ops
