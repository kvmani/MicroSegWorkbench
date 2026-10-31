"""Preprocessing applied identically in training, evaluation and prediction.

* ``resize_factor`` scales images before the model sees them (e.g. 0.5 halves width and height, useful for
  very large micrographs or to give the model more context). Images use area averaging when shrinking and
  bilinear interpolation when enlarging; masks use nearest neighbour so no new class values appear (very thin
  features can disappear when shrinking - check the preview). Evaluation and predictions are always mapped back
  to the ORIGINAL resolution and scored against the original masks.
* ``normalization``: ``dtype_range`` divides by the bit-depth maximum (default); ``per_image_standardize``
  subtracts each image's mean and divides by its standard deviation (per channel), which removes overall
  brightness/contrast differences between acquisitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

NORMALIZATIONS = ("dtype_range", "per_image_standardize")
STD_FLOOR = 1e-3


class PreprocessError(ValueError):
    pass


@dataclass(frozen=True)
class Preprocess:
    resize_factor: float = 1.0
    normalization: str = "dtype_range"

    def __post_init__(self) -> None:
        if not 0.1 <= self.resize_factor <= 4.0:
            raise PreprocessError("resize factor must be between 0.1 and 4.")
        if self.normalization not in NORMALIZATIONS:
            raise PreprocessError(f"normalization must be one of {NORMALIZATIONS}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "Preprocess":
        """From a checkpoint (older checkpoints only recorded ``normalization``)."""
        payload = meta.get("preprocess") or {"normalization": meta.get("normalization", "dtype_range")}
        return cls(**payload)

    def scaled_shape(self, height: int, width: int) -> tuple[int, int]:
        return max(1, round(height * self.resize_factor)), max(1, round(width * self.resize_factor))


def resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize float32 (H, W, C) to ``shape`` (area when shrinking, bilinear when enlarging)."""
    h, w = image.shape[:2]
    if (h, w) == tuple(shape):
        return image
    method = Image.Resampling.BOX if shape[0] < h else Image.Resampling.BILINEAR
    channels = [np.asarray(Image.fromarray(np.ascontiguousarray(image[..., c]), mode="F")
                           .resize((shape[1], shape[0]), method), dtype=np.float32) for c in range(image.shape[2])]
    return np.stack(channels, axis=2)


def resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == tuple(shape):
        return labels
    return np.asarray(Image.fromarray(labels, mode="L").resize((shape[1], shape[0]), Image.Resampling.NEAREST),
                      dtype=np.uint8)


def image_stats(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = image.reshape(-1, image.shape[2]).mean(0).astype(np.float32)
    std = np.maximum(image.reshape(-1, image.shape[2]).std(0), STD_FLOOR).astype(np.float32)
    return mean, std


def normalize(image: np.ndarray, method: str, stats: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Model input from a [0, 1] image. ``stats`` (from the whole image) keep patches consistent with inference."""
    if method == "dtype_range":
        return image
    if method == "per_image_standardize":
        mean, std = stats if stats is not None else image_stats(image)
        return ((image - mean) / std).astype(np.float32)
    raise PreprocessError(f"Unknown normalization {method!r}")
