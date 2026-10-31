"""Preprocessing applied identically in training, evaluation and prediction.

* ``resize_factor`` scales images before the model sees them (e.g. 0.5 halves width and height, useful for
  very large micrographs or to give the model more context). Images use area averaging when shrinking and
  bilinear interpolation when enlarging; masks use nearest neighbour so no new class values appear (very thin
  features can disappear when shrinking - check the preview). Evaluation and predictions are always mapped back
  to the ORIGINAL resolution and scored against the original masks.
* ``exclude_border`` (top, bottom, left, right) pixels at original resolution are removed before anything else -
  e.g. the information bar of SEM images. Excluded pixels are never trained on or scored; predicted masks cover only
  the analysed region and record its position.
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
    exclude_border: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, bottom, left, right (pixels)

    def __post_init__(self) -> None:
        border = tuple(int(v) for v in self.exclude_border)
        if len(border) != 4 or min(border) < 0:
            raise PreprocessError("exclude_border needs four non-negative pixel counts (top, bottom, left, right).")
        object.__setattr__(self, "exclude_border", border)
        if not 0.1 <= self.resize_factor <= 4.0:
            raise PreprocessError("resize factor must be between 0.1 and 4.")
        if self.normalization not in NORMALIZATIONS:
            raise PreprocessError(f"normalization must be one of {NORMALIZATIONS}.")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["exclude_border"] = list(self.exclude_border)
        return out

    def analysed_region(self, height: int, width: int) -> dict[str, int]:
        top, bottom, left, right = self.exclude_border
        h, w = height - top - bottom, width - left - right
        if h < 8 or w < 8:
            raise PreprocessError(f"Excluding borders {self.exclude_border} leaves only {max(h, 0)}x{max(w, 0)} "
                                  f"pixels of a {height}x{width} image.")
        return {"top": top, "left": left, "height": h, "width": w}

    def crop(self, arr: np.ndarray) -> np.ndarray:
        """Remove the excluded border from an (H, W) or (H, W, C) array."""
        if not any(self.exclude_border):
            return arr
        r = self.analysed_region(arr.shape[0], arr.shape[1])
        return arr[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]]

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "Preprocess":
        """From a checkpoint (older checkpoints only recorded ``normalization``)."""
        payload = dict(meta.get("preprocess") or {"normalization": meta.get("normalization", "dtype_range")})
        if "exclude_border" in payload:
            payload["exclude_border"] = tuple(payload["exclude_border"])
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
