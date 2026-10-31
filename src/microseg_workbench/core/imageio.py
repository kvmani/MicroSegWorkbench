"""Loading micrographs and masks without silently changing their content."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .classes import ClassMap

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
MASK_SUFFIX = "_mask"


class ImageReadError(ValueError):
    """A file could not be interpreted; the message is safe to show to users."""


@dataclass(frozen=True)
class LoadedImage:
    """Image as ``(H, W, C)`` array in its native dtype plus loading notes."""

    array: np.ndarray
    notes: tuple[str, ...]

    @property
    def channels(self) -> int:
        return int(self.array.shape[2])


def _open(path: Path) -> Image.Image:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as exc:  # PIL raises many exception types
        raise ImageReadError(f"Cannot read {path.name}: {exc}") from exc


def load_image(path: Path) -> LoadedImage:
    """Load a micrograph as ``(H, W, C)`` uint8/uint16/float32 with C in {1, 3}.

    RGB images whose three channels are identical are reduced to one channel
    (lossless; recorded in ``notes``). Alpha channels are rejected rather than dropped.
    """
    img = _open(path)
    notes: list[str] = []
    mode = img.mode
    if mode == "P":
        raise ImageReadError(f"{path.name} is a palette image; save the micrograph as grayscale or RGB.")
    if mode == "1":
        arr = np.asarray(img, dtype=np.uint8) * 255
    elif mode in ("L", "I;16", "I;16B", "I;16L", "I", "F", "RGB"):
        arr = np.asarray(img)
    elif mode in ("RGBA", "LA"):
        alpha = np.asarray(img)[..., -1]
        if alpha.min() < 255:
            raise ImageReadError(f"{path.name} has transparency; remove the alpha channel before import.")
        arr = np.asarray(img.convert("RGB" if mode == "RGBA" else "L"))
        notes.append("opaque_alpha_channel_dropped")
    else:
        raise ImageReadError(f"{path.name} has unsupported image mode {mode!r}.")
    if arr.dtype == np.int32:
        if arr.min() < 0 or arr.max() > 65535:
            raise ImageReadError(f"{path.name} has 32-bit integer values outside the 16-bit range.")
        arr = arr.astype(np.uint16)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
        arr = arr[..., :1]
        notes.append("identical_rgb_channels_reduced_to_grayscale")
    if arr.dtype not in (np.uint8, np.uint16, np.float32):
        raise ImageReadError(f"{path.name} has unsupported pixel type {arr.dtype}.")
    return LoadedImage(array=np.ascontiguousarray(arr), notes=tuple(notes))


def load_mask_values(path: Path) -> np.ndarray:
    """Raw 2-D mask pixel values (uint8). Rejects colour or palette-less ambiguity."""
    img = _open(path)
    if img.mode == "P":
        # Palette PNGs store indices; the index is the class value.
        arr = np.asarray(img)
    elif img.mode == "L":
        arr = np.asarray(img)
    elif img.mode == "1":
        arr = np.asarray(img, dtype=np.uint8) * 255
    elif img.mode in ("I;16", "I", "I;16B", "I;16L"):
        arr = np.asarray(img)
        if arr.max() > 255 or arr.min() < 0:
            raise ImageReadError(f"Mask {path.name} has values above 255; only 8-bit class values are supported.")
        arr = arr.astype(np.uint8)
    elif img.mode == "RGB":
        rgb = np.asarray(img)
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            arr = rgb[..., 0]
        else:
            raise ImageReadError(
                f"Mask {path.name} is a colour (RGB) image. Export masks as indexed (pixel value = class) "
                "or binary 0/255 PNGs."
            )
    else:
        raise ImageReadError(f"Mask {path.name} has unsupported mode {img.mode!r}.")
    return np.ascontiguousarray(arr.astype(np.uint8))


def decode_mask(raw: np.ndarray, class_map: ClassMap) -> np.ndarray:
    """Convert raw mask values to class values per the class map's encoding.

    Raises ``ImageReadError`` listing any value not declared in the class map.
    """
    if class_map.mask_encoding == "binary_0_255":
        bad = sorted(set(np.unique(raw).tolist()) - {0, 255})
        if bad:
            raise ImageReadError(f"binary 0/255 mask contains other values {bad[:10]}")
        return (raw == 255).astype(np.uint8)
    bad = sorted(set(np.unique(raw).tolist()) - set(class_map.values))
    if bad:
        raise ImageReadError(f"mask contains values {bad[:10]} not declared in the class map {class_map.values}")
    return raw.astype(np.uint8)


def encode_png(arr: np.ndarray) -> bytes:
    """Lossless PNG bytes for a 2-D uint8 or (H, W, 3) uint8 array."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def to_unit_float(arr: np.ndarray) -> np.ndarray:
    """Scale by the dtype range to float32 in [0, 1] (documented, deterministic)."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.float32:
        return arr
    raise ValueError(f"Unsupported dtype {arr.dtype}")
