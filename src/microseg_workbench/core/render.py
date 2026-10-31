"""Rendering overlays and preview grids as RGB uint8 arrays (UI and reports)."""

from __future__ import annotations

import numpy as np

from .augment import IGNORE_INDEX, AugmentConfig, augment
from .classes import ClassMap


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Float ``(H, W, C)`` in [0, 1] -> uint8 RGB."""
    img = np.clip(image, 0.0, 1.0)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return np.rint(img[:, :, :3] * 255).astype(np.uint8)


def overlay(image: np.ndarray, channel_labels: np.ndarray, class_map: ClassMap, alpha: float = 0.45) -> np.ndarray:
    """Blend class colours (by model channel index) over the image; ignore pixels hatched grey."""
    rgb = to_display_rgb(image).astype(np.float32)
    for idx, cls in enumerate(class_map.classes):
        if idx == 0:
            continue
        sel = channel_labels == idx
        rgb[sel] = (1 - alpha) * rgb[sel] + alpha * np.array(cls.rgb(), dtype=np.float32)
    ignore = channel_labels == IGNORE_INDEX
    if ignore.any():
        yy, xx = np.nonzero(ignore)
        rgb[yy, xx] = np.where(((yy + xx) % 4 < 2)[:, None], 90.0, 140.0)
    return np.rint(rgb).astype(np.uint8)


def label_colors(channel_labels: np.ndarray, class_map: ClassMap) -> np.ndarray:
    out = np.zeros((*channel_labels.shape, 3), dtype=np.uint8)
    for idx, cls in enumerate(class_map.classes):
        out[channel_labels == idx] = cls.rgb()
    out[channel_labels == IGNORE_INDEX] = (128, 128, 128)
    return out


def thumbnail(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink an RGB display image so its longer side is at most ``max_side`` (display only, never for analysis)."""
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return rgb
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BOX))


def tile_grid(tiles: list[np.ndarray], columns: int, gap: int = 4, background: int = 255) -> np.ndarray:
    if not tiles:
        raise ValueError("No tiles to arrange.")
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + columns - 1) // columns
    grid = np.full((rows * h + (rows - 1) * gap, columns * w + (columns - 1) * gap, 3), background, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        grid[r * (h + gap): r * (h + gap) + t.shape[0], c * (w + gap): c * (w + gap) + t.shape[1]] = t
    return grid


def augmentation_preview(image: np.ndarray, channel_labels: np.ndarray, class_map: ClassMap, cfg: AugmentConfig,
                         seed: int, count: int = 6) -> tuple[np.ndarray, list[list[str]]]:
    """Two-row grid per variant (image above, overlay below). Column 0 is the unaugmented sample."""
    tops, bottoms, captions = [to_display_rgb(image)], [overlay(image, channel_labels, class_map)], [["original"]]
    for i in range(count):
        rng = np.random.default_rng([seed, i])
        aug_img, aug_lab, ops = augment(image, channel_labels, cfg, rng, crop=False)
        tops.append(to_display_rgb(aug_img))
        bottoms.append(overlay(aug_img, aug_lab, class_map))
        captions.append(ops or ["no change"])
    return tile_grid(tops + bottoms, columns=len(tops)), captions
