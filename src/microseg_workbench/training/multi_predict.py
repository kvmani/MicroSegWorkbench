"""Run several trained models on the same image and compare their predictions (and with ground truth if given)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.classes import ClassMap
from ..core.imageio import load_image, load_mask_values, to_unit_float
from ..core.metrics import confusion_matrix, metrics_from_confusion
from ..core.preprocess import Preprocess
from .engine import load_checkpoint, predict_labels
from .runs import RunDir

NOT_ANALYSED = 255


class CompareError(ValueError):
    pass


def compare_models_on_image(runs: list[RunDir], image_path: Path, truth: np.ndarray | None = None,
                            truth_class_map: ClassMap | None = None, device: str = "cpu") -> dict[str, Any]:
    """Predict ``image_path`` with every run's best model.

    Returns per-model predictions as class values on the full image (``NOT_ANALYSED`` outside a model's analysed
    region), class fractions, timing, per-image scores against ``truth`` (class values) when the class maps
    match, pairwise agreement and a disagreement mask.
    """
    if not runs:
        raise CompareError("Choose at least one trained model.")
    loaded = load_image(Path(image_path))
    image = to_unit_float(loaded.array)
    h, w = image.shape[:2]
    torch_device = torch.device(device)
    models: list[dict[str, Any]] = []
    for run in runs:
        ckpt = run.checkpoints / "best.pt"
        if not ckpt.exists():
            raise CompareError(f"Run {run.run_id} has no trained model.")
        model, meta = load_checkpoint(ckpt, torch_device)
        if model.cfg.in_channels != loaded.channels:
            raise CompareError(f"{run.run_id} expects {model.cfg.in_channels}-channel images; this image has "
                               f"{loaded.channels}.")
        cmap = ClassMap.from_dict(meta["class_map"])
        prep = Preprocess.from_meta(meta)
        region = prep.analysed_region(h, w)
        start = time.perf_counter()
        labels = predict_labels(model, prep.crop(image), torch_device, prep)
        seconds = time.perf_counter() - start
        values = np.full((h, w), NOT_ANALYSED, np.uint8)
        values[region["top"]:region["top"] + region["height"], region["left"]:region["left"] + region["width"]] = \
            np.asarray(cmap.values, np.uint8)[labels]
        counts = np.bincount(labels.ravel(), minlength=cmap.num_classes)
        info = run.info()
        models.append({
            "run_id": run.run_id, "network": info.get("model", {}).get("architecture_title", model.cfg.architecture),
            "class_map": cmap, "values": values, "seconds": seconds, "checkpoint_epoch": meta.get("epoch"),
            "fractions": {c.name: float(counts[k]) / labels.size for k, c in enumerate(cmap.classes)},
        })
    analysed = np.all([m["values"] != NOT_ANALYSED for m in models], axis=0)
    if truth is not None:
        if truth.shape != (h, w):
            raise CompareError("The ground-truth mask does not have the image's size.")
        for m in models:
            if truth_class_map is None or m["class_map"].to_dict()["classes"] != truth_class_map.to_dict()["classes"]:
                m["metrics"] = None
                m["metrics_note"] = "model classes differ from the ground truth's; not scored"
                continue
            lut = m["class_map"].channel_lut()
            sel = analysed
            cm = confusion_matrix(lut[truth][sel], lut[m["values"]][sel], m["class_map"].num_classes)
            m["metrics"] = metrics_from_confusion(cm, m["class_map"].names)
    same_classes = len({tuple(c.name for c in m["class_map"].classes) for m in models}) == 1
    agreement = None
    disagreement = None
    if same_classes and len(models) > 1:
        agreement = [[float((a["values"] == b["values"])[analysed].mean()) if analysed.any() else None
                      for b in models] for a in models]
        stack = np.stack([m["values"] for m in models])
        disagreement = (stack != stack[0]).any(axis=0) & analysed
    return {"image": str(image_path), "height": h, "width": w, "image_array": image, "models": models,
            "analysed_by_all": analysed, "agreement": agreement, "disagreement": disagreement,
            "same_classes": same_classes}


def render_comparison(result: dict[str, Any], truth: np.ndarray | None = None) -> list[tuple[str, np.ndarray]]:
    """(caption, RGB image) tiles: image, ground truth, each model, disagreement map."""
    from ..core.render import overlay, to_display_rgb

    image = result["image_array"]
    tiles = [("image", to_display_rgb(image))]
    first = result["models"][0]["class_map"]
    if truth is not None:
        tiles.append(("ground truth", overlay(image, first.channel_lut()[truth], first)))
    for m in result["models"]:
        lut = m["class_map"].channel_lut()
        score = m.get("metrics") or {}
        fg = score.get("mean_iou_foreground")
        caption = f"{m['run_id']}\n{m['network']}" + (f"\nfeature IoU {fg:.3f}" if fg is not None else "")
        tiles.append((caption, overlay(image, lut[m["values"]], m["class_map"])))
    if result["disagreement"] is not None:
        rgb = (to_display_rgb(image).astype(np.float32) * 0.5).astype(np.uint8)
        rgb[result["disagreement"]] = (235, 104, 52)
        share = float(result["disagreement"][result["analysed_by_all"]].mean()) if result["analysed_by_all"].any() else 0
        tiles.append((f"where models disagree\n{100 * share:.1f}% of pixels", rgb))
    return tiles


def load_truth(mask_path: Path, class_map: ClassMap) -> np.ndarray:
    from ..core.imageio import decode_mask

    return decode_mask(load_mask_values(mask_path), class_map)
