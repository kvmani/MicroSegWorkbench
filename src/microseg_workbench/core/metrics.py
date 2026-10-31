"""Segmentation metrics from an accumulated confusion matrix.

Definitions (per class c, counts over all evaluated pixels, ignore pixels excluded):

* IoU_c   = TP / (TP + FP + FN)
* Dice_c  = 2 TP / (2 TP + FP + FN)
* precision_c = TP / (TP + FP), recall_c = TP / (TP + FN)
* pixel accuracy = trace / total

A ratio whose denominator is zero (class absent from both truth and prediction)
is reported as ``None`` and excluded from means; it is never counted as 0 or 1.
Means are unweighted over classes (macro). ``mean_iou_foreground`` excludes the
first class (background).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .augment import IGNORE_INDEX


def confusion_matrix(truth: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``cm[t, p]`` counts pixels with true class t and predicted class p."""
    if truth.shape != pred.shape:
        raise ValueError(f"Shapes differ: truth {truth.shape} vs prediction {pred.shape}.")
    t = truth.astype(np.int64).ravel()
    p = pred.astype(np.int64).ravel()
    keep = t != IGNORE_INDEX
    t, p = t[keep], p[keep]
    if t.size and (t.max() >= num_classes or t.min() < 0):
        raise ValueError("Truth contains class indices outside the model's classes.")
    if p.size and (p.max() >= num_classes or p.min() < 0):
        raise ValueError("Prediction contains class indices outside the model's classes.")
    return np.bincount(t * num_classes + p, minlength=num_classes**2).reshape(num_classes, num_classes)


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def metrics_from_confusion(cm: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    k = cm.shape[0]
    if cm.shape != (k, k) or len(class_names) != k:
        raise ValueError("Confusion matrix must be square and match the class names.")
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    per_class: list[dict[str, Any]] = []
    for i, name in enumerate(class_names):
        per_class.append({
            "index": i,
            "name": name,
            "iou": _ratio(tp[i], tp[i] + fp[i] + fn[i]),
            "dice": _ratio(2 * tp[i], 2 * tp[i] + fp[i] + fn[i]),
            "precision": _ratio(tp[i], tp[i] + fp[i]),
            "recall": _ratio(tp[i], tp[i] + fn[i]),
            "true_pixels": int(tp[i] + fn[i]),
            "predicted_pixels": int(tp[i] + fp[i]),
        })
    total = int(cm.sum())
    return {
        "pixel_accuracy": _ratio(tp.sum(), total),
        "mean_iou": _mean([c["iou"] for c in per_class]),
        "mean_iou_foreground": _mean([c["iou"] for c in per_class[1:]]),
        "mean_dice": _mean([c["dice"] for c in per_class]),
        "evaluated_pixels": total,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
