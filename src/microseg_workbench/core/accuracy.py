"""Accuracy details beyond pixel overlap: boundary agreement, area (phase) fractions and per-image scores.

Definitions (all on evaluated pixels, i.e. ground-truth pixels that are not ignored, at original resolution):

* **Boundary pixels** of a class are pixels of that class with a 4-neighbour of another class. Image edges are
  not boundaries, and pixels next to ignored ground truth are left out of boundary counts.
* **Boundary precision / recall / F1** (per class, tolerance ``t`` px): a predicted boundary pixel is matched when
  a true boundary pixel of the same class lies within ``t`` px (square neighbourhood); recall likewise for true
  boundary pixels. Counts are summed over all images. F1 is ``None`` when neither truth nor prediction has a
  boundary of that class; it is 0 when only one of them has.
* **Area fraction** of a class: its share of the evaluated pixels, in the ground truth and in the prediction.
  ``area_fraction_error`` = predicted - true (over all images); ``per_image_area_fraction_mae`` is the mean
  absolute per-image difference.
* **Per-image scores**: pixel accuracy, IoU per class and mean IoU of the feature classes (all classes except the
  first) for each image; listed worst first.

Ratios with a zero denominator are ``None`` and excluded from means, as in :mod:`.metrics`.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np

from .augment import IGNORE_INDEX
from .metrics import confusion_matrix, metrics_from_confusion

BOUNDARY_TOLERANCE_PX = 2


def boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of ``mask`` with a 4-neighbour outside it (image edges are not boundaries)."""
    p = np.pad(mask, 1, mode="edge")
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1) x (2r+1) square (separable, no SciPy needed)."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for axis in (0, 1):
        src = out.copy()
        for d in range(1, radius + 1):
            if axis == 0:
                out[d:] |= src[:-d]
                out[:-d] |= src[d:]
            else:
                out[:, d:] |= src[:, :-d]
                out[:, :-d] |= src[:, d:]
    return out


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None and r is None:
        return None
    if p is None or r is None or p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


class AccuracyAccumulator:
    """Add (truth, prediction) pairs of class indices image by image; :meth:`result` gives the summary."""

    def __init__(self, class_names: list[str], tolerance: int = BOUNDARY_TOLERANCE_PX) -> None:
        self.names = list(class_names)
        self.k = len(class_names)
        self.tol = tolerance
        self.counts = np.zeros((self.k, 4), np.int64)  # predicted bnd, matched predicted, true bnd, matched true
        self.images: list[dict[str, Any]] = []

    def add(self, stem: str, truth: np.ndarray, pred: np.ndarray) -> None:
        if truth.shape != pred.shape:
            raise ValueError(f"{stem}: truth {truth.shape} and prediction {pred.shape} differ in size.")
        valid = truth != IGNORE_INDEX
        countable = ~dilate(~valid, 1)  # not next to ignored ground truth
        n = int(valid.sum())
        true_frac, pred_frac = {}, {}
        for c, name in enumerate(self.names):
            t, p = (truth == c) & valid, (pred == c) & valid
            true_frac[name] = float(t.sum()) / n if n else None
            pred_frac[name] = float(p.sum()) / n if n else None
            tb, pb = boundary(t) & countable, boundary(p) & countable
            if tb.any() or pb.any():
                self.counts[c] += (int(pb.sum()), int((pb & dilate(tb, self.tol)).sum()),
                                   int(tb.sum()), int((tb & dilate(pb, self.tol)).sum()))
        m = metrics_from_confusion(confusion_matrix(truth, pred, self.k), self.names)
        self.images.append({
            "stem": stem, "evaluated_pixels": n, "pixel_accuracy": m["pixel_accuracy"],
            "mean_iou_foreground": m["mean_iou_foreground"],
            "iou": {c["name"]: c["iou"] for c in m["per_class"]},
            "area_fraction_true": true_frac, "area_fraction_pred": pred_frac,
        })

    def result(self, confusion: list[list[int]] | None = None) -> dict[str, Any]:
        total = sum(i["evaluated_pixels"] for i in self.images)
        per_class = []
        for c, name in enumerate(self.names):
            pb, mp, tb, mt = (int(v) for v in self.counts[c])
            bp = mp / pb if pb else None
            br = mt / tb if tb else None
            true_px = sum(i["area_fraction_true"][name] * i["evaluated_pixels"] for i in self.images if i["evaluated_pixels"])
            pred_px = sum(i["area_fraction_pred"][name] * i["evaluated_pixels"] for i in self.images if i["evaluated_pixels"])
            errs = [abs(i["area_fraction_pred"][name] - i["area_fraction_true"][name]) for i in self.images
                    if i["evaluated_pixels"]]
            at = true_px / total if total else None
            ap = pred_px / total if total else None
            per_class.append({
                "name": name, "boundary_precision": bp, "boundary_recall": br, "boundary_f1": _f1(bp, br),
                "true_boundary_pixels": tb, "predicted_boundary_pixels": pb,
                "area_fraction_true": at, "area_fraction_pred": ap,
                "area_fraction_error": (ap - at) if at is not None and ap is not None else None,
                "per_image_area_fraction_mae": float(np.mean(errs)) if errs else None,
            })
        fg = [c["boundary_f1"] for c in per_class[1:] if c["boundary_f1"] is not None]
        worst = sorted(self.images, key=lambda i: (i["mean_iou_foreground"] is None,
                                                    i["mean_iou_foreground"] if i["mean_iou_foreground"] is not None else 0))
        out: dict[str, Any] = {
            "boundary_tolerance_px": self.tol,
            "per_class": per_class,
            "mean_boundary_f1_foreground": float(np.mean(fg)) if fg else None,
            "per_image": worst,
            "definitions": "boundary F1 within %d px; area fractions of evaluated pixels; per-image scores worst "
                           "first (see microseg_workbench.core.accuracy)" % self.tol,
        }
        if confusion is not None:
            cm = np.asarray(confusion, np.float64)
            rows = cm.sum(axis=1)
            out["confusion_row_normalized"] = [[float(v / rows[i]) if rows[i] else None for v in cm[i]]
                                               for i in range(len(cm))]
        return out


def per_image_csv(accuracy: dict[str, Any], class_names: list[str]) -> str:
    """Per-image scores and area fractions as CSV text (opens in Excel)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["image", "evaluated_pixels", "pixel_accuracy", "feature_mean_iou"]
               + [f"iou_{n}" for n in class_names]
               + [f"true_area_%_{n}" for n in class_names] + [f"predicted_area_%_{n}" for n in class_names])

    def f(v: float | None, scale: float = 1.0) -> str:
        return "" if v is None else f"{v * scale:.6g}"

    for i in accuracy["per_image"]:
        w.writerow([i["stem"], i["evaluated_pixels"], f(i["pixel_accuracy"]), f(i["mean_iou_foreground"])]
                   + [f(i["iou"].get(n)) for n in class_names]
                   + [f(i["area_fraction_true"].get(n), 100) for n in class_names]
                   + [f(i["area_fraction_pred"].get(n), 100) for n in class_names])
    return buf.getvalue()
