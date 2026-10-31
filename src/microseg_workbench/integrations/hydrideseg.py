"""Optional adapter to HydrideSegmentation's published scientific analysis functions.

Reused (not copied) functions, called through this explicit contract:

* ``src.microseg.evaluation.hydride_statistics.compute_hydride_statistics`` + ``statistics_to_json`` -
  per-feature orientation, size, length, alignment index and histograms of a binary hydride mask;
* ``src.microseg.evaluation.hydride_metrics.scientific_distance_metrics`` - area-fraction and count errors and
  Wasserstein/KS distances between ground-truth and predicted size and orientation distributions.

HydrideSegmentation is found if it is importable (installed) or if ``MSW_HYDRIDESEG_PATH`` names its checkout.
It needs scipy, scikit-image and matplotlib. Without it, these analyses report "not available"; nothing else in
the workbench depends on it. Results are labelled with the HydrideSegmentation version used.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..core.atomic import write_json_atomic
from ..core.classes import ClassMap
from ..core.dataset import DatasetVersion
from ..core.imageio import load_mask_values

ENV = "MSW_HYDRIDESEG_PATH"


class HydrideSegUnavailable(RuntimeError):
    pass


def _load() -> tuple[Any, Any, str]:
    path = os.environ.get(ENV)
    if path and path not in sys.path:
        sys.path.append(path)  # appended, so it never shadows the workbench's own modules
    try:
        from src.microseg.evaluation import hydride_metrics, hydride_statistics  # type: ignore[import-not-found]
    except ImportError as exc:
        hint = f" (looked in {path})" if path else f"; set {ENV} to the HydrideSegmentation checkout"
        raise HydrideSegUnavailable(f"HydrideSegmentation analysis is not available{hint}: {exc}") from exc
    version = "unknown"
    try:
        from src.microseg import version as hv  # type: ignore[import-not-found]

        version = str(getattr(hv, "__version__", getattr(hv, "VERSION", "unknown")))
    except ImportError:
        pass
    return hydride_statistics, hydride_metrics, version


def availability() -> tuple[bool, str]:
    try:
        _, _, version = _load()
    except HydrideSegUnavailable as exc:
        return False, str(exc)
    return True, f"HydrideSegmentation {version}"


def feature_statistics(binary: np.ndarray, microns_per_pixel: float | None = None,
                       min_feature_pixels: int = 1) -> dict[str, Any]:
    stats_mod, _, version = _load()
    result = stats_mod.compute_hydride_statistics(binary.astype(np.uint8), microns_per_pixel=microns_per_pixel,
                                                  min_feature_pixels=min_feature_pixels)
    payload = stats_mod.statistics_to_json(result, decimal_places=6)
    payload["source"] = {"package": "HydrideSegmentation", "version": version,
                         "function": "evaluation.hydride_statistics.compute_hydride_statistics"}
    return payload


def distribution_distances(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    _, metrics_mod, _ = _load()
    return metrics_mod.scientific_distance_metrics(truth.astype(np.uint8), pred.astype(np.uint8))


def _class_value(class_map: ClassMap, class_name: str) -> int:
    for c in class_map.classes:
        if c.name == class_name:
            return c.value
    raise ValueError(f"Class {class_name!r} is not in this model's classes {class_map.names}.")


def analyse_run(run_root: Path, datasets_dir: Path, class_name: str = "hydride",
                microns_per_pixel: float | None = None) -> dict[str, Any]:
    """Compare ground truth and prediction on the run's evaluation images for one class; writes a JSON report."""
    _, _, version = _load()
    evaluation = json.loads((run_root / "evaluation" / "evaluation.json").read_text())
    dataset = DatasetVersion.open(datasets_dir / evaluation["dataset_id"])
    class_map = dataset.class_map
    value = _class_value(class_map, class_name)
    samples = {s["stem"]: s for s in dataset.samples()}
    per_image, all_truth, all_pred = {}, [], []
    for stem in evaluation["images"]:
        truth = load_mask_values(dataset.mask_path(samples[stem])) == value
        pred = np.asarray(Image.open(run_root / "evaluation" / f"{stem}_pred_mask.png")) == value
        per_image[stem] = distribution_distances(truth, pred)
        all_truth.append(truth)
        all_pred.append(pred)
    report = {
        "schema": "microseg-workbench.hydride-analysis/1",
        "run_id": run_root.name,
        "dataset_id": dataset.dataset_id,
        "split": evaluation["split"],
        "class": class_name,
        "microns_per_pixel": microns_per_pixel,
        "per_image_distances": per_image,
        "truth_statistics": [feature_statistics(t, microns_per_pixel)["scalar_metrics"] for t in all_truth],
        "prediction_statistics": [feature_statistics(p, microns_per_pixel)["scalar_metrics"] for p in all_pred],
        "computed_with": f"HydrideSegmentation {version}",
        "review_status": "unreviewed",
    }
    write_json_atomic(run_root / "evaluation" / f"hydride_analysis_{class_name}.json", report)
    return report


def analyse_predictions(folder: Path, class_name: str = "hydride",
                        microns_per_pixel: float | None = None) -> dict[str, Any]:
    """Feature statistics of one class for every ``*_pred_mask.png`` in a ``msw predict`` output folder."""
    _, _, version = _load()
    meta = json.loads((folder / "predictions.json").read_text())
    value = _class_value(ClassMap.from_dict(meta["class_map"]), class_name)
    per_image = {}
    for rec in meta["images"]:
        stem = rec["file"].rsplit(".", 1)[0]
        mask = np.asarray(Image.open(folder / f"{stem}_pred_mask.png")) == value
        per_image[rec["file"]] = feature_statistics(mask, microns_per_pixel)
    report = {"schema": "microseg-workbench.hydride-statistics/1", "run_id": meta["run_id"], "class": class_name,
              "microns_per_pixel": microns_per_pixel, "images": per_image,
              "computed_with": f"HydrideSegmentation {version}", "review_status": "unreviewed"}
    write_json_atomic(folder / f"hydride_statistics_{class_name}.json", report)
    return report
