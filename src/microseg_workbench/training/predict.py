"""Applying a trained model to new, unlabelled micrographs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import TOOL_ID, __version__
from ..core.atomic import publish_dir, sha256_file, staging_dir, write_bytes_atomic, write_json_atomic
from ..core.classes import ClassMap
from ..core.imageio import IMAGE_EXTENSIONS, MASK_SUFFIX, ImageReadError, encode_png, load_image, to_unit_float
from ..core.preprocess import Preprocess
from ..core.render import overlay, thumbnail, tile_grid, to_display_rgb
from .device import choose_device
from .engine import load_checkpoint, predict_labels
from .runs import RunDir, utc_now


class PredictionError(ValueError):
    pass


def predict_folder(run: RunDir, input_folder: Path, output_folder: Path, checkpoint: str = "best",
                   device: str = "auto", progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    """Predict every image in ``input_folder`` (masks named ``*_mask`` are skipped).

    Writes ``<stem>_pred_mask.png`` (class values from the class map), ``<stem>_overlay.png`` and
    ``predictions.json`` into ``output_folder``, which is published atomically and must not exist yet.
    """
    ckpt_path = run.checkpoints / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        raise PredictionError(f"Run {run.run_id} has no {checkpoint} checkpoint.")
    if output_folder.exists():
        raise PredictionError(f"Output folder {output_folder} already exists; choose a new folder name.")
    images = sorted(p for p in Path(input_folder).iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.stem.endswith(MASK_SUFFIX))
    if not images:
        raise PredictionError(f"No images found in {input_folder}.")
    by_name: dict[str, list[str]] = {}
    for p in images:
        by_name.setdefault(p.stem.lower(), []).append(p.name)
    clashes = sorted(", ".join(v) for v in by_name.values() if len(v) > 1)
    if clashes:
        raise PredictionError("Some images share a name and would overwrite each other's results: "
                              + "; ".join(clashes[:5]) + ". Rename or move them, then try again.")
    torch_device, device_info, _ = choose_device(device)
    model, meta = load_checkpoint(ckpt_path, torch_device)
    class_map = ClassMap.from_dict(meta["class_map"])
    prep = Preprocess.from_meta(meta)
    values = np.asarray(class_map.values, dtype=np.uint8)
    stage = staging_dir(output_folder.parent, output_folder.name)
    records, errors = [], []
    try:
        for i, path in enumerate(images, 1):
            if progress:
                progress(i, len(images), path.name)
            try:
                loaded = load_image(path)
            except ImageReadError as exc:
                errors.append({"file": path.name, "error": str(exc)})
                continue
            if loaded.channels != model.cfg.in_channels:
                errors.append({"file": path.name, "error": f"has {loaded.channels} channel(s); the model expects "
                                                           f"{model.cfg.in_channels}"})
                continue
            try:
                region = prep.analysed_region(*loaded.array.shape[:2])
            except ValueError as exc:
                errors.append({"file": path.name, "error": str(exc)})
                continue
            image = prep.crop(to_unit_float(loaded.array))
            labels = predict_labels(model, image, torch_device, prep)
            write_bytes_atomic(stage / f"{path.stem}_pred_mask.png",
                               encode_png(values[labels]))
            write_bytes_atomic(stage / f"{path.stem}_overlay.png",
                               encode_png(tile_grid([thumbnail(to_display_rgb(image), 1024),
                                                     thumbnail(overlay(image, labels, class_map), 1024)], 2)))
            counts = np.bincount(labels.ravel(), minlength=class_map.num_classes)
            records.append({"file": path.name, "image_sha256": sha256_file(path),
                            "image_height": int(loaded.array.shape[0]), "image_width": int(loaded.array.shape[1]),
                            "analysed_region": region, "height": int(labels.shape[0]), "width": int(labels.shape[1]),
                            "class_fractions": {c.name: round(float(counts[k]) / labels.size, 6)
                                                for k, c in enumerate(class_map.classes)}})
        report = {
            "schema": "microseg-workbench.predictions/1",
            "tool": {"id": TOOL_ID, "version": __version__},
            "run_id": run.run_id,
            "checkpoint": f"checkpoints/{checkpoint}.pt",
            "checkpoint_sha256": sha256_file(ckpt_path),
            "checkpoint_epoch": meta.get("epoch"),
            "dataset_id": meta.get("dataset_id"),
            "class_map": class_map.to_dict(),
            "preprocess": prep.to_dict(),
            "device": device_info.to_dict(),
            "input_folder": str(input_folder),
            "created_at": utc_now(),
            "review_status": "unreviewed",
            "images": records,
            "errors": errors,
        }
        write_json_atomic(stage / "predictions.json", report)
        publish_dir(stage, output_folder)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report
