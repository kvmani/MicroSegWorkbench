"""Observations (notes) and the model card: everything needed to understand, trust and re-use a trained model."""

from __future__ import annotations

import json
from typing import Any

from .. import TOOL_ID, __version__
from ..core.atomic import sha256_file, write_json_atomic
from .runs import RunDir, utc_now

MODEL_CARD_SCHEMA = "microseg-workbench.model-card/1"
NOTES_FILE = "notes.jsonl"


class NoteError(ValueError):
    pass


def add_note(run: RunDir, text: str, author: str, kind: str = "observation") -> dict[str, Any]:
    """Append an observation to the run (notes are never edited or deleted; add a new one to correct)."""
    if not text.strip():
        raise NoteError("Write the observation first.")
    note = {"time": utc_now(), "author": author, "kind": kind, "text": text.strip()}
    with open(run.root / NOTES_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(note, sort_keys=True) + "\n")
    if (run.checkpoints / "best.pt").exists():
        write_model_card(run)
    return note


def read_notes(run: RunDir) -> list[dict[str, Any]]:
    path = run.root / NOTES_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_model_card(run: RunDir) -> dict[str, Any]:
    info = run.info()
    cfg = info.get("config", {})
    eval_path = run.evaluation / "evaluation.json"
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    metrics = evaluation.get("metrics") or {}
    diag_path = run.evaluation / "diagnostics.json"
    diagnostics = json.loads(diag_path.read_text())["findings"] if diag_path.exists() else []
    best = run.checkpoints / "best.pt"
    dataset = info.get("dataset", {})
    manifest_path = run.root.parent.parent / "datasets" / str(dataset.get("dataset_id")) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "schema": MODEL_CARD_SCHEMA,
        "run_id": run.run_id,
        "card_written_at": utc_now(),
        "card_written_with": f"{TOOL_ID} {__version__}",
        "status": run.state(),
        "review_status": evaluation.get("review_status", "unreviewed"),
        "network": {"architecture": info.get("model", {}).get("architecture"),
                    "title": info.get("model", {}).get("architecture_title"),
                    "defined_in": info.get("model", {}).get("architecture_module"),
                    "config": {k: info.get("model", {}).get(k) for k in ("in_channels", "num_classes",
                                                                         "base_channels", "depth")},
                    "parameters": info.get("model", {}).get("parameters")},
        "classes": info.get("class_map"),
        "input": {"channels": info.get("model", {}).get("in_channels"),
                  "normalization": cfg.get("normalization"), "resize_factor": cfg.get("resize_factor"),
                  "exclude_border": cfg.get("exclude_border"),
                  "note": "Images are scaled by their bit depth, then preprocessed as above; predictions are "
                          "returned at the original resolution for the analysed region."},
        "training_data": {"dataset_id": dataset.get("dataset_id"), "manifest_sha256": dataset.get("manifest_sha256"),
                          "split_counts": dataset.get("split_counts"),
                          "annotation_source": manifest.get("annotation_source"),
                          "contains_unapproved_annotations": manifest.get("contains_unapproved_annotations"),
                          "split": manifest.get("split_config")},
        "training": {"config": cfg, "resolved": info.get("resolved"), "device": info.get("device"),
                     "environment": info.get("environment"), "tool": info.get("tool"),
                     "plugins": info.get("plugins"), "resumes": info.get("resumes", []),
                     "epochs_completed": len(run.epoch_history()), "stopped_early": run.status().get("stopped_early")},
        "results": {"split": evaluation.get("split"), "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
                    "mean_iou": metrics.get("mean_iou"), "mean_iou_foreground": metrics.get("mean_iou_foreground"),
                    "pixel_accuracy": metrics.get("pixel_accuracy"),
                    "per_class": [{k: c[k] for k in ("name", "iou", "dice", "precision", "recall", "true_pixels")}
                                  for c in metrics.get("per_class", [])],
                    "ground_truth": evaluation.get("ground_truth")},
        "diagnostics": diagnostics,
        "observations": read_notes(run),
        "checkpoint": {"file": "checkpoints/best.pt", "sha256": sha256_file(best) if best.exists() else None},
        "how_to_use": f"msw predict {run.run_id} <folder of images> --out <new folder>  (or step 4 > Segment new "
                      "images in the app)",
    }


def write_model_card(run: RunDir) -> dict[str, Any]:
    card = build_model_card(run)
    write_json_atomic(run.root / "model_card.json", card)
    return card
