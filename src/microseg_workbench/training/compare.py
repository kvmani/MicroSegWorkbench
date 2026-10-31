"""Side-by-side comparison of training runs (torch-free; reads run records only).

Scientific guard: test metrics are only directly comparable when runs were
evaluated on the same dataset version (same images, masks, class map and test
split). Otherwise the comparison carries an explicit warning and the table
groups runs by dataset version.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from .runs import RunDir


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def run_summary(run: RunDir) -> dict[str, Any]:
    info = run.info()
    cfg = info.get("config", {})
    history = run.epoch_history()
    evaluation = None
    path = run.evaluation / "evaluation.json"
    if path.exists():
        evaluation = json.loads(path.read_text(encoding="utf-8"))
    metrics = (evaluation or {}).get("metrics") or {}
    best = max((e for e in history if e.get("val_mean_iou_foreground") is not None),
               key=lambda e: e["val_mean_iou_foreground"], default=None)
    return {
        "run_id": run.run_id,
        "state": run.state(),
        "dataset_id": info.get("dataset", {}).get("dataset_id"),
        "class_names": [c["name"] for c in info.get("class_map", {}).get("classes", [])],
        "preset": cfg.get("preset"),
        "epochs_done": len(history),
        "epochs_planned": cfg.get("epochs"),
        "learning_rate": cfg.get("learning_rate"),
        "batch_size": info.get("resolved", {}).get("batch_size", cfg.get("batch_size")),
        "loss": cfg.get("loss", "cross_entropy"),
        "seed": cfg.get("seed"),
        "model": f"U-Net base {cfg.get('base_channels')} depth {cfg.get('depth')}",
        "parameters": info.get("model", {}).get("parameters"),
        "device": f"{info.get('device', {}).get('kind')}: {info.get('device', {}).get('name')}",
        "mixed_precision": info.get("resolved", {}).get("mixed_precision"),
        "train_seconds": round(sum(e.get("epoch_seconds", 0) for e in history), 1),
        "best_val_epoch": best["epoch"] if best else None,
        "best_val_miou_fg": best["val_mean_iou_foreground"] if best else None,
        "eval_split": (evaluation or {}).get("split"),
        "test_miou": metrics.get("mean_iou"),
        "test_miou_fg": metrics.get("mean_iou_foreground"),
        "test_pixel_accuracy": metrics.get("pixel_accuracy"),
        "test_class_iou": {c["name"]: c["iou"] for c in metrics.get("per_class", [])},
        "review_status": (evaluation or {}).get("review_status"),
        "val_curve": [(e["epoch"], e.get("val_mean_iou_foreground")) for e in history],
        "config_flat": _config_flat(info),
    }


def _config_flat(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    _flatten("", info.get("config", {}), out)
    out["resolved.batch_size"] = info.get("resolved", {}).get("batch_size")
    out["resolved.mixed_precision"] = info.get("resolved", {}).get("mixed_precision")
    out["device.kind"] = info.get("device", {}).get("kind")
    return out


def compare_runs(runs: list[RunDir]) -> dict[str, Any]:
    summaries = [run_summary(r) for r in runs]
    warnings: list[str] = []
    datasets = sorted({s["dataset_id"] for s in summaries if s["dataset_id"]})
    if len(datasets) > 1:
        warnings.append("These runs were evaluated on different dataset versions (" + ", ".join(datasets) +
                        "). Their test scores come from different test sets and must not be ranked against "
                        "each other.")
    if len({tuple(s["class_names"]) for s in summaries}) > 1:
        warnings.append("The runs use different class maps; per-class scores are not comparable.")
    for s in summaries:
        if s["state"] != "completed":
            warnings.append(f"{s['run_id']} is {s['state']}; its scores are incomplete.")
        if s["eval_split"] and s["eval_split"] != "test":
            warnings.append(f"{s['run_id']} was evaluated on the {s['eval_split']} split (no test split).")
    keys = sorted({k for s in summaries for k in s["config_flat"]})
    differences = {k: [s["config_flat"].get(k) for s in summaries] for k in keys
                   if len({json.dumps(s["config_flat"].get(k), sort_keys=True) for s in summaries}) > 1}
    return {"runs": summaries, "warnings": warnings, "differences": differences, "comparable": len(datasets) <= 1,
            "review_status": "unreviewed"}


TABLE_COLUMNS = ["run_id", "state", "dataset_id", "preset", "epochs_done", "learning_rate", "batch_size", "loss",
                 "seed", "model", "device", "train_seconds", "best_val_epoch", "best_val_miou_fg", "test_miou",
                 "test_miou_fg", "test_pixel_accuracy"]


def comparison_csv(comparison: dict[str, Any]) -> str:
    class_names = sorted({n for s in comparison["runs"] for n in s["test_class_iou"]})
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(TABLE_COLUMNS + [f"test_iou_{n}" for n in class_names])
    for s in comparison["runs"]:
        writer.writerow([s[c] for c in TABLE_COLUMNS] + [s["test_class_iou"].get(n) for n in class_names])
    return buf.getvalue()
