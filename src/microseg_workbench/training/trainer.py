"""Real model training with live, file-based progress reporting.

``run_training`` is used identically by the CLI, the desktop UI (which launches
it in a separate process and tails ``events.jsonl``) and the tests. It never
fakes progress: every reported number comes from actual computation, and
resource figures are real measurements (``simulated: false``).
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .. import TOOL_ID, __version__
from ..core.atomic import sha256_file, write_bytes_atomic, write_json_atomic
from ..core.augment import IGNORE_INDEX
from ..core.dataset import MANIFEST_FILENAME, DatasetVersion
from ..core.imageio import encode_png
from ..core.render import overlay, tile_grid, to_display_rgb
from ..core.workspace import Workspace
from .config import TrainConfig
from .data import Sample, epoch_batches, load_split
from .device import choose_device
from .engine import evaluate, load_checkpoint, save_checkpoint
from .model import ModelConfig, UNet, count_parameters
from .runs import RunDir, new_run_id, utc_now

EventCallback = Callable[[dict[str, Any]], None]


class TrainingCancelled(Exception):
    pass


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _resources(device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {"simulated": False}
    try:
        import psutil

        proc = psutil.Process()
        out["process_rss_mb"] = round(proc.memory_info().rss / 2**20, 1)
        out["system_cpu_percent"] = psutil.cpu_percent(interval=None)
        out["system_memory_percent"] = psutil.virtual_memory().percent
    except ImportError:
        out["note"] = "psutil not installed; CPU/RAM not measured"
    if device.type == "cuda":
        out["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 2**20, 1)
        out["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    return out


def _set_determinism(seed: int, num_threads: int) -> None:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _save_fixed_samples(run: RunDir, epoch: int, fixed: list[Sample], preds: dict[str, np.ndarray],
                        class_map) -> list[str]:
    folder = run.samples / f"epoch_{epoch:03d}"
    folder.mkdir(exist_ok=True)
    files = []
    for s in fixed:
        tile = overlay(s.image, preds[s.stem], class_map)
        write_bytes_atomic(folder / f"{s.stem}.png", encode_png(tile))
        files.append(f"samples/{folder.name}/{s.stem}.png")
    return files


def run_training(cfg: TrainConfig, workspace: Workspace, run_id: str | None = None,
                 on_event: EventCallback | None = None) -> RunDir:
    """Train, keep best/last checkpoints, then evaluate the best model on the test split."""
    workspace.ensure()
    run = RunDir(workspace.runs_dir / (run_id or new_run_id())).create()
    total_epochs = cfg.epochs

    def emit(event: dict[str, Any]) -> None:
        event.setdefault("time", utc_now())
        run.append_event(event)
        if on_event:
            on_event(event)

    run.write_status(state="starting", epoch=0, total_epochs=total_epochs, message="Loading dataset")
    try:
        version = DatasetVersion.open(workspace.datasets_dir / cfg.dataset_id)
        problems = version.verify()
        if problems:
            raise ValueError(f"Dataset {cfg.dataset_id} failed its integrity check: {problems[:5]}")
        class_map = version.class_map
        train = load_split(version, "train")
        val = load_split(version, "val")
        test = load_split(version, "test")
        if not train:
            raise ValueError("The training split is empty.")
        channels = {s.image.shape[2] for s in train + val + test}
        if len(channels) != 1:
            raise ValueError(f"Mixed channel counts in dataset: {channels}")

        _set_determinism(cfg.seed, cfg.num_threads)
        device, device_info, device_note = choose_device(cfg.device)
        model_cfg = ModelConfig(in_channels=channels.pop(), num_classes=class_map.num_classes,
                                base_channels=cfg.base_channels, depth=cfg.depth)
        model = UNet(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        steps_per_epoch = -(-len(train) * cfg.patches_per_image // cfg.batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.learning_rate, total_steps=steps_per_epoch * total_epochs, pct_start=0.2)

        fixed_source = "val" if val else "train"
        fixed = (val or train)[: cfg.fixed_samples]
        info = {
            "run_id": run.run_id,
            "created_at": utc_now(),
            "tool": {"id": TOOL_ID, "version": __version__, "git_commit": _git_commit()},
            "config": cfg.to_dict(),
            "model": model_cfg.to_dict() | {"parameters": count_parameters(model)},
            "dataset": {"dataset_id": version.dataset_id, "manifest_sha256": sha256_file(version.root / MANIFEST_FILENAME),
                        "split_counts": {"train": len(train), "val": len(val), "test": len(test)}},
            "class_map": class_map.to_dict(),
            "device": device_info.to_dict() | {"note": device_note},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": torch.__version__, "numpy": np.__version__,
                            "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True},
            "fixed_samples": {"source_split": fixed_source, "stems": [s.stem for s in fixed]},
            "loss": "cross_entropy (ignore_index=255)",
            "model_selection": "best validation mean IoU over foreground classes (falls back to training loss)",
        }
        write_json_atomic(run.run_json, info)
        for s in fixed:
            truth = tile_grid([to_display_rgb(s.image), overlay(s.image, s.labels, class_map)], columns=2)
            write_bytes_atomic(run.samples / f"truth_{s.stem}.png", encode_png(truth))
        emit({"type": "start", "device": device_info.label(), "note": device_note,
              "steps_per_epoch": steps_per_epoch, "epochs": total_epochs, "parameters": count_parameters(model)})

        best_score: float | None = None
        best_epoch = 0
        start = time.monotonic()
        step = 0
        for epoch in range(1, total_epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            batches = epoch_batches(train, cfg.augment, cfg.patches_per_image, cfg.batch_size, cfg.seed, epoch)
            losses = []
            for bi, (xb, yb) in enumerate(batches, 1):
                if run.cancel_requested():
                    raise TrainingCancelled()
                x = torch.from_numpy(xb).to(device)
                y = torch.from_numpy(yb).to(device)
                if bool((y == IGNORE_INDEX).all()):
                    continue
                loss = F.cross_entropy(model(x), y, ignore_index=IGNORE_INDEX)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss became {float(loss)} at epoch {epoch}, step {bi}. "
                                             "Try a lower learning rate.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                losses.append(float(loss.detach()))
                elapsed = time.monotonic() - start
                done = (epoch - 1) * steps_per_epoch + bi
                eta = elapsed / done * (total_epochs * steps_per_epoch - done)
                emit({"type": "step", "epoch": epoch, "batch": bi, "batches": len(batches), "step": step,
                      "loss": losses[-1], "lr": scheduler.get_last_lr()[0], "elapsed_s": round(elapsed, 3)})
                run.write_status(state="running", epoch=epoch, total_epochs=total_epochs, batch=bi,
                                 batches=len(batches), eta_s=round(eta, 1), message=f"Epoch {epoch}/{total_epochs}")

            train_loss = float(np.mean(losses)) if losses else None
            val_metrics, val_loss, preds = evaluate(model, val, device, class_map.names) if val else (None, None, {})
            if fixed and fixed_source == "train":
                _, _, preds = evaluate(model, fixed, device, class_map.names)
            sample_files = _save_fixed_samples(run, epoch, fixed, preds, class_map)
            score = (val_metrics["mean_iou_foreground"] if val_metrics and val_metrics["mean_iou_foreground"] is not None
                     else (-train_loss if train_loss is not None else None))
            improved = score is not None and (best_score is None or score > best_score)
            ckpt = {"run_id": run.run_id, "epoch": epoch, "dataset_id": version.dataset_id,
                    "class_map": class_map.to_dict(), "normalization": cfg.normalization,
                    "val_mean_iou_foreground": val_metrics["mean_iou_foreground"] if val_metrics else None}
            save_checkpoint(run.checkpoints / "last.pt", model, ckpt)
            if improved:
                best_score, best_epoch = score, epoch
                save_checkpoint(run.checkpoints / "best.pt", model, ckpt)
            emit({
                "type": "epoch", "epoch": epoch, "epochs": total_epochs, "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_iou": val_metrics["mean_iou"] if val_metrics else None,
                "val_mean_iou_foreground": val_metrics["mean_iou_foreground"] if val_metrics else None,
                "val_per_class_iou": {c["name"]: c["iou"] for c in val_metrics["per_class"]} if val_metrics else None,
                "best_epoch": best_epoch, "is_best": improved, "epoch_seconds": round(time.monotonic() - epoch_start, 3),
                "sample_images": sample_files, "resources": _resources(device),
            })
        run.write_status(state="evaluating", epoch=total_epochs, total_epochs=total_epochs,
                         message="Evaluating best model on the test split")
        report = evaluate_run(run, workspace, device=device)
        emit({"type": "evaluation", "split": report["split"], "metrics": report["metrics"] and {
            k: report["metrics"][k] for k in ("mean_iou", "mean_iou_foreground", "pixel_accuracy")}})
        run.write_status(state="completed", epoch=total_epochs, total_epochs=total_epochs, best_epoch=best_epoch,
                         message=f"Finished. Best epoch {best_epoch}. Results are unreviewed.")
        emit({"type": "end", "state": "completed"})
    except TrainingCancelled:
        run.write_status(state="cancelled", total_epochs=total_epochs,
                         message="Cancelled by user. The last completed epoch's checkpoint is kept.")
        emit({"type": "end", "state": "cancelled"})
    except Exception as exc:  # recorded for the user, then re-raised for callers/tests
        run.write_status(state="failed", total_epochs=total_epochs, message=str(exc),
                         error_type=type(exc).__name__)
        emit({"type": "end", "state": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise
    return run


def evaluate_run(run: RunDir, workspace: Workspace, device: torch.device | None = None) -> dict[str, Any]:
    """Evaluate ``best.pt`` on the test split (val if no test split) and write ``evaluation/``."""
    device = device or torch.device("cpu")
    model, ckpt = load_checkpoint(run.checkpoints / "best.pt", device)
    version = DatasetVersion.open(workspace.datasets_dir / ckpt["dataset_id"])
    class_map = version.class_map
    if class_map.to_dict() != ckpt["class_map"]:
        raise ValueError("Checkpoint class map differs from the dataset's; refusing to evaluate.")
    split = "test" if version.samples("test") else "val"
    samples = load_split(version, split)
    metrics, loss, preds = evaluate(model, samples, device, class_map.names) if samples else (None, None, {})
    overlays = []
    for s in samples:
        tile = tile_grid([to_display_rgb(s.image), overlay(s.image, s.labels, class_map),
                          overlay(s.image, preds[s.stem], class_map)], columns=3)
        write_bytes_atomic(run.evaluation / f"{s.stem}.png", encode_png(tile))
        Image.fromarray(np.asarray(class_map.values, np.uint8)[preds[s.stem]]).save(
            run.evaluation / f"{s.stem}_pred_mask.png")
        overlays.append(f"evaluation/{s.stem}.png")
    report = {
        "run_id": run.run_id,
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_epoch": ckpt["epoch"],
        "dataset_id": version.dataset_id,
        "split": split,
        "note": None if split == "test" else "No test split in this dataset version; validation split reported.",
        "images": [s.stem for s in samples],
        "loss": loss,
        "metrics": metrics,
        "overlay_layout": "image | ground truth | prediction",
        "prediction_masks": "evaluation/<stem>_pred_mask.png hold class values from the class map",
        "review_status": "unreviewed",
        "evaluated_at": utc_now(),
    }
    write_json_atomic(run.evaluation / "evaluation.json", report)
    return report
