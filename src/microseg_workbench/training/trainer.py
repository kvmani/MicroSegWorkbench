"""Real model training with live, file-based progress reporting, resume and GPU safety.

``run_training`` (new run) and ``resume_training`` (continue a stopped run) are
used identically by the CLI, the desktop UI (which launches them in a separate
process and tails ``events.jsonl``) and the tests. Nothing is faked: every
reported number comes from actual computation and resource figures are real
measurements unless explicitly flagged ``simulated``.

Resuming restores model, optimizer, learning-rate schedule, mixed-precision
scaler and best-model bookkeeping from ``checkpoints/last.pt``. Because batch
order and augmentation are keyed only by (seed, epoch, sample), a resumed run
continues exactly as an uninterrupted run would (verified on CPU).
"""

from __future__ import annotations

import copy
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .. import TOOL_ID, __version__
from ..core.atomic import sha256_file, write_bytes_atomic, write_json_atomic
from ..core.augment import IGNORE_INDEX
from ..core.dataset import MANIFEST_FILENAME, DatasetVersion
from ..core.imageio import encode_png
from ..core.render import overlay, thumbnail, tile_grid, to_display_rgb
from ..core.workspace import Workspace
from .config import TrainConfig
from ..core.preprocess import Preprocess
from .data import Sample, SplitSource, count_batches, cropped, iter_epoch_batches, prefetch
from .device import choose_device, cuda_diagnosis, is_out_of_memory, probe_batch_size
from .engine import evaluate, load_checkpoint, save_checkpoint, segmentation_loss
from .model import ModelConfig, UNet, count_parameters
from .monitor import ResourceSampler
from .runs import ACTIVE_STATES, RunDir, new_run_id, utc_now

EventCallback = Callable[[dict[str, Any]], None]
AUTO_BATCH_START = 64
PREVIEW_MAX_SIDE = 768  # per-epoch fixed-sample previews (display only)
EVAL_PANEL_MAX_SIDE = 1024  # evaluation overlay panels (display only; prediction masks stay full resolution)
CPU_AUTO_BATCH = 8


class TrainingCancelled(Exception):
    pass


class TrainingOutOfMemory(RuntimeError):
    """Raised with a user-facing explanation when a training step runs out of memory."""


class ResumeError(ValueError):
    pass


def lr_schedule(total_steps: int) -> Callable[[int], float]:
    """Learning-rate factor per step: linear warm-up over 10 % of steps, then cosine decay to 5 %.

    Valid for any number of steps (including a single step).
    """
    warmup = max(1, round(0.1 * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    return factor


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


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
        write_bytes_atomic(folder / f"{s.stem}.png",
                           encode_png(thumbnail(overlay(s.image, preds[s.stem], class_map), PREVIEW_MAX_SIDE)))
        files.append(f"samples/{folder.name}/{s.stem}.png")
    return files


def _resolve_batch_size(cfg: TrainConfig, model: UNet, device: torch.device, channels: int, use_amp: bool,
                        n_patches: int) -> tuple[int, list[dict]]:
    if cfg.batch_size > 0:
        return min(cfg.batch_size, max(1, n_patches)), []
    if device.type != "cuda":
        return min(CPU_AUTO_BATCH, n_patches), [{"note": "automatic batch size on CPU uses a fixed default"}]
    patch = cfg.augment.patch_size
    saved = copy.deepcopy(model.state_dict())  # probing must not change weights or BatchNorm statistics

    def try_step(batch: int) -> None:
        x = torch.zeros(batch, channels, patch, patch, device=device)
        y = torch.zeros(batch, patch, patch, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = F.cross_entropy(model(x), y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)

    start = min(AUTO_BATCH_START, max(1, n_patches))
    try:
        # Halving from 64 leaves headroom in practice; a later OOM still gets a clear message.
        found, attempts = probe_batch_size(try_step, start)
    finally:
        model.load_state_dict(saved)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return found, attempts


def _check_memory(train: SplitSource, val: SplitSource | None, cfg: TrainConfig) -> None:
    """Refuse early, with advice, when keeping the data in memory would exhaust the computer's RAM."""
    need = train.estimated_bytes() + (val.estimated_bytes() if val is not None else 0)
    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        return
    if need > 0.6 * available:
        per_image = max(1, train.estimated_bytes() // max(1, len(train)))
        suggestion = max(1, int(0.3 * available // per_image))
        raise MemoryError(
            f"Keeping all {len(train)} training images in memory needs about {need / 2**30:.1f} GB, but only "
            f"{available / 2**30:.1f} GB is free. Set 'images in memory' to about {suggestion} (or reduce the resize "
            "factor) so the data are processed in blocks.")


def run_training(cfg: TrainConfig, workspace: Workspace, run_id: str | None = None,
                 on_event: EventCallback | None = None) -> RunDir:
    """Start a new run: train, keep best/last checkpoints, then evaluate the best model on the test split."""
    workspace.ensure()
    run = RunDir(workspace.runs_dir / (run_id or new_run_id())).create()
    return _execute(run, cfg, workspace, on_event, resume=False)


def resume_training(workspace: Workspace, run_id: str, on_event: EventCallback | None = None) -> RunDir:
    """Continue a cancelled, failed or interrupted run from its last completed epoch."""
    run = RunDir(workspace.runs_dir / run_id)
    info = run.info()
    if not info:
        raise ResumeError(f"{run_id} is not a training run in this workspace.")
    state = run.state()
    if state == "completed":
        raise ResumeError("This run already completed; start a new run to train further.")
    if state in ACTIVE_STATES:
        raise ResumeError("This run is still running.")
    if not (run.checkpoints / "last.pt").exists():
        raise ResumeError("No checkpoint was saved before the run stopped (it stopped in the first epoch). "
                          "Start a new run instead.")
    if run.cancel_flag.exists():
        run.cancel_flag.unlink()
    return _execute(run, TrainConfig.from_dict(info["config"]), workspace, on_event, resume=True)


def _execute(run: RunDir, cfg: TrainConfig, workspace: Workspace, on_event: EventCallback | None,
             resume: bool) -> RunDir:
    total_epochs = cfg.epochs
    last_status = [0.0]

    def emit(event: dict[str, Any]) -> None:
        event.setdefault("time", utc_now())
        run.append_event(event)
        if on_event:
            on_event(event)

    def status(force: bool = True, **fields: Any) -> None:
        now = time.monotonic()
        if force or now - last_status[0] >= 0.5:
            last_status[0] = now
            run.write_status(total_epochs=total_epochs, **fields)

    sampler: ResourceSampler | None = None
    batch_size = cfg.batch_size
    status(state="starting", epoch=0, message="Resuming" if resume else "Loading dataset")
    try:
        version = DatasetVersion.open(workspace.datasets_dir / cfg.dataset_id)
        problems = version.verify()
        if problems:
            raise ValueError(f"Dataset {cfg.dataset_id} failed its integrity check: {problems[:5]}")
        class_map = version.class_map
        prep = cfg.preprocess()
        keep_all = cfg.images_in_memory == 0
        train = SplitSource(version, "train", prep, max_cached=None if keep_all else cfg.images_in_memory)
        val = SplitSource(version, "val", max_cached=None if keep_all else 0)
        test_count = len(version.samples("test"))
        if not len(train):
            raise ValueError("The training split is empty.")
        _check_memory(train, val if keep_all else None, cfg)
        all_records = train.records + val.records + list(version.samples("test"))
        for r in all_records:
            prep.analysed_region(r["height"], r["width"])  # raises with a clear message if borders are too large
        too_small = [r["stem"] for r in train.records
                     if min(prep.scaled_shape(**{k: v for k, v in prep.analysed_region(r["height"], r["width"]).items()
                                                 if k in ("height", "width")})) < 8]
        if too_small:
            raise ValueError(f"Resize factor {prep.resize_factor} makes images too small: {too_small[:3]}")
        channels = train.channels() | val.channels() | {int(r["channels"]) for r in version.samples("test")}
        if len(channels) != 1:
            raise ValueError(f"Mixed channel counts in dataset: {channels}")
        in_channels = channels.pop()

        _set_determinism(cfg.seed, cfg.num_threads)
        device, device_info, device_note = choose_device(cfg.device)
        if cfg.mixed_precision == "on" and device.type != "cuda":
            raise ValueError("Mixed precision needs an NVIDIA GPU; choose 'auto' or 'off' on the CPU.")
        use_amp = device.type == "cuda" and cfg.mixed_precision in ("auto", "on")
        model_cfg = ModelConfig(in_channels=in_channels, num_classes=class_map.num_classes,
                                base_channels=cfg.base_channels, depth=cfg.depth)
        model = UNet(model_cfg).to(device)
        n_patches = len(train) * cfg.patches_per_image

        info = run.info() if resume else {}
        if resume:
            batch_size = int(info["resolved"]["batch_size"])
            probe: list[dict] = []
        else:
            status(state="starting", epoch=0, message="Choosing batch size" if cfg.batch_size == 0 else "Starting")
            batch_size, probe = _resolve_batch_size(cfg, model, device, in_channels, use_amp, n_patches)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        steps_per_epoch = -(-n_patches // batch_size)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule(steps_per_epoch * total_epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        fixed_source = "val" if len(val) else "train"
        fixed_split = SplitSource(version, fixed_source, max_cached=0)
        fixed = [fixed_split.get(i) for i in range(min(cfg.fixed_samples, len(fixed_split)))]
        best_score: float | None = None
        best_epoch, step, start_epoch = 0, 0, 1

        if resume:
            state = torch.load(run.checkpoints / "last.pt", map_location=device, weights_only=True)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            if use_amp and state.get("scaler_state"):
                scaler.load_state_dict(state["scaler_state"])
            best_score, best_epoch = state["best_score"], state["best_epoch"]
            step, start_epoch = state["step"], state["epoch"] + 1
            info.setdefault("resumes", []).append({"at": utc_now(), "from_epoch": state["epoch"],
                                                   "device": device_info.to_dict(), "host": platform.node()})
            write_json_atomic(run.run_json, info)
            emit({"type": "resume", "from_epoch": state["epoch"], "device": device_info.label(), "note": device_note})
        else:
            info = {
                "run_id": run.run_id,
                "created_at": utc_now(),
                "tool": {"id": TOOL_ID, "version": __version__, "git_commit": _git_commit()},
                "config": cfg.to_dict(),
                "resolved": {"batch_size": batch_size, "mixed_precision": use_amp, "steps_per_epoch": steps_per_epoch,
                             "batch_size_probe": probe},
                "model": model_cfg.to_dict() | {"parameters": count_parameters(model)},
                "dataset": {"dataset_id": version.dataset_id,
                            "manifest_sha256": sha256_file(version.root / MANIFEST_FILENAME),
                            "split_counts": {"train": len(train), "val": len(val), "test": test_count}},
                "class_map": class_map.to_dict(),
                "device": device_info.to_dict() | {"note": device_note, "cuda": cuda_diagnosis()},
                "host": platform.node(),
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "torch": torch.__version__, "torch_cuda": torch.version.cuda, "numpy": np.__version__,
                                "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True},
                "fixed_samples": {"source_split": fixed_source, "stems": [s.stem for s in fixed],
                                  "preview_max_side_px": PREVIEW_MAX_SIDE},
                "loss": f"{cfg.loss} (ignore label 255 excluded); validation loss reported as cross-entropy",
                "model_selection": "best validation mean IoU over foreground classes (falls back to training loss)",
            }
            write_json_atomic(run.run_json, info)
            for s in fixed:
                s = cropped(s, prep)
                truth = tile_grid([thumbnail(to_display_rgb(s.image), PREVIEW_MAX_SIDE),
                                   thumbnail(overlay(s.image, s.labels, class_map), PREVIEW_MAX_SIDE)], columns=2)
                write_bytes_atomic(run.samples / f"truth_{s.stem}.png", encode_png(truth))
            emit({"type": "start", "device": device_info.label(), "note": device_note, "batch_size": batch_size,
                  "mixed_precision": use_amp, "steps_per_epoch": steps_per_epoch, "epochs": total_epochs,
                  "parameters": count_parameters(model)})

        stopped_early: int | None = None
        sampler = ResourceSampler(interval_s=2.0, torch_device=device).start()
        last_resource_emit = 0.0
        start = time.monotonic()
        steps_done_here = 0
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            n_batches = count_batches(len(train), cfg.patches_per_image, batch_size)
            batches = prefetch(iter_epoch_batches(train, cfg.augment, cfg.patches_per_image, batch_size, cfg.seed,
                                                  epoch, prep.normalization, cfg.images_in_memory))
            losses = []
            for bi, (xb, yb) in enumerate(batches, 1):
                if run.cancel_requested():
                    raise TrainingCancelled()
                x = torch.from_numpy(xb).to(device, non_blocking=True)
                y = torch.from_numpy(yb).to(device, non_blocking=True)
                if bool((y == IGNORE_INDEX).all()):
                    continue
                try:
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        loss = segmentation_loss(model(x), y, cfg.loss)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Loss became {float(loss)} at epoch {epoch}, step {bi}. "
                                                 "Try a lower learning rate.")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as exc:
                    if is_out_of_memory(exc):
                        where = "GPU" if device.type == "cuda" else "computer"
                        raise TrainingOutOfMemory(
                            f"The {where} ran out of memory with batch size {batch_size} and "
                            f"{cfg.augment.patch_size} px patches. Resume or restart with a smaller batch size "
                            f"(e.g. {max(1, batch_size // 2)}), smaller patches, or the 'auto' batch size.") from exc
                    raise
                scheduler.step()
                step += 1
                steps_done_here += 1
                losses.append(float(loss.detach()))
                elapsed = time.monotonic() - start
                remaining = (total_epochs - epoch) * steps_per_epoch + (n_batches - bi)
                eta = elapsed / steps_done_here * remaining
                emit({"type": "step", "epoch": epoch, "batch": bi, "batches": n_batches, "step": step,
                      "loss": losses[-1], "lr": scheduler.get_last_lr()[0], "elapsed_s": round(elapsed, 3)})
                status(force=bi == n_batches, state="running", epoch=epoch, batch=bi, batches=n_batches,
                       eta_s=round(eta, 1), loss=losses[-1], message=f"Epoch {epoch}/{total_epochs}")
                now = time.monotonic()
                if now - last_resource_emit >= 2.0:
                    last_resource_emit = now
                    emit({"type": "resources", "epoch": epoch, "elapsed_s": round(elapsed, 3),
                          "images_per_s": round(steps_done_here * batch_size / max(elapsed, 1e-9), 2),
                          **sampler.latest()})

            train_loss = float(np.mean(losses)) if losses else None
            fixed_stems = {f.stem for f in fixed}
            val_metrics, val_loss, preds = (evaluate(model, val, device, class_map.names, prep, keep=fixed_stems)
                                            if len(val) else (None, None, {}))
            if fixed and fixed_source == "train":
                _, _, preds = evaluate(model, fixed, device, class_map.names, prep)
            sample_files = _save_fixed_samples(run, epoch, [cropped(f, prep) for f in fixed], preds, class_map)
            score = (val_metrics["mean_iou_foreground"] if val_metrics and val_metrics["mean_iou_foreground"] is not None
                     else (-train_loss if train_loss is not None else None))
            improved = score is not None and (best_score is None or score > best_score)
            if improved:
                best_score, best_epoch = score, epoch
            meta = {"run_id": run.run_id, "epoch": epoch, "dataset_id": version.dataset_id,
                    "class_map": class_map.to_dict(), "normalization": cfg.normalization,
                    "preprocess": prep.to_dict(),
                    "val_mean_iou_foreground": val_metrics["mean_iou_foreground"] if val_metrics else None}
            if improved:
                save_checkpoint(run.checkpoints / "best.pt", model, meta)
            save_checkpoint(run.checkpoints / "last.pt", model, meta | {
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "best_score": best_score, "best_epoch": best_epoch, "step": step})
            emit({
                "type": "epoch", "epoch": epoch, "epochs": total_epochs, "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_iou": val_metrics["mean_iou"] if val_metrics else None,
                "val_mean_iou_foreground": val_metrics["mean_iou_foreground"] if val_metrics else None,
                "val_per_class_iou": {c["name"]: c["iou"] for c in val_metrics["per_class"]} if val_metrics else None,
                "best_epoch": best_epoch, "is_best": improved, "epoch_seconds": round(time.monotonic() - epoch_start, 3),
                "sample_images": sample_files, "resources": sampler.latest(),
            })
            patience = cfg.early_stopping_patience
            if patience and len(val) and epoch < total_epochs and epoch - best_epoch >= patience:
                stopped_early = epoch
                emit({"type": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "patience": patience})
                break
        sampler.stop()
        status(state="evaluating", epoch=total_epochs, message="Evaluating best model on the test split")
        report = evaluate_run(run, workspace, device=device)
        emit({"type": "evaluation", "split": report["split"], "metrics": report["metrics"] and {
            k: report["metrics"][k] for k in ("mean_iou", "mean_iou_foreground", "pixel_accuracy")}})
        early = (f" Stopped early after epoch {stopped_early}: no validation improvement for "
                 f"{cfg.early_stopping_patience} epochs." if stopped_early else "")
        status(state="completed", epoch=stopped_early or total_epochs, best_epoch=best_epoch,
               stopped_early=stopped_early,
               message=f"Finished. Best epoch {best_epoch}.{early} Results are unreviewed.")
        emit({"type": "end", "state": "completed"})
    except TrainingCancelled:
        status(state="cancelled", message="Cancelled by user. The last completed epoch's checkpoint is kept; "
                                          "the run can be resumed.")
        emit({"type": "end", "state": "cancelled"})
    except Exception as exc:  # recorded for the user, then re-raised for callers/tests
        kind = "out_of_memory" if isinstance(exc, TrainingOutOfMemory) else type(exc).__name__
        status(state="failed", message=str(exc), error_type=kind, batch_size=batch_size)
        emit({"type": "end", "state": "failed", "error": str(exc), "error_type": kind,
              "traceback": traceback.format_exc()})
        raise
    finally:
        if sampler is not None:
            sampler.stop()
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
    source = SplitSource(version, split, max_cached=0)
    prep = Preprocess.from_meta(ckpt)
    values = np.asarray(class_map.values, np.uint8)

    def save(s: Sample, pred: np.ndarray) -> None:
        tile = tile_grid([thumbnail(to_display_rgb(s.image), EVAL_PANEL_MAX_SIDE),
                          thumbnail(overlay(s.image, s.labels, class_map), EVAL_PANEL_MAX_SIDE),
                          thumbnail(overlay(s.image, pred, class_map), EVAL_PANEL_MAX_SIDE)], columns=3)
        write_bytes_atomic(run.evaluation / f"{s.stem}.png", encode_png(tile))
        write_bytes_atomic(run.evaluation / f"{s.stem}_pred_mask.png", encode_png(values[pred]))

    metrics, loss, _ = (evaluate(model, source, device, class_map.names, prep, keep=set(), on_prediction=save)
                        if len(source) else (None, None, {}))
    samples = source.records
    report = {
        "run_id": run.run_id,
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_epoch": ckpt["epoch"],
        "dataset_id": version.dataset_id,
        "split": split,
        "note": None if split == "test" else "No test split in this dataset version; validation split reported.",
        "images": [r["stem"] for r in samples],
        "loss": loss,
        "metrics": metrics,
        "preprocess": prep.to_dict(),
        "resolution": "metrics computed at original image resolution against original masks, excluding any "
                      "excluded border; prediction masks cover the analysed region only",
        "overlay_layout": f"image | ground truth | prediction (each panel at most {EVAL_PANEL_MAX_SIDE} px; display "
                          "only - prediction masks are full resolution)",
        "prediction_masks": "evaluation/<stem>_pred_mask.png hold class values from the class map",
        "review_status": "unreviewed",
        "ground_truth": ("contains masks NOT approved in the annotation tool"
                         if version.manifest.get("contains_unapproved_annotations") else "as in dataset version"),
        "evaluated_at": utc_now(),
    }
    write_json_atomic(run.evaluation / "evaluation.json", report)
    return report
