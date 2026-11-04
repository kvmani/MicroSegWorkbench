"""Installation self-test for roll-out: is this workstation (and the shared folder) ready for real work?

Runs the real code paths on tiny SYNTHETIC data in a temporary folder inside the workspace: device detection, data
validation, dataset version, a short training run on the chosen device, evaluation, prediction, CPU
reproducibility and - when a shared folder is given - publishing and downloading through a temporary store inside
that folder (the real records are never touched). Everything temporary is removed afterwards unless ``keep``.

Each check is PASS, WARN (works, but needs attention), FAIL or SKIP. A text report is saved in the workspace
(``selftest-reports/``) for the roll-out sign-off.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import __version__
from .core.workspace import Workspace

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
MIN_FREE_GB = 10


@dataclass
class Check:
    name: str
    status: str
    detail: str
    seconds: float = 0.0


@dataclass
class SelfTest:
    checks: list[Check] = field(default_factory=list)
    report_path: Path | None = None

    @property
    def ok(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    def text(self) -> str:
        lines = [f"MicroSegWorkbench {__version__} self-test - {datetime.now():%Y-%m-%d %H:%M}",
                 f"Computer: {platform.node()}  {platform.platform()}  Python {platform.python_version()}", ""]
        lines += [f"[{c.status}] {c.name}: {c.detail}" + (f" ({c.seconds:.1f} s)" if c.seconds >= 0.05 else "")
                  for c in self.checks]
        n_fail = sum(c.status == FAIL for c in self.checks)
        n_warn = sum(c.status == WARN for c in self.checks)
        lines += ["", "RESULT: " + ("READY" if not n_fail else "NOT READY")
                  + f" - {n_fail} failure(s), {n_warn} warning(s). Synthetic data only; not a scientific result."]
        return "\n".join(lines)


def run_selftest(workspace: Workspace, share: Path | None = None, device: str = "auto", keep: bool = False,
                 progress: Callable[[str], None] | None = None) -> SelfTest:
    result = SelfTest()
    say = progress or (lambda _m: None)

    def check(name: str, fn: Callable[[], tuple[str, str]]) -> bool:
        say(f"{name}...")
        start = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as exc:  # report every failure; never crash the self-test itself
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
            if os.environ.get("MSW_DEBUG"):
                traceback.print_exc()
        result.checks.append(Check(name, status, detail, time.perf_counter() - start))
        say(f"[{status}] {name}: {detail}")
        return status != FAIL

    state: dict = {}
    workspace.ensure()
    tmp_root = workspace.root / f".selftest-{uuid.uuid4().hex[:8]}"

    def software() -> tuple[str, str]:
        import numpy

        parts = [f"Python {platform.python_version()}", f"numpy {numpy.__version__}"]
        try:
            import torch

            parts.append(f"torch {torch.__version__}")
        except ImportError:
            return FAIL, "PyTorch is not installed - training is impossible. Re-run the installer."
        missing = []
        for mod in ("PySide6", "psutil"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        if sys.version_info < (3, 11):
            return FAIL, ", ".join(parts) + " - Python 3.11 or newer is required."
        if missing:
            return WARN, ", ".join(parts) + f"; not installed: {', '.join(missing)} (desktop app or memory readings " \
                                            "unavailable)"
        return PASS, ", ".join(parts) + ", PySide6, psutil"

    def compute() -> tuple[str, str]:
        from .training.device import choose_device

        torch_device, info, _note = choose_device(device)
        state["device"] = str(torch_device)
        if info.kind == "cuda":
            mem = info.total_memory_mb or 0
            if mem < 6000:
                return WARN, f"{info.label()} - small GPU memory; use the Quick CPU or Standard preset with small " \
                             "patches."
            return PASS, info.label()
        return WARN, ("no usable NVIDIA GPU - training runs on the CPU, which works but is much slower. For real "
                      "datasets use a workstation with an NVIDIA GPU; if it has one, 'msw devices' explains why it "
                      "is not used.")

    def disk() -> tuple[str, str]:
        tmp_root.mkdir(parents=True)
        probe = tmp_root / "write-test.bin"
        probe.write_bytes(os.urandom(1024))
        probe.unlink()
        free_gb = shutil.disk_usage(tmp_root).free / 2**30
        where = f"workspace {workspace.root} is writable, {free_gb:.0f} GB free"
        if free_gb < MIN_FREE_GB:
            return WARN, where + f" - keep at least {MIN_FREE_GB} GB free for dataset copies and checkpoints."
        return PASS, where

    def data() -> tuple[str, str]:
        from .core.dataset import create_dataset_version, validate_folder
        from .core.splits import DEFAULT_GROUP_REGEX, SplitConfig
        from .core.synthetic import SyntheticSpec, write_synthetic_dataset

        folder = tmp_root / "data"
        write_synthetic_dataset(folder, SyntheticSpec(specimens=4, images_per_specimen=2, size=64, seed=5))
        report = validate_folder(folder, DEFAULT_GROUP_REGEX)
        if not report.ok:
            return FAIL, "validation of correct synthetic data failed: " + "; ".join(i.message for i in report.issues[:3])
        state["ws"] = ws = Workspace(tmp_root / "ws").ensure()
        state["version"] = version = create_dataset_version(report, SplitConfig(seed=1), ws.datasets_dir, "selftest")
        return PASS, f"8 image/mask pairs validated; dataset version {version.dataset_id} created and verified"

    def train() -> tuple[str, str]:
        import math

        from .core.augment import AugmentConfig
        from .training.config import TrainConfig
        from .training.trainer import run_training

        ws, version = state["ws"], state["version"]
        cfg = TrainConfig(dataset_id=version.dataset_id, epochs=3, base_channels=8, depth=2, patches_per_image=4,
                          batch_size=4, device=state.get("device", "cpu").split(":")[0],
                          augment=AugmentConfig(patch_size=32))
        run = run_training(cfg, ws, run_id="selftest-run")
        state["run"] = run
        if run.state() != "completed":
            return FAIL, f"training ended as {run.state()}: {run.status().get('message')}"
        import json

        evaluation = json.loads((run.evaluation / "evaluation.json").read_text())
        m = evaluation["metrics"]
        values = [m["pixel_accuracy"], evaluation["loss"]]
        if not all(v is not None and math.isfinite(v) for v in values) or not evaluation.get("accuracy"):
            return FAIL, "evaluation produced missing or non-finite numbers"
        return PASS, (f"3 epochs on {cfg.device.upper()}, evaluated on {len(evaluation['images'])} test images "
                      f"(pixel accuracy {m['pixel_accuracy']:.3f}; tiny model, the number itself is not meaningful)")

    def predict() -> tuple[str, str]:
        from .training.predict import predict_folder

        version, run = state["version"], state["run"]
        folder = version.image_path(version.samples()[0]).parent
        report = predict_folder(run, folder, tmp_root / "predictions", device=state.get("device", "cpu").split(":")[0])
        if not report["images"] or report["errors"]:
            return FAIL, f"{len(report['images'])} segmented, {len(report['errors'])} error(s)"
        if not (tmp_root / "predictions" / "predictions.csv").exists():
            return FAIL, "predictions.csv missing"
        return PASS, f"{len(report['images'])} images segmented; masks, overlays, predictions.json/.csv written"

    def reproducible() -> tuple[str, str]:
        from .core.augment import AugmentConfig
        from .training.config import TrainConfig
        from .training.trainer import run_training

        ws, version = state["ws"], state["version"]
        losses = []
        for i in range(2):
            cfg = TrainConfig(dataset_id=version.dataset_id, epochs=1, base_channels=4, depth=2, patches_per_image=2,
                              batch_size=4, device="cpu", seed=11, augment=AugmentConfig(patch_size=32))
            run = run_training(cfg, ws, run_id=f"selftest-repeat-{i}")
            losses.append(run.epoch_history()[0]["train_loss"])
        if losses[0] != losses[1]:
            return FAIL, f"two identical CPU runs gave different losses ({losses[0]} vs {losses[1]})"
        return PASS, "two identical CPU runs gave identical results"

    def shared() -> tuple[str, str]:
        from .collab.share import SharedStore

        if share is None:
            return SKIP, "no shared folder given (msw selftest --share PATH, or configure one in step 6)"
        if not Path(share).exists():
            return FAIL, f"{share} is not reachable from this computer (mapped drive? permissions?)"
        probe_root = Path(share) / f".selftest-{platform.node()}-{uuid.uuid4().hex[:8]}"
        try:
            store = SharedStore.create(probe_root)
            store.publish_dataset(state["version"], user="selftest")
            store.publish_run(state["run"], user="selftest", note="installation self-test (synthetic)")
            other = Workspace(tmp_root / "ws-download").ensure()
            pulled = store.pull_run("selftest-run", other)
            if not (pulled.checkpoints / "best.pt").exists():
                return FAIL, "downloaded run is incomplete"
            problems = store.check()
            if problems:
                return FAIL, f"shared files failed verification: {problems[:2]}"
        finally:
            shutil.rmtree(probe_root, ignore_errors=True)
        return PASS, f"publish, verified download and clean-up work in {share} (temporary folder, real records untouched)"

    check("Software", software) and check("Compute device", compute)
    try:
        if check("Workspace disk", disk) and check("Data validation and dataset version", data):
            if check("Training and evaluation", train):
                check("Segmenting new images", predict)
                check("Shared folder", shared)
            check("Reproducibility (CPU)", reproducible)
    finally:
        if not keep:
            shutil.rmtree(tmp_root, ignore_errors=True)
    try:
        reports = workspace.root / "selftest-reports"
        reports.mkdir(exist_ok=True)
        result.report_path = reports / f"selftest-{platform.node()}-{datetime.now():%Y%m%d-%H%M%S}.txt"
        result.report_path.write_text(result.text(), encoding="utf-8")
    except OSError:
        result.report_path = None
    return result
