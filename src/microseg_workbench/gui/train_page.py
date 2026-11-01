"""Step 3: train a model and watch it learn.

Training runs in a separate Python process (``msw train`` / ``msw resume``) so the
window stays responsive and a crash in training cannot take the UI down. This page
tails the run's ``events.jsonl`` / ``status.json`` files - the same records the CLI writes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer  # noqa: F401
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QScrollArea, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from ..core.augment import AugmentConfig, AugmentConfigError
from ..core.dataset import list_dataset_versions
from ..training.config import PRESETS, TrainConfigError, preset_config
from ..training.runs import ACTIVE_STATES, TERMINAL_STATES, EventTail, RunDir, new_run_id
from .context import AppContext
from .widgets import ImageView, LineChart, hint, run_background

HELP = ("<b>Step 3 - Training.</b> Choose a dataset version and a preset. <i>Quick CPU</i> trains a small model in "
        "about a minute on any computer; <i>Standard</i> and <i>Large GPU</i> are meant for NVIDIA GPUs and pick the "
        "batch size automatically to fit GPU memory. Charts update after every epoch; the <i>Prediction evolution</i> "
        "tab shows the model's output on the same validation images after each epoch, and <i>Computer resources</i> "
        "shows CPU, memory and GPU use. <b>Cancel</b> stops after the current step; a stopped run can be resumed "
        "from step 4.")


def worker_command(cli_args: list[str]) -> tuple[str, list[str]]:
    """Program and arguments that run ``msw <cli_args>`` in a child process (source install or frozen build)."""
    if getattr(sys, "frozen", False):
        return sys.executable, ["--cli", *cli_args]
    return sys.executable, ["-m", "microseg_workbench", *cli_args]


def _hide_console_window(proc: QProcess) -> None:
    """On Windows, start the training child without flashing a console window (no-op elsewhere)."""
    if sys.platform != "win32" or not hasattr(proc, "setCreateProcessArgumentsModifier"):
        return
    create_no_window = 0x08000000

    def modifier(args) -> None:  # QProcess.CreateProcessArguments
        args.flags |= create_no_window

    try:
        proc.setCreateProcessArgumentsModifier(modifier)
    except Exception:  # never block training over a cosmetic detail
        pass


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    if s < 60:
        return f"{s} s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


class MemoryBar(QWidget):
    """Label + bar for a used/total memory figure."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = QLabel(title)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title.setMinimumWidth(170)
        layout.addWidget(self.title)
        layout.addWidget(self.bar, 1)
        self.set(None, None)

    def set(self, used: float | None, total: float | None, unit: str = "MB", suffix: str = "") -> None:
        if used is None:
            self.bar.setValue(0)
            self.bar.setFormat("not available")
            return
        if total:
            self.bar.setValue(int(1000 * min(1.0, used / total)))
            self.bar.setFormat(f"{used:,.0f} / {total:,.0f} {unit}{suffix}")
        else:
            self.bar.setValue(0)
            self.bar.setFormat(f"{used:,.0f} {unit}{suffix}")


class TrainPage(QWidget):
    def __init__(self, ctx: AppContext, augment_config: Callable[[], AugmentConfig],
                 preprocess: Callable[[], Any] | None = None) -> None:
        super().__init__()
        self.ctx = ctx
        self.augment_config = augment_config
        self.preprocess = preprocess
        self.process: QProcess | None = None
        self.run: RunDir | None = None
        self.tail: EventTail | None = None
        self.stderr = ""
        self.fixed_stems: list[str] = []
        self.epochs_seen: list[int] = []
        self.final_state: str | None = None
        self.t0: float | None = None
        self.device_info_changed = None
        self.series_remaining = 0
        self.series_runs: list[str] = []

        self.dataset = QComboBox()
        self.preset = QComboBox()
        for name in PRESETS:
            self.preset.addItem(name.replace("_", " ").title().replace("Cpu", "CPU").replace("Gpu", "GPU"), name)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.preset_note = QLabel("")
        self.preset_note.setWordWrap(True)
        self.preset_note.setMinimumHeight(2 * self.preset_note.fontMetrics().lineSpacing() + 4)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.batch = QSpinBox()
        self.batch.setRange(0, 512)
        self.batch.setSpecialValueText("Automatic")
        self.batch.setToolTip("Images per training step. 'Automatic' (0) picks the largest size that fits in GPU "
                              "memory (8 on CPU).")
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setRange(1e-5, 0.5)
        self.lr.setSingleStep(1e-4)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(1234)
        self.loss = QComboBox()
        self.loss.addItem("Cross-entropy + Dice (good for rare features)", "cross_entropy_dice")
        self.loss.addItem("Cross-entropy only", "cross_entropy")
        self.device = QComboBox()
        self.device.addItem("Automatic (GPU if available)", "auto")
        self.device.addItem("CPU", "cpu")
        self.device.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.precision = QComboBox()
        self.precision.addItem("Automatic (on for GPU)", "auto")
        self.precision.addItem("Off (full precision)", "off")
        self.precision.addItem("On (GPU only)", "on")
        self.precision.setToolTip("Mixed precision roughly halves GPU memory use and speeds up training on modern "
                                  "NVIDIA GPUs. It is recorded with the run.")
        self.in_memory = QSpinBox()
        self.in_memory.setRange(0, 100000)
        self.in_memory.setSpecialValueText("All")
        self.in_memory.setToolTip("How many training images are kept in memory at once. 'All' is fastest. For "
                                  "datasets larger than the computer's memory, set a number: images are then processed "
                                  "in blocks of that size (recorded with the run, so results stay reproducible).")
        self.repeats = QSpinBox()
        self.repeats.setRange(1, 20)
        self.repeats.setToolTip("Train this many runs one after another with seeds seed, seed+1, ... to see how much "
                                "the result depends on chance. Compare them in step 5 (mean and spread are shown).")
        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setSpecialValueText("Off")
        self.patience.setToolTip("Stop when the validation score has not improved for this many epochs. The best "
                                 "checkpoint is kept and evaluated as usual.")
        self.device_info = QLabel("Detecting devices...")
        self.device_info.setWordWrap(True)
        self.start_btn = QPushButton("Start training")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)

        setup = QGroupBox("Setup")
        setup_outer = QVBoxLayout(setup)
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        setup_outer.addWidget(form_host)
        form.addRow("Dataset version:", self.dataset)
        form.addRow("Preset:", self.preset)
        form.addRow(self.preset_note)
        form.addRow("Epochs:", self.epochs)
        form.addRow("Batch size:", self.batch)
        form.addRow("Learning rate:", self.lr)
        form.addRow("Loss:", self.loss)
        form.addRow("Random seed:", self.seed)
        form.addRow("Repeat with seeds:", self.repeats)
        form.addRow("Device:", self.device)
        form.addRow("Mixed precision:", self.precision)
        form.addRow("Early stopping after:", self.patience)
        form.addRow("Images in memory:", self.in_memory)
        setup_outer.addWidget(self.device_info)
        self.ready = QLabel("")
        self.ready.setWordWrap(True)
        self.ready.setTextFormat(Qt.TextFormat.RichText)
        self.ready.setStyleSheet("QLabel { background: #f6f6f4; border: 1px solid #d8d8d4; padding: 6px; }")
        setup_outer.addWidget(self.ready)
        self.summary_card = QLabel("")
        self.summary_card.setWordWrap(True)
        self.summary_card.setTextFormat(Qt.TextFormat.RichText)
        self.summary_card.setStyleSheet("QLabel { background: #eef4fb; border: 1px solid #c9dcf2; padding: 6px; }")
        setup_outer.addWidget(self.summary_card)
        self.estimate: dict | None = None
        self.estimating = False
        self.estimate_timer = QTimer(self)
        self.estimate_timer.setSingleShot(True)
        self.estimate_timer.setInterval(600)
        self.estimate_timer.timeout.connect(self.refresh_estimate)
        for w in (self.epochs, self.batch, self.preset, self.device, self.dataset):
            sig = getattr(w, "valueChanged", None) or w.currentIndexChanged
            sig.connect(self.schedule_estimate)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        setup_outer.addLayout(buttons)
        setup_outer.addStretch(1)
        self.form = form
        self.blocking: list[str] = []
        self.set_advanced(ctx.advanced)
        ctx.mode_changed.connect(self.set_advanced)

        self.progress = QProgressBar()
        self.progress.setFormat("%p%")
        self.status = QLabel("Not running.")
        self.status.setWordWrap(True)
        self.resources = QLabel("")
        self.resources.setWordWrap(True)
        self.loss_chart = LineChart("Loss (lower is better)", "loss")
        self.iou_chart = LineChart("Validation mean IoU of feature classes (higher is better)", "IoU", (0.0, 1.0))
        charts = QHBoxLayout()
        charts.addWidget(self.loss_chart)
        charts.addWidget(self.iou_chart)

        # prediction evolution tab
        self.sample_pick = QComboBox()
        self.sample_pick.currentIndexChanged.connect(self._show_evolution)
        self.epoch_slider = QSlider(Qt.Orientation.Horizontal)
        self.epoch_slider.setRange(1, 1)
        self.epoch_slider.valueChanged.connect(self._show_evolution)
        self.epoch_label = QLabel("epoch -")
        self.truth_view = ImageView("Ground truth")
        self.pred_view = ImageView("Prediction")
        evo = QWidget()
        ev = QVBoxLayout(evo)
        pick = QHBoxLayout()
        pick.addWidget(QLabel("Image:"))
        pick.addWidget(self.sample_pick, 1)
        pick.addWidget(self.epoch_label)
        pick.addWidget(self.epoch_slider, 2)
        views = QHBoxLayout()
        views.addWidget(self.truth_view, 2)
        views.addWidget(self.pred_view, 1)
        ev.addLayout(pick)
        ev.addWidget(QLabel("Left: image | ground truth. Right: prediction at the chosen epoch."))
        ev.addLayout(views, 1)

        # resources tab
        self.util_chart = LineChart("Utilisation (%)", "%", (0.0, 100.0), x_label="minutes")
        self.ram_bar = MemoryBar("Training process RAM")
        self.sys_ram_bar = MemoryBar("Computer RAM in use")
        self.vram_bar = MemoryBar("GPU memory")
        self.throughput = QLabel("")
        self.sim_label = QLabel("")
        res = QWidget()
        rl = QVBoxLayout(res)
        grid = QGridLayout()
        for i, w in enumerate((self.ram_bar, self.sys_ram_bar, self.vram_bar)):
            grid.addWidget(w, i, 0)
        rl.addLayout(grid)
        rl.addWidget(self.throughput)
        rl.addWidget(self.sim_label)
        rl.addWidget(self.util_chart, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(evo, "Prediction evolution")
        self.tabs.addTab(res, "Computer resources")

        right = QVBoxLayout()
        right.addWidget(self.progress)
        right.addWidget(self.status)
        right.addWidget(self.resources)
        right.addLayout(charts, 1)
        right.addWidget(self.tabs, 1)
        body = QHBoxLayout()
        setup_scroll = QScrollArea()  # Advanced mode has many rows; scroll rather than clip the checklist
        setup_scroll.setWidgetResizable(True)
        setup_scroll.setWidget(setup)
        setup_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        setup_scroll.setMinimumWidth(440)
        setup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body.addWidget(setup_scroll)
        body.addLayout(right, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(hint(HELP))
        layout.addLayout(body, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.poll)
        ctx.datasets_changed.connect(self.refresh)
        self._apply_preset()
        self.refresh()
        self._detect_devices()

    # ----- setup -------------------------------------------------------------------------------------------
    EXPERT_FIELDS = ("batch", "lr", "loss", "seed", "repeats", "precision", "patience", "in_memory")

    def set_advanced(self, on: bool) -> None:
        """Beginner mode shows dataset, preset, epochs and device only; Advanced shows every training option."""
        for name in self.EXPERT_FIELDS:
            self.form.setRowVisible(getattr(self, name), on)

    def schedule_estimate(self, *_args) -> None:
        if self.dataset.currentData():
            self.estimate_timer.start()

    def refresh_estimate(self) -> None:
        """Estimate this run in the background (one timed training step) and show the summary card."""
        if self.is_running() or self.estimating:
            return
        try:
            cfg = self.build_config()
        except Exception as exc:  # invalid settings are explained by the checklist / start button
            self.summary_card.setText(f"<b>This run:</b> cannot be estimated ({exc}).")
            return
        ws = self.ctx.workspace
        self.estimating = True
        self.summary_card.setText("<b>This run:</b> estimating...")

        def work() -> dict:
            from ..training.estimate import estimate_run

            return estimate_run(cfg, ws)

        def done(est: dict) -> None:
            self.estimating = False
            self.estimate = est
            mem = est["data_memory_bytes"] / 2**20
            batch = f"{est['batch_size']}" + (" (automatic: checked again when training starts)" if est["auto_batch"]
                                              else "")
            self.summary_card.setText(
                f"<b>This run:</b> {est['epochs']} epochs x {est['steps_per_epoch']} steps; each epoch uses "
                f"{est['patches_per_epoch']} patches of {est['patch_size']} px from {est['train_images']} training "
                f"images; batch size {batch}.<br>Model: U-Net with {est['parameters']:,} parameters on "
                f"{est['device']}.<br><b>Time:</b> about {_fmt_seconds(est['total_seconds'])} "
                f"<span style='color:#6b6b6b'>(rough, from timing one step on this computer)</span>. "
                f"<b>Images in memory:</b> about {mem:.1f} MB.")

        def failed(message: str, _details: str) -> None:
            self.estimating = False
            self.summary_card.setText(f"<b>This run:</b> could not be estimated ({message}).")

        run_background(work, done, failed)

    def update_readiness(self, items: list[tuple[str, str]]) -> None:
        """Show the 'Ready to train?' checklist; items are (state, text) with state ok/warn/todo/block."""
        icons = {"ok": ("&#10003;", "#1a7f37"), "warn": ("&#9888;", "#9a5b00"), "todo": ("&#9675;", "#1f5fae"),
                 "block": ("&#10007;", "#b42318")}
        rows = [f"<span style='color:{icons[st][1]}'>{icons[st][0]}</span> {text}" for st, text in items]
        self.ready.setText("<b>Ready to train?</b><br>" + "<br>".join(rows))
        self.blocking = [text for st, text in items if st == "block"]
        if not self.is_running():
            self.start_btn.setEnabled(not self.blocking)

    def refresh(self) -> None:
        current = self.dataset.currentData()
        self.dataset.clear()
        for v in list_dataset_versions(self.ctx.workspace.datasets_dir):
            counts = v.manifest["split_counts"]
            self.dataset.addItem(f"{v.dataset_id}  (train {counts['train']} / val {counts['val']} / "
                                 f"test {counts['test']})", v.dataset_id)
        idx = self.dataset.findData(current)
        self.dataset.setCurrentIndex(idx if idx >= 0 else self.dataset.count() - 1)

    def _apply_preset(self) -> None:
        values = PRESETS[self.preset.currentData()]
        self.preset_note.setText(values["description"])
        self.epochs.setValue(values["epochs"])
        self.batch.setValue(values["batch_size"])
        self.lr.setValue(values["learning_rate"])

    def _detect_devices(self) -> None:
        def detect() -> str:
            from ..training.device import choose_device, cuda_diagnosis, detect_devices

            labels = [d.label() for d in detect_devices()]
            return "Found: " + "; ".join(labels) + ". " + choose_device("auto")[2] + " " + cuda_diagnosis()

        def show(text: str) -> None:
            self.device_info.setText(text)
            if self.device_info_changed:
                self.device_info_changed()

        run_background(detect, show,
                       lambda msg, _d: show(f"Device detection failed: {msg}. CPU will be used."))

    def build_config(self):
        dataset_id = self.dataset.currentData()
        if not dataset_id:
            raise TrainConfigError("Create a dataset version in step 1 first.")
        overrides = dict(epochs=self.epochs.value(), batch_size=self.batch.value(), learning_rate=self.lr.value(),
                         seed=self.seed.value(), device=self.device.currentData(), loss=self.loss.currentData(),
                         mixed_precision=self.precision.currentData(), images_in_memory=self.in_memory.value(),
                         early_stopping_patience=self.patience.value())
        if self.preprocess is not None:
            prep = self.preprocess()
            overrides.update(resize_factor=prep.resize_factor, normalization=prep.normalization,
                             exclude_border=prep.exclude_border)
        preset = self.preset.currentData()
        augment = self.augment_config()
        if "patch_size" in PRESETS[preset] and augment.patch_size == AugmentConfig().patch_size:
            # the preset's larger patch applies unless the user changed patch size in step 2
            from dataclasses import replace

            augment = replace(augment, patch_size=PRESETS[preset]["patch_size"])
        return preset_config(preset, dataset_id, augment=augment, **overrides)

    # ----- run control -------------------------------------------------------------------------------------
    def is_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        if self.is_running():
            return
        self.series_remaining = self.repeats.value() - 1
        self.series_runs = []
        self._start_one(0)

    def _start_one(self, seed_offset: int) -> None:
        try:
            cfg = self.build_config()
            if seed_offset:
                from dataclasses import replace

                cfg = replace(cfg, seed=cfg.seed + seed_offset)
        except (TrainConfigError, AugmentConfigError, ValueError) as exc:
            self.ctx.error(self, "Cannot start training", str(exc))
            return
        ws = self.ctx.workspace
        run_id = new_run_id()
        requests = ws.runs_dir / ".requests"
        requests.mkdir(parents=True, exist_ok=True)
        cfg_path = requests / f"{run_id}.json"
        payload = cfg.to_dict()
        payload.pop("dataset_id")
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.series_runs.append(run_id)
        self._launch(RunDir(ws.runs_dir / run_id),
                     ["train", cfg.dataset_id, "--config", str(cfg_path), "--run-id", run_id, "--quiet"])

    def resume(self, run: RunDir) -> bool:
        """Continue a stopped run (called from the Results page)."""
        if self.is_running():
            self.ctx.error(self, "Resume", "Another training run is in progress.")
            return False
        if not run.resumable():
            self.ctx.error(self, "Resume", f"Run {run.run_id} cannot be resumed (state: {run.state()}).")
            return False
        self._launch(run, ["resume", run.run_id, "--quiet"])
        return True

    def _launch(self, run: RunDir, args: list[str]) -> None:
        self.run = run
        self.tail = EventTail(run.events)
        self.stderr = ""
        self.final_state = None
        self.t0 = None
        self.fixed_stems, self.epochs_seen = [], []
        for chart in (self.loss_chart, self.iou_chart, self.util_chart):
            chart.clear()
        self.sample_pick.clear()
        self.truth_view.set_pixmap(None)
        self.pred_view.set_pixmap(None)
        self.progress.setValue(0)
        self.status.setText("Starting training process...")

        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        src_root = str(Path(__file__).resolve().parents[2])
        env.insert("PYTHONPATH", os.pathsep.join(p for p in (src_root, env.value("PYTHONPATH")) if p))
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        _hide_console_window(proc)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._finished)
        proc.start(*worker_command(["--workspace", str(self.ctx.workspace.root), *args]))
        self.process = proc
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel")
        self.timer.start()
        self.ctx.status_message.emit(f"Training {'resumed' if args[0] == 'resume' else 'started'}: {run.run_id}")

    def cancel(self) -> None:
        if not self.is_running() or self.run is None:
            return
        if self.cancel_btn.text() == "Cancel":
            self.series_remaining = 0  # cancelling also cancels the remaining repeats
            self.run.request_cancel()
            self.cancel_btn.setText("Stop immediately")
            self.status.setText("Cancelling after the current step...")
        elif self.ctx.confirm(self, "Stop immediately", "Kill the training process now? Work since the last "
                                                        "epoch is lost."):
            self.process.kill()

    def _read_stderr(self) -> None:
        if self.process:
            self.stderr += bytes(self.process.readAllStandardError().data()).decode("utf-8", "replace")
            self.stderr = self.stderr[-20000:]

    def _finished(self, exit_code: int, _status: Any) -> None:
        self.timer.stop()
        self._read_stderr()
        self.poll()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        state = self.run.state() if self.run else None
        if state in ACTIVE_STATES or state == "interrupted":
            self.run.write_status(state="cancelled", total_epochs=self.run.status().get("total_epochs"),
                                  message="Training process was stopped. The last completed epoch can be resumed.")
            state = "cancelled"
        self.final_state = state if state in TERMINAL_STATES else "failed"
        self.poll()
        if self.final_state == "failed":
            message = (self.run.status().get("message") if self.run else None) or "The training process stopped."
            self.status.setText(f"<span style='color:#b42318'><b>Training failed:</b> {message}</span>")
            self.ctx.error(self, "Training failed", message, self.stderr)
        self.ctx.runs_changed.emit()
        if self.final_state == "completed" and self.series_remaining > 0:
            done = len(self.series_runs)
            self.series_remaining -= 1
            self.ctx.status_message.emit(f"Run {done} of {done + self.series_remaining + 1} finished; starting the "
                                         "next seed.")
            self._start_one(done)
        else:
            self.series_remaining = 0

    # ----- live updates ------------------------------------------------------------------------------------
    def poll(self) -> None:
        if self.run is None or self.tail is None:
            return
        for event in self.tail.read_new():
            self._handle(event)
        status = self.run.status()
        state = status.get("state")
        if state in ("running", "starting", "evaluating"):
            epoch, total = status.get("epoch", 0), status.get("total_epochs", 1)
            batch, batches = status.get("batch", 0), max(1, status.get("batches", 1))
            frac = ((max(epoch, 1) - 1) + batch / batches) / max(1, total) if state == "running" else \
                (1.0 if state == "evaluating" else 0.0)
            self.progress.setValue(int(100 * frac))
            loss = f" - current loss {status['loss']:.4f}" if status.get("loss") is not None else ""
            self.status.setText(f"{status.get('message', '')} - batch {batch}/{batches}{loss} - "
                                f"time left about {_fmt_seconds(status.get('eta_s'))}" if state == "running"
                                else status.get("message", ""))
        elif state in TERMINAL_STATES:
            if state == "completed":
                self.progress.setValue(100)
            self.status.setText(f"<b>{state.title()}.</b> {status.get('message', '')}"
                                + (" See step 4 for test results." if state == "completed" else ""))

    def _handle(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind in ("start", "resume"):
            extra = f" Batch size {event['batch_size']}." if "batch_size" in event else ""
            if event.get("mixed_precision"):
                extra += " Mixed precision on."
            self.resources.setText(f"<b>Device:</b> {event['device']}. {event['note']}{extra}")
            info = self.run.info() if self.run else {}
            stems = info.get("fixed_samples", {}).get("stems", [])
            if stems != self.fixed_stems:
                self.fixed_stems = stems
                self.sample_pick.blockSignals(True)
                self.sample_pick.clear()
                self.sample_pick.addItems(stems)
                self.sample_pick.blockSignals(False)
        elif kind == "epoch":
            e = event["epoch"]
            self.loss_chart.add_point("training", e, event["train_loss"])
            self.loss_chart.add_point("validation", e, event["val_loss"])
            self.iou_chart.add_point("validation mIoU", e, event["val_mean_iou_foreground"])
            if e not in self.epochs_seen:
                self.epochs_seen.append(e)
            follow = self.epoch_slider.value() == self.epoch_slider.maximum()
            self.epoch_slider.setRange(1, max(self.epochs_seen))
            if follow:
                self.epoch_slider.setValue(e)
            self._show_evolution()
            self._show_resources(event["resources"], None)
        elif kind == "resources":
            self._show_resources(event, event.get("images_per_s"))

    def _show_resources(self, res: dict[str, Any], images_per_s: float | None) -> None:
        if self.t0 is None:
            self.t0 = res.get("time")
        minutes = ((res.get("time") or 0) - (self.t0 or 0)) / 60.0
        self.ram_bar.set(res.get("process_rss_mb"), None)
        if res.get("system_memory_percent") is not None:
            total = res.get("system_memory_total_mb")
            self.sys_ram_bar.set(res["system_memory_percent"] / 100 * total if total else None, total)
        if res.get("system_cpu_percent") is not None:
            self.util_chart.add_point("CPU", minutes, res["system_cpu_percent"])
        gpus = res.get("gpus") or []
        if gpus:
            g = gpus[0]
            sim = " (SIMULATED)" if g.get("simulated") else ""
            self.vram_bar.set(g.get("memory_used_mb"), g.get("memory_total_mb"), suffix=sim)
            if g.get("utilization_percent") is not None:
                self.util_chart.add_point("GPU" + sim, minutes, g["utilization_percent"])
            self.sim_label.setText("<span style='color:#8a5a00'><b>GPU figures are SIMULATED</b> "
                                   "(test mode, no GPU hardware).</span>" if g.get("simulated") else
                                   f"GPU: {g.get('name')} - temperature {g.get('temperature_c')} C - via "
                                   f"{g.get('source')}")
        elif res.get("torch_gpu_allocated_mb") is not None:
            self.vram_bar.set(res["torch_gpu_allocated_mb"], None, suffix=" allocated by PyTorch")
        else:
            self.vram_bar.set(None, None)
            self.sim_label.setText("GPU: not used for this run.")
        if images_per_s is not None:
            self.throughput.setText(f"Throughput: {images_per_s:.1f} training patches per second. "
                                    "All figures are measured on this computer unless marked SIMULATED.")

    def _show_evolution(self) -> None:
        if not self.run or not self.sample_pick.currentText() or not self.epochs_seen:
            return
        stem, epoch = self.sample_pick.currentText(), self.epoch_slider.value()
        self.epoch_label.setText(f"epoch {epoch}")
        truth = self.run.samples / f"truth_{stem}.png"
        pred = self.run.samples / f"epoch_{epoch:03d}" / f"{stem}.png"
        if truth.exists():
            self.truth_view.set_pixmap(QPixmap(str(truth)))
        if pred.exists():
            self.pred_view.set_pixmap(QPixmap(str(pred)))

    def shutdown(self) -> None:
        """Called when the window closes: stop a running training process."""
        if self.is_running() and self.run is not None:
            self.run.request_cancel()
            if not self.process.waitForFinished(15000):
                self.process.kill()
                self.process.waitForFinished(5000)
