"""Step 3: train a model and watch it learn.

Training runs in a separate Python process (``msw train``) so the window stays
responsive and a crash in training cannot take the UI down. This page tails the
run's ``events.jsonl`` / ``status.json`` files - the same records the CLI writes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from ..core.augment import AugmentConfig, AugmentConfigError
from ..core.dataset import list_dataset_versions
from ..training.config import PRESETS, TrainConfigError, preset_config
from ..training.runs import TERMINAL_STATES, EventTail, RunDir, new_run_id
from .context import AppContext
from .widgets import ImageView, LineChart, hint, run_background

HELP = ("<b>Step 3 - Training.</b> Choose a dataset version and a preset. <i>Quick CPU</i> trains a tiny model in "
        "about a minute on any computer; <i>Standard</i> is meant for a GPU. The charts update after every epoch, "
        "and the images below show how the model's prediction on the same validation images improves. "
        "<b>Cancel</b> stops after the current step and keeps the last saved checkpoint.")


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


class TrainPage(QWidget):
    def __init__(self, ctx: AppContext, augment_config: Callable[[], AugmentConfig]) -> None:
        super().__init__()
        self.ctx = ctx
        self.augment_config = augment_config
        self.process: QProcess | None = None
        self.run: RunDir | None = None
        self.tail: EventTail | None = None
        self.stderr = ""
        self.fixed_stems: list[str] = []
        self.epochs_seen: list[int] = []
        self.final_state: str | None = None

        self.dataset = QComboBox()
        self.preset = QComboBox()
        for name, values in PRESETS.items():
            self.preset.addItem(name.replace("_", " ").title().replace("Cpu", "CPU"), name)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.preset_note = QLabel("")
        self.preset_note.setWordWrap(True)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.batch = QSpinBox()
        self.batch.setRange(1, 512)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setRange(1e-5, 0.5)
        self.lr.setSingleStep(1e-4)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(1234)
        self.device = QComboBox()
        self.device.addItem("Automatic (GPU if available)", "auto")
        self.device.addItem("CPU", "cpu")
        self.device.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device_info = QLabel("Detecting devices...")
        self.device_info.setWordWrap(True)
        self.start_btn = QPushButton("Start training")
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)

        setup = QGroupBox("Setup")
        form = QFormLayout(setup)
        form.addRow("Dataset version:", self.dataset)
        form.addRow("Preset:", self.preset)
        form.addRow(self.preset_note)
        form.addRow("Epochs:", self.epochs)
        form.addRow("Batch size:", self.batch)
        form.addRow("Learning rate:", self.lr)
        form.addRow("Random seed:", self.seed)
        form.addRow("Device:", self.device)
        form.addRow(self.device_info)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.cancel_btn)
        form.addRow(buttons)

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

        self.sample_pick = QComboBox()
        self.sample_pick.currentIndexChanged.connect(self._show_evolution)
        self.epoch_slider = QSlider(Qt.Orientation.Horizontal)
        self.epoch_slider.setRange(1, 1)
        self.epoch_slider.valueChanged.connect(self._show_evolution)
        self.epoch_label = QLabel("epoch -")
        self.truth_view = ImageView("Ground truth")
        self.pred_view = ImageView("Prediction")
        evo = QGroupBox("Prediction evolution on fixed validation images (left: image | truth, right: prediction)")
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
        ev.addLayout(views, 1)

        right = QVBoxLayout()
        right.addWidget(self.progress)
        right.addWidget(self.status)
        right.addWidget(self.resources)
        right.addLayout(charts, 1)
        right.addWidget(evo, 1)
        body = QHBoxLayout()
        body.addWidget(setup)
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
            from ..training.device import choose_device, detect_devices

            labels = [d.label() for d in detect_devices()]
            return "Found: " + "; ".join(labels) + ". " + choose_device("auto")[2]

        run_background(detect, self.device_info.setText,
                       lambda msg, _d: self.device_info.setText(f"Device detection failed: {msg}. CPU will be used."))

    def build_config(self):
        dataset_id = self.dataset.currentData()
        if not dataset_id:
            raise TrainConfigError("Create a dataset version in step 1 first.")
        return preset_config(self.preset.currentData(), dataset_id, epochs=self.epochs.value(),
                             batch_size=self.batch.value(), learning_rate=self.lr.value(), seed=self.seed.value(),
                             device=self.device.currentData(), augment=self.augment_config())

    # ----- run control -------------------------------------------------------------------------------------
    def is_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        if self.is_running():
            return
        try:
            cfg = self.build_config()
        except (TrainConfigError, AugmentConfigError) as exc:
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

        self.run = RunDir(ws.runs_dir / run_id)
        self.tail = EventTail(self.run.events)
        self.stderr = ""
        self.final_state = None
        self.fixed_stems, self.epochs_seen = [], []
        self.loss_chart.clear()
        self.iou_chart.clear()
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
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._finished)
        proc.start(sys.executable, ["-m", "microseg_workbench", "--workspace", str(ws.root), "train",
                                    cfg.dataset_id, "--config", str(cfg_path), "--run-id", run_id, "--quiet"])
        self.process = proc
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel")
        self.timer.start()
        self.ctx.status_message.emit(f"Training started: {run_id}")

    def cancel(self) -> None:
        if not self.is_running() or self.run is None:
            return
        if self.cancel_btn.text() == "Cancel":
            self.run.request_cancel()
            self.cancel_btn.setText("Stop immediately")
            self.status.setText("Cancelling after the current step...")
        elif self.ctx.confirm(self, "Stop immediately", "Kill the training process now? Work since the last "
                                                        "epoch is lost."):
            self.process.kill()
            self.run.write_status(state="cancelled", message="Stopped immediately by user; last epoch checkpoint kept.")

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
        state = self.run.status().get("state") if self.run else None
        self.final_state = state if state in TERMINAL_STATES else "failed"
        if self.final_state == "failed":
            message = (self.run.status().get("message") if self.run else None) or "The training process stopped."
            self.status.setText(f"<span style='color:#b42318'><b>Training failed:</b> {message}</span>")
            self.ctx.error(self, "Training failed", message, self.stderr)
        self.ctx.runs_changed.emit()

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
            self.status.setText(f"{status.get('message', '')} - batch {batch}/{batches} - "
                                f"time left about {_fmt_seconds(status.get('eta_s'))}" if state == "running"
                                else status.get("message", ""))
        elif state in TERMINAL_STATES:
            if state == "completed":
                self.progress.setValue(100)
            self.status.setText(f"<b>{state.title()}.</b> {status.get('message', '')}"
                                + (" See step 4 for test results." if state == "completed" else ""))

    def _handle(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "start":
            self.resources.setText(f"<b>Device:</b> {event['device']}. {event['note']}")
            info = self.run.info() if self.run else {}
            self.fixed_stems = info.get("fixed_samples", {}).get("stems", [])
            self.sample_pick.blockSignals(True)
            self.sample_pick.clear()
            self.sample_pick.addItems(self.fixed_stems)
            self.sample_pick.blockSignals(False)
        elif kind == "epoch":
            e = event["epoch"]
            self.loss_chart.add_point("training", e, event["train_loss"])
            self.loss_chart.add_point("validation", e, event["val_loss"])
            self.iou_chart.add_point("validation mIoU", e, event["val_mean_iou_foreground"])
            res = event["resources"]
            parts = [f"epoch time {event['epoch_seconds']:.1f}s"]
            if "process_rss_mb" in res:
                parts.append(f"training process RAM {res['process_rss_mb']:.0f} MB")
                parts.append(f"system CPU {res['system_cpu_percent']:.0f}%")
            if "gpu_allocated_mb" in res:
                parts.append(f"GPU memory {res['gpu_allocated_mb']:.0f} MB (peak {res['gpu_peak_allocated_mb']:.0f})")
            else:
                parts.append("GPU: not used")
            if event.get("is_best"):
                parts.append(f"best so far (epoch {e})")
            self.resources.setText(" | ".join(parts) + " (measured)")
            self.epochs_seen.append(e)
            follow = self.epoch_slider.value() == self.epoch_slider.maximum()
            self.epoch_slider.setRange(1, e)
            if follow:
                self.epoch_slider.setValue(e)
            self._show_evolution()

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
