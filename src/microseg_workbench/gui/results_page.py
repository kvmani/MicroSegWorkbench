"""Step 4: review runs, test-set metrics, per-class results and overlays."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..training.runs import RunDir, list_runs
from .context import AppContext
from .widgets import ImageView, hint, run_background

HELP = ("<b>Step 4 - Results.</b> Each training run keeps its exact settings, dataset version, software versions "
        "and checkpoints. Metrics here come from the <i>test</i> split, which the model never saw during training "
        "or model selection. IoU and Dice: 1 is perfect, 0 is no overlap; 'n/a' means the class occurs in neither "
        "truth nor prediction. Results are <b>unreviewed</b> until a scientist approves them.")
UNREVIEWED = ("<span style='color:#8a5a00'><b>&#9888; UNREVIEWED</b> - not approved for scientific use or "
              "deployment.</span>")


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


class ResultsPage(QWidget):
    def __init__(self, ctx: AppContext, resume: Callable[[RunDir], bool] | None = None) -> None:
        super().__init__()
        self.ctx = ctx
        self.resume_callback = resume
        self.last_prediction: dict | None = None
        self.predicting = False
        self.runs = QListWidget()
        self.runs.currentItemChanged.connect(self._show_run)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.open_btn = QPushButton("Open run folder")
        self.open_btn.clicked.connect(self._open_folder)
        self.resume_btn = QPushButton("Resume training")
        self.resume_btn.setToolTip("Continue a cancelled, failed or interrupted run from its last completed epoch.")
        self.resume_btn.clicked.connect(self._resume)
        self.predict_btn = QPushButton("Segment new images...")
        self.predict_btn.setToolTip("Apply this run's best model to a folder of new micrographs.")
        self.predict_btn.clicked.connect(self._predict_dialog)

        self.banner = QLabel("")
        self.details = QLabel("Select a run.")
        self.details.setWordWrap(True)
        self.metrics = QTableWidget(0, 7)
        self.metrics.setHorizontalHeaderLabels(["Class", "IoU", "Dice", "Precision", "Recall", "True pixels",
                                                "Predicted pixels"])
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history = QTableWidget(0, 4)
        self.history.setHorizontalHeaderLabels(["Epoch", "Training loss", "Validation loss", "Validation mIoU (features)"])
        self.history.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.overlay_pick = QComboBox()
        self.overlay_pick.currentIndexChanged.connect(self._show_overlay)
        self.overlay = ImageView("Test image | ground truth | prediction")
        overlays = QWidget()
        ov = QVBoxLayout(overlays)
        ov.addWidget(self.overlay_pick)
        ov.addWidget(self.overlay, 1)
        tabs = QTabWidget()
        tabs.addTab(self.metrics, "Per-class test metrics")
        tabs.addTab(overlays, "Test overlays")
        tabs.addTab(self.history, "Training history")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("Training runs (newest first):"))
        lv.addWidget(self.runs, 1)
        row = QHBoxLayout()
        row.addWidget(refresh)
        row.addWidget(self.open_btn)
        lv.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(self.resume_btn)
        row2.addWidget(self.predict_btn)
        lv.addLayout(row2)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.banner)
        rv.addWidget(self.details)
        rv.addWidget(tabs, 1)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 750])
        layout = QVBoxLayout(self)
        layout.addWidget(hint(HELP))
        layout.addWidget(splitter, 1)
        ctx.runs_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        current = self.runs.currentItem().data(256) if self.runs.currentItem() else None
        self.runs.clear()
        for run in list_runs(self.ctx.workspace.runs_dir):
            state = run.state()
            item = QListWidgetItem(f"{run.run_id}\n   {state} - {run.info().get('dataset', {}).get('dataset_id', '?')}")
            item.setData(256, str(run.root))
            self.runs.addItem(item)
            if str(run.root) == current:
                self.runs.setCurrentItem(item)
        if self.runs.currentItem() is None and self.runs.count():
            self.runs.setCurrentRow(0)

    def selected_run(self) -> RunDir | None:
        item = self.runs.currentItem()
        from pathlib import Path

        return RunDir(Path(item.data(256))) if item else None

    def _show_run(self, *_args) -> None:
        run = self.selected_run()
        self.metrics.setRowCount(0)
        self.history.setRowCount(0)
        self.overlay_pick.clear()
        self.overlay.set_pixmap(None)
        self.resume_btn.setEnabled(bool(run and run.resumable()))
        self.predict_btn.setEnabled(bool(run and (run.checkpoints / "best.pt").exists()))
        if run is None:
            return
        info, status = run.info(), run.status()
        cfg = info.get("config", {})
        evaluation_path = run.evaluation / "evaluation.json"
        report = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
        self.banner.setText(UNREVIEWED if report else "")
        lines = [
            f"<b>State:</b> {run.state()} - {status.get('message', '')}"
            + (" <i>This run can be resumed.</i>" if run.resumable() else ""),
            f"<b>Dataset:</b> {info.get('dataset', {}).get('dataset_id')} &nbsp; <b>Preset:</b> {cfg.get('preset')} "
            f"&nbsp; <b>Epochs:</b> {cfg.get('epochs')} &nbsp; <b>Seed:</b> {cfg.get('seed')}",
            f"<b>Device:</b> {info.get('device', {}).get('name')} ({info.get('device', {}).get('kind')}) &nbsp; "
            f"<b>Software:</b> {info.get('tool', {}).get('id')} {info.get('tool', {}).get('version')}, "
            f"torch {info.get('environment', {}).get('torch')}",
        ]
        if report and report.get("metrics"):
            m = report["metrics"]
            lines.append(f"<b>{report['split'].title()} split</b> ({len(report['images'])} images, checkpoint from "
                         f"epoch {report['checkpoint_epoch']}): mean IoU {_f(m['mean_iou'])}, feature mean IoU "
                         f"{_f(m['mean_iou_foreground'])}, pixel accuracy {_f(m['pixel_accuracy'])}")
            if report.get("note"):
                lines.append(f"<i>{report['note']}</i>")
            self.metrics.setRowCount(len(m["per_class"]))
            for r, c in enumerate(m["per_class"]):
                for col, v in enumerate([c["name"], _f(c["iou"]), _f(c["dice"]), _f(c["precision"]),
                                         _f(c["recall"]), str(c["true_pixels"]), str(c["predicted_pixels"])]):
                    self.metrics.setItem(r, col, QTableWidgetItem(v))
            self.metrics.resizeColumnsToContents()
            for stem in report["images"]:
                self.overlay_pick.addItem(stem)
        elif status.get("state") == "completed":
            lines.append("No evaluation found.")
        self.details.setText("<br>".join(lines))
        epochs = run.epoch_history()
        self.history.setRowCount(len(epochs))
        for r, e in enumerate(epochs):
            for col, v in enumerate([str(e["epoch"]), _f(e["train_loss"]), _f(e["val_loss"]),
                                     _f(e["val_mean_iou_foreground"])]):
                self.history.setItem(r, col, QTableWidgetItem(v))

    def _show_overlay(self) -> None:
        run = self.selected_run()
        stem = self.overlay_pick.currentText()
        if run and stem:
            self.overlay.set_pixmap(QPixmap(str(run.evaluation / f"{stem}.png")))

    def _open_folder(self) -> None:
        run = self.selected_run()
        if run:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.root)))

    def _resume(self) -> None:
        run = self.selected_run()
        if run and self.resume_callback and self.resume_callback(run):
            self.ctx.status_message.emit(f"Resumed {run.run_id}; see step 3 for progress.")

    def _predict_dialog(self) -> None:
        run = self.selected_run()
        if not run:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder with new micrographs")
        if not folder:
            return
        out = QFileDialog.getSaveFileName(self, "Name of the new results folder",
                                          str(Path(folder).parent / f"predictions-{run.run_id}"))[0]
        if out:
            self.predict(run, Path(folder), Path(out))

    def predict(self, run: RunDir, input_folder: Path, output_folder: Path) -> None:
        from ..training.predict import predict_folder

        self.predicting = True
        self.predict_btn.setEnabled(False)
        self.ctx.status_message.emit(f"Segmenting images in {input_folder}...")

        def done(report: dict) -> None:
            self.predicting = False
            self.last_prediction = report
            self.predict_btn.setEnabled(True)
            skipped = f"; {len(report['errors'])} skipped" if report["errors"] else ""
            self.ctx.status_message.emit(f"{len(report['images'])} image(s) segmented into {output_folder}{skipped}. "
                                         "Results are unreviewed.")
            if self.ctx.interactive:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_folder)))

        def failed(message: str, details: str) -> None:
            self.predicting = False
            self.predict_btn.setEnabled(True)
            self.ctx.error(self, "Segment new images", message, details)

        run_background(lambda: predict_folder(run, input_folder, output_folder), done, failed)
