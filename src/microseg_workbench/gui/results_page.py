"""Step 4: review runs, test-set metrics, per-class results and overlays."""

from __future__ import annotations

import json

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..training.runs import RunDir, list_runs
from .context import AppContext
from .widgets import ImageView, hint

HELP = ("<b>Step 4 - Results.</b> Each training run keeps its exact settings, dataset version, software versions "
        "and checkpoints. Metrics here come from the <i>test</i> split, which the model never saw during training "
        "or model selection. IoU and Dice: 1 is perfect, 0 is no overlap; 'n/a' means the class occurs in neither "
        "truth nor prediction. Results are <b>unreviewed</b> until a scientist approves them.")
UNREVIEWED = ("<span style='color:#8a5a00'><b>&#9888; UNREVIEWED</b> - not approved for scientific use or "
              "deployment.</span>")


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


class ResultsPage(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.runs = QListWidget()
        self.runs.currentItemChanged.connect(self._show_run)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.open_btn = QPushButton("Open run folder")
        self.open_btn.clicked.connect(self._open_folder)

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
            state = run.status().get("state", "unknown")
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
        if run is None:
            return
        info, status = run.info(), run.status()
        cfg = info.get("config", {})
        evaluation_path = run.evaluation / "evaluation.json"
        report = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
        self.banner.setText(UNREVIEWED if report else "")
        lines = [
            f"<b>State:</b> {status.get('state')} - {status.get('message', '')}",
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
        epochs = [e for e in run.read_events() if e["type"] == "epoch"]
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
