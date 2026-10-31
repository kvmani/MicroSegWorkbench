"""Step 5: compare training runs side by side."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..training.compare import comparison_csv, compare_runs
from ..training.runs import RunDir, list_runs
from .context import AppContext
from .widgets import SERIES_COLORS, ImageView, LineChart, hint

HELP = ("<b>Step 5 - Compare.</b> Tick two or more runs to see their settings and scores side by side. Only the "
        "settings that differ are listed. Test scores are comparable only between runs on the <i>same dataset "
        "version</i>; otherwise a warning is shown. The curve chart shows up to four runs.")

ROWS = [("Dataset version", "dataset_id"), ("State", "state"), ("Preset", "preset"), ("Epochs done", "epochs_done"),
        ("Learning rate", "learning_rate"), ("Batch size", "batch_size"), ("Loss", "loss"), ("Seed", "seed"),
        ("Model", "model"), ("Parameters", "parameters"), ("Device", "device"), ("Training time (s)", "train_seconds"),
        ("Best validation epoch", "best_val_epoch"), ("Best validation mIoU (features)", "best_val_miou_fg"),
        ("Test mIoU", "test_miou"), ("Test mIoU (features)", "test_miou_fg"),
        ("Test pixel accuracy", "test_pixel_accuracy")]


def _cell(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


class ComparePage(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.comparison: dict | None = None
        self.runs = QListWidget()
        self.runs.itemChanged.connect(lambda _item: self.update_comparison())
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        export = QPushButton("Export CSV...")
        export.clicked.connect(self._export)
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diff = QTableWidget(0, 0)
        self.diff.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.chart = LineChart("Validation mean IoU of feature classes per epoch", "IoU", (0.0, 1.0))
        self.image_pick = QComboBox()
        self.image_pick.currentIndexChanged.connect(self._show_images)
        self.views_box = QHBoxLayout()
        overlays = QWidget()
        ov = QVBoxLayout(overlays)
        ov.addWidget(QLabel("Same test image, each run's prediction (only for runs on the same dataset version):"))
        ov.addWidget(self.image_pick)
        ov.addLayout(self.views_box, 1)
        self.views: list[ImageView] = []

        tabs = QTabWidget()
        tabs.addTab(self.table, "Scores and settings")
        tabs.addTab(self.diff, "What differs")
        tabs.addTab(self.chart, "Validation curves")
        tabs.addTab(overlays, "Test predictions")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("Tick runs to compare:"))
        lv.addWidget(self.runs, 1)
        row = QHBoxLayout()
        row.addWidget(refresh)
        row.addWidget(export)
        lv.addLayout(row)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.warnings)
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
        checked = set(self.checked_ids())
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in list_runs(self.ctx.workspace.runs_dir):
            item = QListWidgetItem(f"{run.run_id}  ({run.state()})")
            item.setData(256, str(run.root))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if run.run_id in checked else Qt.CheckState.Unchecked)
            self.runs.addItem(item)
        self.runs.blockSignals(False)
        self.update_comparison()

    def checked_ids(self) -> list[str]:
        return [Path(self.runs.item(i).data(256)).name for i in range(self.runs.count())
                if self.runs.item(i).checkState() == Qt.CheckState.Checked]

    def set_checked(self, run_ids: list[str]) -> None:
        self.runs.blockSignals(True)
        for i in range(self.runs.count()):
            item = self.runs.item(i)
            item.setCheckState(Qt.CheckState.Checked if Path(item.data(256)).name in run_ids
                               else Qt.CheckState.Unchecked)
        self.runs.blockSignals(False)
        self.update_comparison()

    def update_comparison(self) -> None:
        ids = self.checked_ids()
        self.chart.clear()
        self.table.clear()
        self.diff.clear()
        self.image_pick.clear()
        if len(ids) < 2:
            self.comparison = None
            self.warnings.setText("Tick at least two runs.")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.diff.setRowCount(0)
            self.diff.setColumnCount(0)
            return
        runs = [RunDir(self.ctx.workspace.runs_dir / r) for r in ids]
        c = compare_runs(runs)
        self.comparison = c
        notes = [f"<span style='color:#8a5a00'><b>&#9888;</b> {w}</span>" for w in c["warnings"]]
        if c.get("seed_summary"):
            ss = c["seed_summary"]
            st = ss["test_miou_fg"] or ss["test_miou"]
            per_class = ", ".join(f"{n} {v['mean']:.3f} &plusmn; {v['sd']:.3f}" for n, v in ss["test_class_iou"].items()
                                  if v)
            notes.append(f"<b>These runs differ only by random seed</b> ({len(ss['seeds'])} runs): test mIoU of feature "
                         f"classes {st['mean']:.3f} &plusmn; {st['sd']:.3f} (min {st['min']:.3f}, max {st['max']:.3f})."
                         f" Per class: {per_class}. A large spread means single-run differences are not meaningful.")
        notes.append("All scores are <b>unreviewed</b>.")
        self.warnings.setText("<br>".join(notes))
        summaries = c["runs"]
        class_names = sorted({n for s in summaries for n in s["test_class_iou"]})
        rows = ROWS + [(f"Test IoU: {n}", ("class", n)) for n in class_names]
        self.table.setColumnCount(len(summaries))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels([s["run_id"] for s in summaries])
        self.table.setVerticalHeaderLabels([label for label, _ in rows])
        for col, s in enumerate(summaries):
            for row, (_label, key) in enumerate(rows):
                value = s["test_class_iou"].get(key[1]) if isinstance(key, tuple) else s[key]
                self.table.setItem(row, col, QTableWidgetItem(_cell(value)))
        self.table.resizeColumnsToContents()
        diffs = c["differences"]
        self.diff.setColumnCount(len(summaries))
        self.diff.setRowCount(len(diffs))
        self.diff.setHorizontalHeaderLabels([s["run_id"] for s in summaries])
        self.diff.setVerticalHeaderLabels(list(diffs))
        for row, values in enumerate(diffs.values()):
            for col, v in enumerate(values):
                self.diff.setItem(row, col, QTableWidgetItem(_cell(v)))
        self.diff.resizeColumnsToContents()
        for s in summaries[: len(SERIES_COLORS)]:
            self.chart.set_series(s["run_id"], s["val_curve"])
        if len(summaries) > len(SERIES_COLORS):
            self.chart.title = (f"Validation mean IoU of feature classes per epoch (first {len(SERIES_COLORS)} "
                                "of the ticked runs)")
        if c["comparable"]:
            first = runs[0].evaluation / "evaluation.json"
            if first.exists():
                import json

                for stem in json.loads(first.read_text()).get("images", []):
                    self.image_pick.addItem(stem)

    def _show_images(self) -> None:
        while self.views_box.count():
            w = self.views_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.views = []
        stem = self.image_pick.currentText()
        if not stem:
            return
        for run_id in self.checked_ids()[:4]:
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.addWidget(QLabel(run_id))
            view = ImageView()
            path = self.ctx.workspace.runs_dir / run_id / "evaluation" / f"{stem}.png"
            if path.exists():
                view.set_pixmap(QPixmap(str(path)))
            bl.addWidget(view, 1)
            self.views.append(view)
            self.views_box.addWidget(box)

    def _export(self) -> None:
        if not self.comparison:
            self.ctx.error(self, "Export", "Tick at least two runs first.")
            return
        path = QFileDialog.getSaveFileName(self, "Save comparison", "comparison.csv", "CSV (*.csv)")[0]
        if path:
            Path(path).write_text(comparison_csv(self.comparison), encoding="utf-8")
