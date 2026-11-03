"""Step 5: compare training runs side by side."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout, QScrollArea, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..training.compare import comparison_csv, compare_runs
from ..training.runs import RunDir, list_runs
from .context import AppContext
from .widgets import SERIES_COLORS, ImageView, LineChart, hint, run_background

HELP = ("<b>Step 5 - Compare.</b> Tick two or more runs to see their settings and scores side by side. Only the "
        "settings that differ are listed. Test scores are comparable only between runs on the <i>same dataset "
        "version</i>; otherwise a warning is shown. The curve chart shows up to four runs.")

ROWS = [("Dataset version", "dataset_id"), ("State", "state"), ("Preset", "preset"), ("Epochs done", "epochs_done"),
        ("Learning rate", "learning_rate"), ("Batch size", "batch_size"), ("Loss", "loss"), ("Seed", "seed"),
        ("Network", "model"), ("Parameters", "parameters"), ("Device", "device"), ("Training time (s)", "train_seconds"),
        ("Best validation epoch", "best_val_epoch"), ("Best validation mIoU (features)", "best_val_miou_fg"),
        ("Test mIoU", "test_miou"), ("Test mIoU (features)", "test_miou_fg"),
        ("Test pixel accuracy", "test_pixel_accuracy")]


def _score(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def network_name(run: RunDir) -> str:
    """Short network name for run lists, e.g. 'U-Net' or 'SegFormer-lite'."""
    title = run.info().get("model", {}).get("architecture_title") or "U-Net (convolutional)"
    return title.split(" (")[0]


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
        tabs.addTab(self._build_same_image_tab(), "Same image, several models")
        self.compare_tabs = tabs

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

    # ---- same image, several models -----------------------------------------------------------------------
    def _build_same_image_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.addWidget(QLabel("Run every ticked model on one image and look at the results side by side. Pick an image "
                             "from a dataset version (its mask is then used as ground truth) or any image file."))
        src = QHBoxLayout()
        self.si_dataset = QComboBox()
        self.si_dataset.currentIndexChanged.connect(self._si_load_samples)
        self.si_sample = QComboBox()
        pick_file = QPushButton("Other image file...")
        pick_file.clicked.connect(self._si_pick_file)
        self.si_run_btn = QPushButton("Run ticked models on this image")
        self.si_run_btn.clicked.connect(self.run_same_image)
        src.addWidget(QLabel("Dataset version:"))
        src.addWidget(self.si_dataset, 2)
        src.addWidget(QLabel("Image:"))
        src.addWidget(self.si_sample, 1)
        src.addWidget(pick_file)
        src.addWidget(self.si_run_btn)
        lay.addLayout(src)
        self.si_file: Path | None = None
        self.si_busy = False
        self.si_result: dict | None = None
        self.si_note = QLabel("")
        self.si_note.setWordWrap(True)
        lay.addWidget(self.si_note)
        grid_host = QWidget()
        self.si_grid = QGridLayout(grid_host)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_host)
        lay.addWidget(scroll, 3)
        self.si_table = QTableWidget(0, 0)
        self.si_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.si_table, 1)
        self.ctx.datasets_changed.connect(self._si_fill_datasets)
        self._si_fill_datasets()
        return tab

    def _si_fill_datasets(self) -> None:
        from ..core.dataset import list_dataset_versions

        self.si_dataset.blockSignals(True)
        self.si_dataset.clear()
        for v in list_dataset_versions(self.ctx.workspace.datasets_dir):
            self.si_dataset.addItem(v.dataset_id, v.dataset_id)
        self.si_dataset.blockSignals(False)
        self._si_load_samples()

    def _si_load_samples(self) -> None:
        from ..core.dataset import DatasetVersion

        self.si_sample.clear()
        dataset_id = self.si_dataset.currentData()
        if dataset_id:
            version = DatasetVersion.open(self.ctx.workspace.datasets_dir / dataset_id)
            for s in sorted(version.samples(), key=lambda s: (s["split"] != "test", s["stem"])):
                self.si_sample.addItem(f"{s['stem']} ({s['split']})", s["stem"])
        self.si_file = None

    def _si_pick_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path = QFileDialog.getOpenFileName(self, "Image to segment", "",
                                           "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)")[0]
        if path:
            self.use_image_file(Path(path))

    def use_image_file(self, path: Path) -> None:
        self.si_file = path
        self.si_note.setText(f"Image file: {path} (no ground truth).")

    def run_same_image(self) -> None:
        from ..core.dataset import DatasetVersion
        from ..training.multi_predict import compare_models_on_image, load_truth

        ids = self.checked_ids()[:6]
        if not ids:
            self.ctx.error(self, "Same image", "Tick one or more trained runs on the left first.")
            return
        runs = [RunDir(self.ctx.workspace.runs_dir / r) for r in ids]
        truth = truth_map = None
        if self.si_file is not None:
            image_path = self.si_file
        else:
            dataset_id, stem = self.si_dataset.currentData(), self.si_sample.currentData()
            if not dataset_id or not stem:
                self.ctx.error(self, "Same image", "Choose a dataset image or an image file.")
                return
            version = DatasetVersion.open(self.ctx.workspace.datasets_dir / dataset_id)
            sample = next(s for s in version.samples() if s["stem"] == stem)
            image_path = version.image_path(sample)
            truth_map = version.class_map
            truth = load_truth(version.mask_path(sample), truth_map)
        self.si_busy = True
        self.si_run_btn.setEnabled(False)
        self.si_note.setText(f"Running {len(runs)} model(s)...")

        def done(result: dict) -> None:
            self.si_busy = False
            self.si_run_btn.setEnabled(True)
            self.si_result = result
            self._si_show(result, truth)

        def failed(message: str, details: str) -> None:
            self.si_busy = False
            self.si_run_btn.setEnabled(True)
            self.si_note.setText(f"<span style='color:#b42318'>{message}</span>")
            self.ctx.error(self, "Same image", message, details)

        run_background(lambda: compare_models_on_image(runs, image_path, truth, truth_map), done, failed)

    def _si_show(self, result: dict, truth) -> None:
        from ..training.multi_predict import render_comparison

        while self.si_grid.count():
            w = self.si_grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        tiles = render_comparison(result, truth)
        cols = min(len(tiles), 5)  # image, truth, two models and the disagreement map fit on one row
        for i, (caption, rgb) in enumerate(tiles):
            box = QWidget()
            bl = QVBoxLayout(box)
            view = ImageView()
            view.setMinimumSize(*((170, 170) if cols == 5 else (220, 220)))
            view.set_array(rgb)
            cap = QLabel(caption.replace("\n", "<br>"))
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            bl.addWidget(view, 1)
            bl.addWidget(cap)
            self.si_grid.addWidget(box, i // cols, i % cols)
        models = result["models"]
        names = sorted({n for m in models for n in m["fractions"]})
        has_truth = any(m.get("metrics") for m in models)
        headers = ["Model", "Network", "Seconds"] + [f"{n} %" for n in names]
        if has_truth:
            headers += ["Feature IoU (this image)"] + [f"IoU {n}" for n in names]
        self.si_table.setColumnCount(len(headers))
        self.si_table.setHorizontalHeaderLabels(headers)
        self.si_table.setRowCount(len(models))
        for r, m in enumerate(models):
            row = [m["run_id"], m["network"], f"{m['seconds']:.2f}"] + \
                  [f"{100 * m['fractions'].get(n, 0):.2f}" for n in names]
            if has_truth:
                met = m.get("metrics") or {}
                per = {c["name"]: c["iou"] for c in met.get("per_class", [])}
                row += [_score(met.get("mean_iou_foreground"))] + [_score(per.get(n)) for n in names]
            for c, v in enumerate(row):
                self.si_table.setItem(r, c, QTableWidgetItem(v))
        self.si_table.resizeColumnsToContents()
        notes = [] if result["same_classes"] else ["The models use different classes; agreement is not computed."]
        if result["agreement"] is not None and len(models) == 2:
            notes.append(f"The two models agree on {100 * result['agreement'][0][1]:.1f}% of pixels.")
        for m in models:
            for w in m.get("input_warnings", []):
                notes.append(f"<span style='color:#9a5b00'>&#9888; {m['run_id']}: {w}.</span>")
        notes.append("Predictions are unreviewed.")
        self.si_note.setText(" ".join(notes))

    def refresh(self) -> None:
        checked = set(self.checked_ids())
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in list_runs(self.ctx.workspace.runs_dir):
            item = QListWidgetItem(f"{run.run_id}  ({run.state()}, {network_name(run)})")
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
