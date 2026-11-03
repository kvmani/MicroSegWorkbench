"""Step 1: choose and check an image/mask folder, then create a dataset version."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListView,
    QListWidget, QListWidgetItem, QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..core.dataset import ValidationReport, create_dataset_version, plan_split, validate_folder
from ..core.imageio import decode_mask, load_image, load_mask_values, to_unit_float
from ..core.render import overlay, tile_grid, to_display_rgb
from ..core.splits import DEFAULT_GROUP_REGEX, SplitConfig, SplitError
from ..core.synthetic import SyntheticSpec, write_synthetic_dataset
from .context import AppContext
from .data_widgets import ClassBars, SplitView
from .widgets import ImageView, array_to_pixmap, hint, run_background

THUMB = 96


def _pair_thumbnail(pair, class_map) -> np.ndarray:
    """Small image + mask overlay for the gallery; problem pairs get a red frame."""
    try:
        image = to_unit_float(load_image(pair.image_path).array)
        step = max(1, int(np.ceil(max(image.shape[:2]) / THUMB)))
        if class_map is not None and pair.ok:
            labels = class_map.channel_lut()[decode_mask(load_mask_values(pair.mask_path), class_map)]
            rgb = overlay(image[::step, ::step], labels[::step, ::step], class_map, alpha=0.4)
        else:
            rgb = to_display_rgb(image[::step, ::step])
    except Exception:
        rgb = np.full((THUMB, THUMB, 3), 230, np.uint8)
    rgb = np.ascontiguousarray(rgb)
    if not pair.ok:
        rgb = rgb.copy()
        rgb[:3], rgb[-3:], rgb[:, :3], rgb[:, -3:] = (180, 35, 24), (180, 35, 24), (180, 35, 24), (180, 35, 24)
    return rgb

HELP = ("<b>Step 1 - Data.</b> Pick a folder with micrographs (<i>name</i>.png/.tif) and masks "
        "(<i>name</i>_mask.png) plus a <i>classes.json</i> that names each mask value. "
        "<b>Check folder</b> verifies pairing, sizes and class values. The <b>group pattern</b> decides which "
        "images belong together (e.g. the same specimen); a whole group always stays in one split so the test "
        "set never contains near-copies of training images. No folder yet? Use <b>Create demo data</b>.")


class DataPage(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.report: ValidationReport | None = None
        self.busy = False
        self.last_version_id: str | None = None

        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Folder with images, *_mask.png masks and classes.json")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        self.demo_btn = QPushButton("Create demo data")
        self.demo_btn.setToolTip("Writes small SYNTHETIC micrographs into the workspace so you can try every step.")
        self.demo_btn.clicked.connect(self.create_demo_data)
        self.import_btn = QPushButton("Import OnlineAnnotator export...")
        self.import_btn.setToolTip("Convert an OnlineAnnotator ML export (ZIP) into a checked folder, keeping class "
                                   "numbers and who annotated/approved each mask.")
        self.import_btn.clicked.connect(self._import_dialog)
        self.regex = QLineEdit(DEFAULT_GROUP_REGEX)
        self.regex.setToolTip("Regular expression; its first (...) group is the specimen/source name. "
                              "Default: text before the first underscore, e.g. S01 in S01_0003.png.")
        self.check_btn = QPushButton("Check folder")
        self.check_btn.setDefault(True)
        self.check_btn.clicked.connect(self.check_folder)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Folder:"))
        row1.addWidget(self.folder, 1)
        row1.addWidget(browse)
        row1.addWidget(self.demo_btn)
        row1.addWidget(self.import_btn)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Group pattern:"))
        row2.addWidget(self.regex, 1)
        row2.addWidget(self.check_btn)

        self.summary = QLabel("No folder checked yet.")
        self.summary.setWordWrap(True)
        self.issues = QListWidget()
        self.issues.setMaximumHeight(90)
        self.issues.setToolTip("Click a problem to show its image.")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Image", "Group", "Size (HxW)", "Class pixels", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._show_selected)
        self.preview = ImageView("Select an image to see it with its mask")
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 90)
        self.opacity.setValue(45)
        self.opacity.setMaximumWidth(180)
        self.opacity.setToolTip("How strongly the mask colours cover the image (0 = image only).")
        self.opacity.valueChanged.connect(self._show_selected)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(THUMB, THUMB))
        self.gallery.setGridSize(QSize(THUMB + 24, THUMB + 30))
        self.gallery.setResizeMode(QListView.ResizeMode.Adjust)
        self.gallery.setMovement(QListView.Movement.Static)
        self.gallery.setWordWrap(True)
        self.gallery.currentRowChanged.connect(self._gallery_selected)
        self.issues.itemClicked.connect(self._issue_clicked)
        self.class_bars = ClassBars()
        self.split_view = SplitView()
        self.thumbs_ready = False

        split_box = QGroupBox("Create dataset version (fixed, reproducible split)")
        form = QFormLayout(split_box)
        self.name = QLineEdit("dataset")
        self.train = self._fraction(0.6)
        self.val = self._fraction(0.2)
        self.test = self._fraction(0.2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(42)
        for w in (self.train, self.val, self.test, self.seed):
            w.valueChanged.connect(self._update_split_preview)
        self.import_warnings: list[str] = []
        self.split_source = QComboBox()
        self.split_source.addItem("Seeded split by group (fractions below)", "group")
        self.split_source.addItem("As assigned in OnlineAnnotator", "annotator")
        self.split_source.setToolTip("Use the train/val/test assignment made in the annotation tool. Groups must "
                                     "not span splits.")
        self.split_source.currentIndexChanged.connect(self._update_split_preview)
        self.split_preview = QLabel("")
        self.split_preview.setWordWrap(True)
        self.create_btn = QPushButton("Create dataset version")
        self.create_btn.setEnabled(False)
        self.create_btn.clicked.connect(self.create_version)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Name:", self.name)
        form.addRow("Split:", self.split_source)
        form.addRow("Train fraction:", self.train)
        form.addRow("Validation fraction:", self.val)
        form.addRow("Test fraction:", self.test)
        form.addRow("Split seed:", self.seed)
        form.addRow(self.split_view)
        form.addRow(self.split_preview)
        form.addRow(self.create_btn)
        self.form = form
        self.set_advanced(ctx.advanced)
        ctx.mode_changed.connect(self.set_advanced)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.summary)
        lv.addWidget(self.class_bars)
        lv.addWidget(self.issues)
        self.views = QTabWidget()
        self.views.addTab(self.gallery, "Pictures")
        self.views.addTab(self.table, "List")
        lv.addWidget(self.views, 1)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.preview, 1)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Mask opacity"))
        orow.addWidget(self.opacity)
        orow.addStretch(1)
        rv.addLayout(orow)
        rv.addWidget(split_box)
        rv.addWidget(self.result)  # outside the form: wrapped labels in a form layout get clipped
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([600, 450])

        layout = QVBoxLayout(self)
        layout.addWidget(hint(HELP))
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(splitter, 1)

    def set_advanced(self, on: bool) -> None:
        """Beginner mode hides the split seed and the text listing of the split (the tiles show it)."""
        self.form.setRowVisible(self.seed, on)
        self.form.setRowVisible(self.split_preview, on)

    @staticmethod
    def _fraction(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, 1.0)
        box.setSingleStep(0.05)
        box.setDecimals(2)
        box.setValue(value)
        return box

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose dataset folder", self.folder.text())
        if folder:
            self.folder.setText(folder)
            self.check_folder()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        for w in (self.check_btn, self.demo_btn, self.import_btn):
            w.setEnabled(not busy)
        self.create_btn.setEnabled(not busy and bool(self.report and self.report.ok))
        if message:
            self.ctx.status_message.emit(message)

    def create_demo_data(self) -> None:
        out = self.ctx.workspace.demo_dir
        self._set_busy(True, "Creating synthetic demo data...")

        def done(_: object) -> None:
            self._set_busy(False, f"Synthetic demo data written to {out}")
            self.folder.setText(str(out))
            self.name.setText("demo")
            self.check_folder()

        run_background(lambda: write_synthetic_dataset(out, SyntheticSpec()), done, self._failed("Demo data"))

    def _import_dialog(self) -> None:
        bundle = QFileDialog.getOpenFileName(self, "OnlineAnnotator export", "", "ZIP files (*.zip)")[0]
        if bundle:
            self.import_bundle(Path(bundle))

    def import_bundle(self, bundle: Path) -> None:
        from ..integrations.onlineannotator import import_bundle

        base = self.ctx.workspace.root / "imports" / bundle.stem
        dest, n = base, 2
        while dest.exists():
            dest, n = base.with_name(f"{base.name}-{n}"), n + 1
        self._set_busy(True, f"Importing {bundle.name}...")

        def done(report) -> None:
            self._set_busy(False, f"Imported {report.images} image(s) into {report.folder}")
            self.import_warnings = report.warnings
            self.folder.setText(str(report.folder))
            self.name.setText(bundle.stem.split("_")[0] or "imported")
            if report.splits and set(report.splits) <= {"train", "val", "test"}:
                self.split_source.setCurrentIndex(self.split_source.findData("annotator"))
            self.check_folder()
            for w in report.warnings:
                self.ctx.error(self, "Imported with warnings", w)

        run_background(lambda: import_bundle(bundle, dest), done, self._failed("Import OnlineAnnotator export"))

    def _failed(self, title: str):
        def handler(message: str, details: str) -> None:
            self._set_busy(False)
            self.ctx.error(self, title, message, details)
        return handler

    def check_folder(self) -> None:
        folder = Path(self.folder.text().strip())
        if not self.folder.text().strip():
            self.ctx.error(self, "Check folder", "Choose a folder first (or create demo data).")
            return
        regex = self.regex.text()
        self._set_busy(True, f"Checking {folder}...")
        self.report = None

        def done(report: ValidationReport) -> None:
            self.report = report
            self._set_busy(False, report.summary())
            self._show_report()
            self.ctx.workflow_changed.emit()

        run_background(lambda: validate_folder(folder, regex), done, self._failed("Check folder"))

    def _show_report(self) -> None:
        report = self.report
        assert report is not None
        classes = ""
        if report.class_map:
            totals = {c.value: sum(p.class_pixels.get(c.value, 0) for p in report.pairs if p.ok)
                      for c in report.class_map.classes}
            images_with = {c.value: sum(1 for p in report.pairs if p.ok and p.class_pixels.get(c.value, 0))
                           for c in report.class_map.classes}
            grand = sum(totals.values()) or 1
            classes = "<br>Classes (mask value = name): " + ", ".join(f"{c.value} = {c.name}"
                                                                       for c in report.class_map.classes)
            n_ok = sum(1 for p in report.pairs if p.ok)
            self.class_bars.set_rows([(c.name, c.color, totals[c.value] / grand, images_with[c.value], n_ok)
                                      for c in report.class_map.classes])
            rare = [c.name for c in report.class_map.classes[1:] if 0 < totals[c.value] / grand < 0.02]
            if rare:
                classes += (f"<br><span style='color:#8a5a00'>Rare classes ({', '.join(rare)}) cover under 2% of "
                            "pixels: prefer the Cross-entropy + Dice loss and check their per-class scores.</span>")
        verdict = ("<span style='color:#1a7f37'><b>Ready.</b></span>" if report.ok else
                   "<span style='color:#b42318'><b>Fix the errors below before creating a dataset.</b></span>")
        self.summary.setText(f"{verdict} {report.summary()}{classes}")
        self.issues.clear()
        for issue in report.issues:
            issue_item = QListWidgetItem(f"{issue.severity.upper()}: {f'[{issue.stem}] ' if issue.stem else ''}{issue.message}")
            issue_item.setForeground(QColor("#b42318" if issue.severity == "error" else "#8a5a00"))
            self.issues.addItem(issue_item)
        if not report.issues:
            self.issues.addItem("No problems found.")
        self.table.setRowCount(len(report.pairs))
        names = {c.value: c.name for c in report.class_map.classes} if report.class_map else {}
        for row, pair in enumerate(report.pairs):
            pixels = ", ".join(f"{names.get(v, v)}: {n}" for v, n in pair.class_pixels.items() if n)
            values = [pair.stem, pair.group, f"{pair.height}x{pair.width}" if pair.height else "-", pixels,
                      "OK" if pair.ok else "Problem"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 4 and not pair.ok:
                    item.setForeground(QColor("#b42318"))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self._fill_gallery()
        if report.pairs:
            first_problem = next((i for i, p in enumerate(report.pairs) if not p.ok), 0)
            self.table.selectRow(first_problem)
        self.create_btn.setEnabled(report.ok)
        self._update_split_preview()

    def _fill_gallery(self) -> None:
        report = self.report
        self.gallery.blockSignals(True)
        self.gallery.clear()
        problems: dict[str, list[str]] = {}
        for issue in report.issues:
            if issue.stem:
                problems.setdefault(issue.stem, []).append(issue.message)
        for pair in report.pairs:
            item = QListWidgetItem(("\u26a0 " if not pair.ok else "") + pair.stem)
            item.setToolTip("\n".join(problems.get(pair.stem, [])) or f"{pair.stem} - group {pair.group}")
            if not pair.ok:
                item.setForeground(QColor("#b42318"))
            self.gallery.addItem(item)
        self.gallery.blockSignals(False)
        self.thumbs_ready = False
        pairs, class_map = list(report.pairs), report.class_map

        def done(thumbs) -> None:
            if self.report is not report:
                return  # a newer check replaced this one
            for i, rgb in enumerate(thumbs):
                if i < self.gallery.count():
                    self.gallery.item(i).setIcon(QIcon(array_to_pixmap(rgb)))
            self.thumbs_ready = True

        run_background(lambda: [_pair_thumbnail(p, class_map) for p in pairs], done,
                       lambda msg, details: setattr(self, "thumbs_ready", True))

    def _gallery_selected(self, row: int) -> None:
        if row >= 0 and row != self._selected_row():
            self.table.selectRow(row)

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else -1

    def _issue_clicked(self, item: QListWidgetItem) -> None:
        if not self.report:
            return
        text = item.text()
        for i, pair in enumerate(self.report.pairs):
            if f"[{pair.stem}]" in text:
                self.table.selectRow(i)
                return

    def _split_config(self) -> SplitConfig:
        return SplitConfig(train=self.train.value(), val=self.val.value(), test=self.test.value(),
                           seed=self.seed.value(), group_regex=self.report.group_regex if self.report else self.regex.text(),
                           source=self.split_source.currentData())

    def _update_split_preview(self) -> None:
        if not self.report or not self.report.ok:
            self.split_preview.setText("")
            if self.report is not None:
                self.split_view.set_error("Fix the problems listed on the left; the split appears here once the "
                                          "folder is ready.")
            else:
                self.split_view.set_plan({})
            return
        try:
            plan = plan_split(self.report, self._split_config())
        except SplitError as exc:
            self.split_preview.setText(f"<span style='color:#b42318'>{exc}</span>")
            self.split_view.set_error(str(exc))
            self.create_btn.setEnabled(False)
            return
        self.create_btn.setEnabled(True)
        for w in (self.train, self.val, self.test, self.seed):
            w.setEnabled(self.split_source.currentData() == "group")
        groups: dict[str, str] = {}
        for p in self.report.pairs:
            if p.ok:
                groups[p.group] = plan[p.stem]
        counts = Counter(plan.values())
        sizes = Counter(p.group for p in self.report.pairs if p.ok)
        self.split_view.set_plan({g: (s, sizes[g]) for g, s in groups.items()})
        lines = []
        for split in ("train", "val", "test"):
            members = sorted(g for g, s in groups.items() if s == split)
            lines.append(f"<b>{split}</b>: {counts.get(split, 0)} images - groups {', '.join(members) or 'none'}")
        self.split_preview.setText("<br>".join(lines))

    def _show_selected(self, *_args) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or not self.report:
            return
        row = rows[0].row()
        if self.gallery.currentRow() != row:
            self.gallery.blockSignals(True)
            self.gallery.setCurrentRow(row)
            self.gallery.blockSignals(False)
        pair = self.report.pairs[row]
        try:
            image = to_unit_float(load_image(pair.image_path).array)
            if self.report.class_map and pair.ok:
                labels = self.report.class_map.channel_lut()[decode_mask(load_mask_values(pair.mask_path),
                                                                         self.report.class_map)]
                alpha = self.opacity.value() / 100
                shown = overlay(image, labels, self.report.class_map, alpha=alpha) if alpha else to_display_rgb(image)
                self.preview.set_array(tile_grid([to_display_rgb(image), shown], columns=2))
            else:
                self.preview.set_array(to_display_rgb(image))
        except Exception as exc:  # unreadable file: say so in the preview instead of crashing
            self.preview.set_pixmap(None)
            self.preview.setText(f"Cannot display: {exc}")

    def create_version(self) -> None:
        if not self.report or not self.report.ok:
            return
        report, cfg, name = self.report, self._split_config(), self.name.text().strip()
        datasets_dir = self.ctx.workspace.datasets_dir
        self._set_busy(True, "Creating dataset version (copying and hashing files)...")

        def done(version) -> None:
            self._set_busy(False, f"Dataset version {version.dataset_id} ready")
            self.last_version_id = version.dataset_id
            counts = version.manifest["split_counts"]
            from ..core.health import dataset_health

            marks = {"ok": ("&#10003;", "#1a7f37"), "info": ("&#8505;", "#1f5fae"), "warn": ("&#9888;", "#9a5b00"),
                     "block": ("&#10007;", "#b42318")}
            health = "<br>".join(f"<span style='color:{marks[lv][1]}'>{marks[lv][0]}</span> {msg}"
                                 for lv, msg in dataset_health(version))
            self.result.setText(f"Created <b>{version.dataset_id}</b> - train {counts['train']}, "
                                f"validation {counts['val']}, test {counts['test']}. Continue with step 2.<br>{health}")
            self.ctx.datasets_changed.emit()

        run_background(lambda: create_dataset_version(report, cfg, datasets_dir, name), done,
                       self._failed("Create dataset version"))

