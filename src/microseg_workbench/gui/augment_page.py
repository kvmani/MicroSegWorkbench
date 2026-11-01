"""Step 2: preprocessing and a visual augmentation pipeline builder.

The pipeline is drawn in the exact order training applies it (from ``core.augment_pipeline.PIPELINE``). Every block
shows what it does to the user's own image; galleries and the step-by-step strip come from the real ``augment``
function (via its trace), so what is shown is what training will do.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from ..core.augment import PHOTOMETRIC_OPS, AugmentConfig, AugmentConfigError, augment
from ..core.augment_pipeline import (
    INTENSITY_GROUP_FIELD, feature_widths, PIPELINE, PRESET_HELP, PRESETS, STEPS, apply_preset, expected_rate, list_recipes,
    load_recipe, matching_preset, preview_step, sanity_check, save_recipe,
)
from ..core.dataset import DatasetVersion, list_dataset_versions
from ..core.imageio import load_image, load_mask_values, to_unit_float
from ..core.preprocess import NORMALIZATIONS, Preprocess, PreprocessError, resize_image, resize_labels
from ..core.render import overlay, thumbnail, to_display_rgb
from .context import AppContext
from .preprocess_widgets import BorderEditor
from .pipeline_widgets import (
    ACCENT, GEOMETRY_COLOR, INTENSITY_COLOR, Arrow, ClickableImage, Dots, SliderSpin, StepCard, group_frame,
    make_spin, rect_tile_caption,
)
from .widgets import ImageView, hint, run_background

HELP = ("<b>Step 2 - Preprocessing and augmentation.</b> <b>A.</b> Leave out image borders and choose the image "
        "size. <b>B.</b> During training every patch passes through the augmentation pipeline "
        "<b>from left to right, top row first</b>. Pick a starting point (<i>Gentle</i> or <i>Standard</i> "
        "suit most data), then switch blocks on or off and drag the sliders: each block shows its weakest and "
        "strongest effect on your own image, and the patches at the bottom show what the model will see. Blue blocks "
        "move image and mask together; orange blocks change the image only. Run <b>Check these settings</b> before "
        "training.")

PERCENT = {"hflip", "vflip", "rot90", "rotate_prob", "photometric_prob"}
DEFAULT_ON = {"hflip": 0.5, "vflip": 0.5, "rot90": 0.5, "rotate_prob": 0.3, "contrast": 0.15, "brightness": 0.08,
              "gamma": 0.2, "noise_std": 0.02, "blur_max_sigma": 1.0}
GALLERY = 12


def _fmt(field: str):
    if field in PERCENT:
        return lambda v: f"{v * 100:.0f} %"
    return {"rotate_max_deg": lambda v: f"up to {v:.0f} deg", "patch_size": lambda v: f"{v:.0f} px",
            "contrast": lambda v: f"+/- {v * 100:.0f} %", "brightness": lambda v: f"+/- {v:.2f}",
            "gamma": lambda v: f"+/- {v:.2f}", "noise_std": lambda v: f"sd {v:.3f}",
            "blur_max_sigma": lambda v: f"up to {v:.1f} px"}.get(field, lambda v: f"{v:.3g}")


class AugmentPage(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.seed = 0
        self._sample_cache: tuple[Any, ...] | None = None
        self.selected_step = "rotate"
        self.tile_index = 0
        self.last_check: dict | None = None
        self._checked_settings: tuple | None = None
        self.checking = False
        self._remembered: dict[str, float] = dict(DEFAULT_ON)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(250)
        self.refresh_timer.timeout.connect(self.preview)

        # ---- top bar: data, presets, recipes ----------------------------------------------------------------
        self.dataset = QComboBox()
        self.dataset.currentIndexChanged.connect(self._load_samples)
        self.sample = QComboBox()
        self.sample.currentIndexChanged.connect(self._sample_changed)
        self.preset_buttons: dict[str, QPushButton] = {}
        self.preset_group = QButtonGroup(self)
        presets = QHBoxLayout()
        presets.addWidget(QLabel("Start from:"))
        for name in PRESETS:
            b = QPushButton(name.title())
            b.setCheckable(True)
            b.setToolTip(PRESET_HELP[name])
            b.clicked.connect(lambda _c=False, n=name: self.use_preset(n))
            self.preset_group.addButton(b)
            self.preset_buttons[name] = b
            presets.addWidget(b)
        self.recipes = QComboBox()
        self.recipes.setToolTip("Your saved settings in this workspace")
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(lambda: self.load_recipe(self.recipes.currentText()))
        save_btn = QPushButton("Save as...")
        save_btn.clicked.connect(self._save_dialog)
        top = QHBoxLayout()
        top.addWidget(QLabel("Dataset version:"))
        top.addWidget(self.dataset, 2)
        top.addWidget(QLabel("Example image:"))
        top.addWidget(self.sample, 1)
        top.addSpacing(12)
        top.addLayout(presets)
        top.addSpacing(12)
        top.addWidget(QLabel("My settings:"))
        top.addWidget(self.recipes, 1)
        top.addWidget(load_btn)
        top.addWidget(save_btn)

        # ---- exact-value inputs (source of truth) ---------------------------------------------------------------
        d = AugmentConfig()
        self.inputs: dict[str, QSpinBox | QDoubleSpinBox] = {}
        for f, (lo, hi, step) in {"hflip": (0, 1, 0.05), "vflip": (0, 1, 0.05), "rot90": (0, 1, 0.05),
                                  "rotate_prob": (0, 1, 0.05), "rotate_max_deg": (0, 90, 1),
                                  "photometric_prob": (0, 1, 0.05), "patch_size": (16, 1024, 8),
                                  "contrast": (0, 0.9, 0.01), "brightness": (0, 0.5, 0.01), "gamma": (0, 1, 0.05),
                                  "noise_std": (0, 0.3, 0.005), "blur_max_sigma": (0, 5, 0.1)}.items():
            spin = make_spin(lo, hi, step, getattr(d, f), integer=f == "patch_size")
            spin.valueChanged.connect(self._changed)
            self.inputs[f] = spin
        self.sliders = {f: SliderSpin(spin, _fmt(f)) for f, spin in self.inputs.items()}

        # ---- pipeline view ----------------------------------------------------------------------------------
        self.cards: dict[str, StepCard] = {}
        geo_frame, geo_lay = group_frame("1. Geometry - image and mask move together (blue)", GEOMETRY_COLOR)
        geo_row = QHBoxLayout()
        int_frame, int_lay = group_frame("2. Intensity - image only, the mask never changes (orange)", INTENSITY_COLOR)
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Apply intensity changes to"))
        group_row.addWidget(self.sliders["photometric_prob"], 1)
        self.group_dots = Dots()
        group_row.addWidget(self.group_dots)
        group_row.addWidget(QLabel("of patches (then every block switched on below, in order)"))
        int_lay.addLayout(group_row)
        int_row = QHBoxLayout()
        for step in PIPELINE:
            prob = self.sliders[step.prob_field] if step.prob_field else None
            strength = self.sliders[step.strength_field] if step.strength_field else None
            card = StepCard(step.id, step.title, step.kind, step.help, prob, strength, ("", ""))
            card.enable.toggled.connect(lambda on, s=step.id: self._toggle(s, on))
            card.selected.connect(self.select_step)
            self.cards[step.id] = card
            row = int_row if step.kind == "intensity" else geo_row
            if row.count():
                row.addWidget(Arrow())
            row.addWidget(card)
        self.extra: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for name, spec in sorted(PHOTOMETRIC_OPS.items()):
            prob_spin = make_spin(0, 1, 0.05, 0.5)
            prob_spin.valueChanged.connect(self._changed)
            card = StepCard(f"extra:{name}", name, "plugin", f"{spec['description']} (plugin {spec['module']})",
                            SliderSpin(prob_spin, _fmt("hflip")), None, ("original", "applied"))
            card.enable.setChecked(False)
            card.enable.toggled.connect(self._changed)
            card.selected.connect(self.select_step)
            self.cards[f"extra:{name}"] = card
            self.extra[name] = (card.enable, prob_spin)
            int_row.addWidget(Arrow())
            int_row.addWidget(card)
        out_label = QLabel("Training\npatch")
        out_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        out_label.setStyleSheet(f"QLabel {{ color: {ACCENT}; font-weight: 600; }}")
        int_row.addWidget(Arrow())
        int_row.addWidget(out_label)
        geo_row.addStretch(1)
        int_row.addStretch(1)
        geo_lay.addLayout(geo_row)
        int_lay.addLayout(int_row)
        pipe = QWidget()
        pv = QVBoxLayout(pipe)
        pv.addWidget(geo_frame)
        pv.addWidget(int_frame)
        pv.addStretch(1)
        pipe_scroll = QScrollArea()
        pipe_scroll.setWidgetResizable(True)
        pipe_scroll.setWidget(pipe)

        # ---- bottom tabs ------------------------------------------------------------------------------------
        self.tiles: list[ClickableImage] = []
        gallery = QWidget()
        gl = QVBoxLayout(gallery)
        gbar = QHBoxLayout()
        self.show_mask = QCheckBox("Show mask")
        self.show_mask.setChecked(True)
        self.show_mask.toggled.connect(self._redraw_tiles)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 90)
        self.opacity.setValue(45)
        self.opacity.setMaximumWidth(160)
        self.opacity.valueChanged.connect(self._redraw_tiles)
        new_btn = QPushButton("New random examples")
        new_btn.clicked.connect(self._new_examples)
        gbar.addWidget(QLabel("What the model will see - 12 random training patches. Click one to see how it was made."))
        gbar.addStretch(1)
        gbar.addWidget(self.show_mask)
        gbar.addWidget(QLabel("opacity"))
        gbar.addWidget(self.opacity)
        gbar.addWidget(new_btn)
        grid = QHBoxLayout()
        for i in range(GALLERY):
            t = ClickableImage(104)
            t.clicked.connect(lambda i=i: self.show_steps(i))
            self.tiles.append(t)
            grid.addWidget(t)
        grid.addStretch(1)
        self.captions = QLabel("")
        self.captions.setWordWrap(True)
        gl.addLayout(gbar)
        gl.addLayout(grid)
        gl.addWidget(self.captions)
        gl.addStretch(1)
        self._patches: list[tuple[np.ndarray, np.ndarray, list]] = []

        steps_tab = QWidget()
        self.steps_layout = QHBoxLayout()
        sw = QWidget()
        sw.setLayout(self.steps_layout)
        steps_scroll = QScrollArea()
        steps_scroll.setWidgetResizable(True)
        steps_scroll.setWidget(sw)
        self.steps_note = QLabel("Click a training patch to follow it through the pipeline.")
        sl = QVBoxLayout(steps_tab)
        sl.addWidget(self.steps_note)
        sl.addWidget(steps_scroll, 1)

        detail = QWidget()
        dl = QVBoxLayout(detail)
        self.detail_title = QLabel("")
        self.detail_title.setWordWrap(True)
        self.detail_view = ImageView("Select a block")
        dl.addWidget(self.detail_title)
        dl.addWidget(self.detail_view, 1)

        check = QWidget()
        cl = QVBoxLayout(check)
        crow = QHBoxLayout()
        self.check_btn = QPushButton("Check these settings")
        self.check_btn.setToolTip("Runs the real pipeline on 200 patches and looks for problems.")
        self.check_btn.clicked.connect(self.run_check)
        crow.addWidget(self.check_btn)
        crow.addStretch(1)
        self.findings = QLabel("Not checked yet.")
        self.findings.setWordWrap(True)
        self.findings.setTextFormat(Qt.TextFormat.RichText)
        self.rates = QTableWidget(0, 3)
        self.rates.setHorizontalHeaderLabels(["Block", "Set to", "Measured on 200 patches"])
        self.rates.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        cl.addLayout(crow)
        cl.addWidget(self.findings)
        cl.addWidget(self.rates, 1)

        prep = QWidget()
        prep_form = QFormLayout(prep)
        self.prep_form = prep_form
        self.resize_factor = QDoubleSpinBox()
        self.resize_factor.setRange(0.1, 4.0)
        self.resize_factor.setSingleStep(0.05)
        self.resize_factor.setValue(1.0)
        self.resize_factor.setToolTip("1.0 keeps the original size. Masks are resized with nearest neighbour; very "
                                      "thin features can vanish when shrinking - check the patches.")
        self.normalization = QComboBox()
        labels = {"dtype_range": "Scale by bit depth (default)", "per_image_standardize": "Per-image standardization"}
        for key in NORMALIZATIONS:
            self.normalization.addItem(labels[key], key)
        self.border: list[QSpinBox] = []
        border_row = QHBoxLayout()
        for label in ("top", "bottom", "left", "right"):
            b = QSpinBox()
            b.setRange(0, 10000)
            b.setToolTip(f"Pixels at the {label} edge that are never trained on or scored (e.g. an SEM data bar).")
            b.valueChanged.connect(self._prep_changed)
            border_row.addWidget(QLabel(label))
            border_row.addWidget(b)
            self.border.append(b)
        self.resize_factor.valueChanged.connect(self._prep_changed)
        self.normalization.currentIndexChanged.connect(self._prep_changed)
        prep_form.addRow("Intensity normalization:", self.normalization)

        self.tabs = QTabWidget()
        self.tabs.addTab(gallery, "Training patches")
        self.tabs.addTab(steps_tab, "Step by step")
        self.tabs.addTab(detail, "Selected block")
        self.tabs.addTab(check, "Check settings")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(pipe_scroll)
        splitter.addWidget(self.tabs)
        splitter.setSizes([560, 250])
        aug_tab = QWidget()
        al = QVBoxLayout(aug_tab)
        al.setContentsMargins(0, 4, 0, 0)
        al.addLayout(top)
        al.addWidget(splitter, 1)

        # ---- A. preprocessing tab: drag the border, compare sizes -----------------------------------------
        self.border_editor = BorderEditor()
        self.border_editor.changed.connect(self._border_dragged)
        no_border = QPushButton("No border")
        no_border.clicked.connect(lambda: [b.setValue(0) for b in self.border])
        bl = QVBoxLayout()
        bt = QLabel("<b>Exclude border</b> - drag the dashed edges to leave out a strip such as an SEM data bar. "
                    "Excluded pixels are never trained on or scored.")
        bt.setWordWrap(True)
        bl.addWidget(bt)
        bl.addWidget(self.border_editor, 1)
        brow = QHBoxLayout()
        brow.addLayout(border_row)
        brow.addWidget(no_border)
        brow.addStretch(1)
        bl.addLayout(brow)
        self.resize_slider = SliderSpin(self.resize_factor, lambda v: f"x {v:.2f}")
        self.size_before = ImageView("original")
        self.size_after = ImageView("as the model sees it")
        self.size_info = QLabel("")
        self.size_info.setWordWrap(True)
        self.size_info.setTextFormat(Qt.TextFormat.RichText)
        rl = QVBoxLayout()
        rt = QLabel("<b>Resize</b> - shrink very large micrographs to save memory, or enlarge tiny features. Results "
                    "are always scored at the original size.")
        rt.setWordWrap(True)
        rl.addWidget(rt)
        rl.addWidget(self.resize_slider)
        views = QHBoxLayout()
        for title, view in (("Original pixels", self.size_before), ("After resizing (same area)", self.size_after)):
            col = QVBoxLayout()
            col.addWidget(QLabel(title))
            col.addWidget(view, 1)
            views.addLayout(col)
        rl.addLayout(views, 1)
        rl.addWidget(self.size_info)
        rl.addWidget(prep)
        prep_tab = QWidget()
        pl = QHBoxLayout(prep_tab)
        pl.addLayout(bl, 3)
        pl.addLayout(rl, 2)
        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(prep_tab, "A. Preprocessing")
        self.main_tabs.addTab(aug_tab, "B. Augmentation pipeline")
        self.main_tabs.setCurrentIndex(1)
        layout = QVBoxLayout(self)
        layout.addWidget(hint(HELP))
        layout.addWidget(self.main_tabs, 1)
        ctx.datasets_changed.connect(self.refresh)
        self._sync_cards()
        self.set_advanced(ctx.advanced)
        ctx.mode_changed.connect(self.set_advanced)
        self.refresh()

    def set_advanced(self, on: bool) -> None:
        """Advanced mode shows exact-value boxes next to every slider and the normalization choice."""
        self.advanced = on
        for s in self.sliders.values():
            s.set_exact_visible(on)
        for name, (_check, prob) in self.extra.items():
            prob.setVisible(on)
        self.prep_form.setRowVisible(self.normalization, on)

    def check_is_stale(self) -> bool:
        """True when the settings changed after the last check."""
        try:
            return self._checked_settings != (self.config(), self.preprocess())
        except (AugmentConfigError, PreprocessError):
            return True

    # ---- data -------------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        current = self.dataset.currentData()
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for v in list_dataset_versions(self.ctx.workspace.datasets_dir):
            self.dataset.addItem(v.dataset_id, v.dataset_id)
        idx = self.dataset.findData(current)
        self.dataset.setCurrentIndex(idx if idx >= 0 else self.dataset.count() - 1)
        self.dataset.blockSignals(False)
        self.recipes.clear()
        self.recipes.addItems(list_recipes(self.ctx.workspace.root))
        self._load_samples()

    def _version(self) -> DatasetVersion | None:
        dataset_id = self.dataset.currentData()
        return DatasetVersion.open(self.ctx.workspace.datasets_dir / dataset_id) if dataset_id else None

    def _load_samples(self) -> None:
        self.sample.blockSignals(True)
        self.sample.clear()
        version = self._version()
        if version:
            for s in version.samples("train"):
                self.sample.addItem(s["stem"], s["stem"])
        self.sample.blockSignals(False)
        self._sample_changed()

    def _sample_changed(self) -> None:
        self._sample_cache = None
        self.schedule()

    def _prep_changed(self) -> None:
        self._sample_cache = None
        self.schedule()

    def _model_scale_sample(self) -> tuple[np.ndarray, np.ndarray, Any, str]:
        """(image, channel labels, version, size note) of the chosen example after preprocessing (cached)."""
        version = self._version()
        stem = self.sample.currentData()
        prep = self.preprocess()
        key = (version.dataset_id if version else None, stem, prep)
        if self._sample_cache and self._sample_cache[0] == key:
            return self._sample_cache[1]
        sample = next(s for s in version.samples() if s["stem"] == stem)
        image = to_unit_float(load_image(version.image_path(sample)).array)
        labels = version.class_map.channel_lut()[load_mask_values(version.mask_path(sample))]
        image, labels = prep.crop(image), prep.crop(labels)
        original = labels.shape
        shape = prep.scaled_shape(*original)
        image, labels = resize_image(image, shape), resize_labels(labels, shape)
        note = (f"Resized from {original[0]}x{original[1]} to {shape[0]}x{shape[1]} px. " if shape != original else "")
        if any(prep.exclude_border):
            note = f"Border excluded; analysed region {original[0]}x{original[1]} px. " + note
        if prep.normalization != "dtype_range":
            note += "Shown before standardization (standardized values are not displayable). "
        value = (image, labels, version, note)
        self._sample_cache = (key, value)
        return value

    # ---- settings -------------------------------------------------------------------------------------------
    def preprocess(self) -> Preprocess:
        return Preprocess(resize_factor=round(self.resize_factor.value(), 4),
                          normalization=self.normalization.currentData(),
                          exclude_border=tuple(b.value() for b in self.border))

    def config(self) -> AugmentConfig:
        """Current settings; raises AugmentConfigError with a readable message if invalid."""
        extra = tuple({"name": n, "probability": round(prob.value(), 4)}
                      for n, (check, prob) in self.extra.items() if check.isChecked())
        return AugmentConfig(**{name: w.value() for name, w in self.inputs.items()}, extra_ops=extra)

    def set_config(self, cfg: AugmentConfig) -> None:
        for name, spin in self.inputs.items():
            spin.blockSignals(True)
            spin.setValue(getattr(cfg, name))
            spin.blockSignals(False)
        chosen = {o["name"]: o for o in cfg.extra_ops}
        for name, (check, prob) in self.extra.items():
            check.blockSignals(True)
            check.setChecked(name in chosen)
            check.blockSignals(False)
            if name in chosen:
                prob.setValue(float(chosen[name].get("probability", 0.5)))
        for s in self.sliders.values():
            s._from_spin()
        self._changed()

    def reset(self) -> None:
        self.use_preset("standard")

    def use_preset(self, name: str) -> None:
        self.set_config(apply_preset(self.config(), name))

    def load_recipe(self, name: str) -> None:
        if not name:
            self.ctx.error(self, "Load settings", "There are no saved settings yet.")
            return
        self.set_config(load_recipe(self.ctx.workspace.root, name))
        self.ctx.status_message.emit(f"Loaded augmentation settings '{name}'.")

    def save_recipe(self, name: str) -> None:
        try:
            path = save_recipe(self.ctx.workspace.root, name, self.config())
        except (ValueError, AugmentConfigError) as exc:
            self.ctx.error(self, "Save settings", str(exc))
            return
        self.recipes.clear()
        self.recipes.addItems(list_recipes(self.ctx.workspace.root))
        self.recipes.setCurrentText(path.stem)
        self.ctx.status_message.emit(f"Saved augmentation settings as '{path.stem}'.")

    def _save_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Save augmentation settings", "Name:")
        if ok:
            self.save_recipe(name)

    def _toggle(self, step_id: str, on: bool) -> None:
        step = STEPS[step_id]
        field = step.prob_field or step.strength_field
        spin = self.inputs[field]
        if on and spin.value() == 0:
            spin.setValue(self._remembered.get(field, DEFAULT_ON.get(field, 0.5)))
            if step.kind == "intensity" and self.inputs[INTENSITY_GROUP_FIELD].value() == 0:
                self.inputs[INTENSITY_GROUP_FIELD].setValue(0.8)
            if step_id == "rotate" and self.inputs["rotate_max_deg"].value() == 0:
                self.inputs["rotate_max_deg"].setValue(15)
        elif not on and spin.value() > 0:
            self._remembered[field] = spin.value()
            spin.setValue(0)

    def _changed(self, *_args) -> None:
        self._sync_cards()
        self.schedule()
        self.ctx.workflow_changed.emit()

    def _sync_cards(self) -> None:
        try:
            cfg = self.config()
        except AugmentConfigError:
            return
        for step in PIPELINE:
            card = self.cards[step.id]
            rate = expected_rate(cfg, step)
            card.set_rate(rate)
            if step.kind != "crop":
                card.enable.blockSignals(True)
                card.enable.setChecked(rate > 0)
                card.enable.blockSignals(False)
        for name, (check, prob) in self.extra.items():
            self.cards[f"extra:{name}"].set_rate(prob.value() if check.isChecked() else 0.0)
        self.group_dots.set_fraction(cfg.photometric_prob)
        current = matching_preset(cfg)
        for name, b in self.preset_buttons.items():
            b.setChecked(name == current)
        if current is None:
            self.preset_group.setExclusive(False)
            for b in self.preset_buttons.values():
                b.setChecked(False)
            self.preset_group.setExclusive(True)

    def schedule(self) -> None:
        if self.dataset.count():
            self.refresh_timer.start()

    # ---- preprocessing previews ------------------------------------------------------------------------------
    def _border_dragged(self, t: int, b: int, left: int, right: int) -> None:
        for spin, v in zip(self.border, (t, b, left, right)):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        self._prep_changed()

    def _original_sample(self) -> tuple[np.ndarray, np.ndarray, Any]:
        version = self._version()
        stem = self.sample.currentData()
        key = (version.dataset_id if version else None, stem)
        if getattr(self, "_orig_cache", None) and self._orig_cache[0] == key:
            return self._orig_cache[1]
        sample = next(s for s in version.samples() if s["stem"] == stem)
        image = to_unit_float(load_image(version.image_path(sample)).array)
        labels = version.class_map.channel_lut()[load_mask_values(version.mask_path(sample))]
        self._orig_cache = (key, (image, labels, version))
        return self._orig_cache[1]

    def refresh_preprocess(self) -> None:
        """Update the border editor, the before/after resize views and the class retention figures."""
        if not self.dataset.count() or self.sample.currentData() is None:
            return
        image, labels, version = self._original_sample()
        cmap = version.class_map
        step = max(1, int(np.ceil(max(labels.shape) / 1600)))
        self.border_editor.set_image(overlay(image[::step, ::step], labels[::step, ::step], cmap, alpha=0.35),
                                     true_size=labels.shape)
        self.border_editor.set_border(*(b.value() for b in self.border))
        prep = self.preprocess()
        try:
            img, lab = prep.crop(image), prep.crop(labels)
        except PreprocessError as exc:
            self.size_info.setText(f"<span style='color:#b42318'>{exc}</span>")
            return
        shape = prep.scaled_shape(*lab.shape)
        f = prep.resize_factor
        r = int(min(160, *lab.shape))
        ry, rx = (lab.shape[0] - r) // 2, (lab.shape[1] - r) // 2
        reg_img, reg_lab = img[ry:ry + r, rx:rx + r], lab[ry:ry + r, rx:rx + r]
        small_shape = (max(1, round(r * f)), max(1, round(r * f)))
        small_img, small_lab = resize_image(reg_img, small_shape), resize_labels(reg_lab, small_shape)
        self.size_before.set_array(overlay(reg_img, reg_lab, cmap, alpha=0.35))
        self.size_after.set_array(overlay(small_img, small_lab, cmap, alpha=0.35))
        widths = feature_widths(lab, cmap.names)
        rows = [f"{name} about {w:.1f} px" + (f" -> {w * f:.1f} px" if f != 1 else "") for name, w in widths.items()]
        text = (f"Image {lab.shape[1]} x {lab.shape[0]} px -> model sees {shape[1]} x {shape[0]} px."
                + (" Typical feature width: " + ", ".join(rows) + "." if rows else ""))
        too_thin = [name for name, w in widths.items() if w * f < 1.5]
        if f < 1 and too_thin:
            text += (f"<br><span style='color:#9a5b00'>&#9888; {', '.join(too_thin)} would be thinner than about "
                     "1.5 pixels and break into fragments - use a larger resize factor.</span>")
        self.size_info.setText(text)

    # ---- previews -------------------------------------------------------------------------------------------
    def preview(self) -> None:
        version = self._version()
        if version is None or self.sample.currentData() is None:
            self.ctx.error(self, "Preview", "Create a dataset version in step 1 first.")
            return
        try:
            cfg = self.config()
            self.preprocess()
            image, labels, version, note = self._model_scale_sample()
        except (AugmentConfigError, PreprocessError) as exc:
            self.ctx.error(self, "Augmentation settings", str(exc))
            return
        self.class_map = version.class_map
        self._patches = []
        for i in range(GALLERY):
            trace: list = []
            img, lab, _ = augment(image, labels, cfg, np.random.default_rng([self.seed, i]), trace=trace)
            self._patches.append((img, lab, trace))
        self._redraw_tiles()
        self._update_thumbnails(image, labels, cfg)
        self.captions.setText(note + f"Each patch is {cfg.patch_size}x{cfg.patch_size} px. Grey hatching = pixels "
                              "from outside the image, ignored in training.")
        self.show_steps(self.tile_index)
        self.select_step(self.selected_step)
        self.refresh_preprocess()

    def _render(self, img: np.ndarray, lab: np.ndarray, with_mask: bool | None = None) -> np.ndarray:
        show = self.show_mask.isChecked() if with_mask is None else with_mask
        return overlay(img, lab, self.class_map, alpha=self.opacity.value() / 100) if show else to_display_rgb(img)

    def _redraw_tiles(self) -> None:
        for t, (img, lab, _trace) in zip(self.tiles, self._patches):
            t.set_array(self._render(img, lab))

    def _new_examples(self) -> None:
        self.seed += 1
        self.preview()

    def _region(self, image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = labels.shape
        s = min(160, h, w)
        y, x = (h - s) // 2, (w - s) // 2
        return image[y:y + s, x:x + s], labels[y:y + s, x:x + s]

    def _update_thumbnails(self, image: np.ndarray, labels: np.ndarray, cfg: AugmentConfig) -> None:
        img, lab = self._region(image, labels)
        for step_id, card in self.cards.items():
            a, b, ca, cb = self._step_pair(step_id, img, lab, cfg, image, labels)
            card.thumb_a.set_array(a)
            card.thumb_b.set_array(b)
            card.cap_a.setText(ca)
            card.cap_b.setText(cb)

    def _step_pair(self, step_id: str, img: np.ndarray, lab: np.ndarray, cfg: AugmentConfig,
                   full_img: np.ndarray, full_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
        geo = step_id in ("rotate", "hflip", "vflip", "rot90", "crop")
        if step_id == "crop":
            p = min(cfg.patch_size, *full_lab.shape)
            view = self._render(full_img, full_lab, True).copy()
            view[:2, :p] = view[p - 2:p, :p] = (42, 120, 214)
            view[:p, :2] = view[:p, p - 2:p] = (42, 120, 214)
            return thumbnail(view, 256), self._render(*preview_crop(full_img, full_lab, p), True), \
                f"{full_lab.shape[1]}x{full_lab.shape[0]} image", f"{cfg.patch_size} px patch"
        if step_id.startswith("extra:"):
            a, _ = preview_step(img, lab, step_id, 0, -1)
            b, _ = preview_step(img, lab, step_id, 0, +1)
            return to_display_rgb(a), to_display_rgb(b), "original", "applied"
        step = STEPS[step_id]
        value = getattr(cfg, step.strength_field) if step.strength_field else 0.0
        if step_id == "rotate":
            value = cfg.rotate_max_deg
        a, la = preview_step(img, lab, step_id, value, -1, seed=1)
        b, lb = preview_step(img, lab, step_id, value, +1, seed=1)
        caps = {"rotate": (f"-{value:.0f} deg", f"+{value:.0f} deg"), "hflip": ("before", "mirrored"),
                "vflip": ("before", "mirrored"), "rot90": ("90 deg", "270 deg"),
                "contrast": ("lowest", "highest"), "brightness": ("darkest", "lightest"),
                "gamma": ("lowest", "highest"), "noise_std": ("none", "strongest"),
                "blur_max_sigma": ("sharp", "strongest")}[step_id]
        if step_id in ("hflip", "vflip"):
            a, la = img, lab
        return self._render(a, la, geo), self._render(b, lb, geo), caps[0], caps[1]

    def select_step(self, step_id: str) -> None:
        self.selected_step = step_id
        for sid, card in self.cards.items():
            card.set_selected(sid == step_id)
        if not self._patches or self._sample_cache is None:
            return
        image, labels, _v, _n = self._sample_cache[1]
        cfg = self.config()
        img, lab = self._region(image, labels)
        a, b, ca, cb = self._step_pair(step_id, img, lab, cfg, image, labels)
        from ..core.render import tile_grid

        geo = step_id in ("rotate", "hflip", "vflip", "rot90", "crop")
        before = self._render(img, lab, geo)
        tiles = [before, a, b] if step_id != "crop" else [a, b]
        h = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
        self.detail_view.set_array(tile_grid(tiles, columns=len(tiles)))
        if step_id.startswith("extra:"):
            name = step_id[6:]
            title, text = name, PHOTOMETRIC_OPS[name]["description"]
        else:
            title, text = STEPS[step_id].title, STEPS[step_id].help
        order = "original | " + f"{ca} | {cb}" if step_id != "crop" else f"{ca} | {cb}"
        self.detail_title.setText(f"<b>{title}</b> - {text}<br><span style='color:#6b6b6b'>Shown: {order}</span>")

    def show_steps(self, index: int) -> None:
        self.tile_index = index
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()  # deleteLater is deferred; hide now so old tiles never overlap new ones
                w.deleteLater()
        if not self._patches:
            return
        trace = self._patches[index][2]
        for k, (step, text, img, lab) in enumerate(trace):
            if k:
                self.steps_layout.addWidget(Arrow())
            col = QVBoxLayout()
            tile = ClickableImage(120)
            tile.set_array(self._render(thumbnail_pair(img), thumbnail_labels(lab, img), True))
            col.addWidget(tile)
            col.addWidget(rect_tile_caption("input image" if step == "input" else text))
            w = QWidget()
            w.setLayout(col)
            self.steps_layout.addWidget(w)
        self.steps_layout.addStretch(1)
        self.steps_note.setText(f"Patch {index + 1}: every step that was applied, in order. Steps not shown were "
                                "skipped for this patch (by chance, according to their probability).")

    # ---- sanity check ---------------------------------------------------------------------------------------
    def run_check(self) -> None:
        version = self._version()
        if version is None:
            self.ctx.error(self, "Check settings", "Create a dataset version in step 1 first.")
            return
        try:
            cfg, prep = self.config(), self.preprocess()
        except (AugmentConfigError, PreprocessError) as exc:
            self.ctx.error(self, "Check settings", str(exc))
            return
        self.checking = True
        self.check_btn.setEnabled(False)
        self.findings.setText("Checking...")

        def work() -> dict:
            from ..training.data import SplitSource

            source = SplitSource(version, "train", prep, max_cached=0)
            samples = [(s.image, s.labels) for s in (source.get(i) for i in range(min(8, len(source))))]
            return sanity_check(samples, cfg, version.class_map.names, n=200)

        checked = (cfg, prep)

        def done(report: dict) -> None:
            self.checking = False
            self._checked_settings = checked
            self.check_btn.setEnabled(True)
            self.last_check = report
            icons = {"ok": ("&#10003;", "#1a7f37"), "warn": ("&#9888;", "#8a5a00"), "info": ("&#8505;", "#1f5fae")}
            self.findings.setText("<br>".join(f"<span style='color:{icons[level][1]}'>{icons[level][0]}</span> {msg}"
                                              for level, msg in report["findings"]))
            self.rates.setRowCount(len(report["rates"]))
            for r, row in enumerate(report["rates"]):
                for c, v in enumerate([row["title"], f"{100 * row['expected']:.0f} %", f"{100 * row['observed']:.0f} %"]):
                    self.rates.setItem(r, c, QTableWidgetItem(v))
            self.rates.resizeColumnsToContents()
            self.ctx.workflow_changed.emit()

        def failed(message: str, details: str) -> None:
            self.checking = False
            self.check_btn.setEnabled(True)
            self.ctx.error(self, "Check settings", message, details)

        run_background(work, done, failed)


def preview_crop(image: np.ndarray, labels: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    return image[:p, :p], labels[:p, :p]


def thumbnail_pair(img: np.ndarray, max_side: int = 240) -> np.ndarray:
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    return img[::step, ::step]


def thumbnail_labels(lab: np.ndarray, img: np.ndarray, max_side: int = 240) -> np.ndarray:
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    return lab[::step, ::step]
