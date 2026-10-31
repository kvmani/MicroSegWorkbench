"""Step 2: configure augmentation and preview it on real samples."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QSizePolicy, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..core.augment import AugmentConfig, AugmentConfigError
from ..core.dataset import DatasetVersion, list_dataset_versions
from ..core.imageio import load_image, load_mask_values, to_unit_float
from ..core.preprocess import NORMALIZATIONS, Preprocess, PreprocessError, resize_image, resize_labels
from ..core.render import augmentation_preview
from .context import AppContext
from .widgets import ImageView, hint

HELP = ("<b>Step 2 - Preprocessing and augmentation.</b> <i>Resize factor</i> scales images before the model sees "
        "them (0.5 = half width and height, for very large micrographs); results are always mapped back and scored "
        "at the original resolution. <i>Per-image standardization</i> removes brightness/contrast differences "
        "between acquisitions. During training each image is randomly flipped, rotated and its "
        "brightness/contrast varied so the model learns features rather than one specimen's look. The mask is "
        "always moved together with the image; brightness changes never touch the mask. Grey hatching marks "
        "pixels outside the original image, which are <i>ignored</i> during training. Press <b>Preview</b> to "
        "see the settings on a real sample (top: image, bottom: mask overlay). These settings are used in step 3.")

FIELDS = [
    # name, label, kind, min, max, step, tooltip
    ("patch_size", "Training patch size (px)", int, 16, 1024, 8, "Square crop fed to the model; multiple of 8."),
    ("hflip", "Left-right flip probability", float, 0, 1, 0.05, "Chance of a horizontal mirror."),
    ("vflip", "Up-down flip probability", float, 0, 1, 0.05, "Chance of a vertical mirror."),
    ("rot90", "90-degree rotation probability", float, 0, 1, 0.05, "Chance of rotating by 90, 180 or 270 degrees."),
    ("rotate_prob", "Free rotation probability", float, 0, 1, 0.05,
     "Chance of a small arbitrary rotation (corners become 'ignore')."),
    ("rotate_max_deg", "Free rotation max angle (deg)", float, 0, 180, 1, "Largest free rotation angle."),
    ("photometric_prob", "Brightness/contrast probability", float, 0, 1, 0.05,
     "Chance of applying the intensity changes below."),
    ("brightness", "Brightness change (max)", float, 0, 0.5, 0.01, "Largest added intensity, as a fraction of full scale."),
    ("contrast", "Contrast change (max)", float, 0, 0.9, 0.01, "Largest relative contrast change."),
    ("gamma", "Gamma change (max log)", float, 0, 1, 0.05, "Non-linear intensity change; 0 = off."),
    ("noise_std", "Noise level (std)", float, 0, 0.3, 0.005, "Added Gaussian noise; 0 = off."),
]


class AugmentPage(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.dataset = QComboBox()
        self.dataset.currentIndexChanged.connect(self._load_samples)
        self.sample = QComboBox()
        self.inputs: dict[str, QSpinBox | QDoubleSpinBox] = {}
        prep_box = QGroupBox("Preprocessing (training, evaluation and prediction)")
        prep_form = QFormLayout(prep_box)
        self.resize = QDoubleSpinBox()
        self.resize.setRange(0.1, 4.0)
        self.resize.setSingleStep(0.05)
        self.resize.setValue(1.0)
        self.resize.setToolTip("1.0 keeps the original size. Masks are resized with nearest neighbour; very thin "
                               "features can vanish when shrinking - check the preview.")
        self.normalization = QComboBox()
        labels = {"dtype_range": "Scale by bit depth (default)",
                  "per_image_standardize": "Per-image standardization"}
        for key in NORMALIZATIONS:
            self.normalization.addItem(labels[key], key)
        prep_form.addRow("Resize factor:", self.resize)
        prep_form.addRow("Intensity normalization:", self.normalization)
        box = QGroupBox("Augmentation settings")
        form = QFormLayout(box)
        defaults = AugmentConfig()
        for name, label, kind, lo, hi, step, tip in FIELDS:
            w = QSpinBox() if kind is int else QDoubleSpinBox()
            if isinstance(w, QDoubleSpinBox):
                w.setDecimals(3)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(getattr(defaults, name))
            w.setToolTip(tip)
            form.addRow(label + ":", w)
            self.inputs[name] = w
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setToolTip("Different preview seeds show different random variants.")
        form.addRow("Preview seed:", self.seed)
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self.reset)
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)
        buttons = QHBoxLayout()
        buttons.addWidget(reset)
        buttons.addWidget(self.preview_btn)
        form.addRow(buttons)

        self.view = ImageView("Press Preview")
        self.captions = QLabel("")
        self.captions.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(QLabel("Dataset version:"))
        top.addWidget(self.dataset, 2)
        top.addWidget(QLabel("Sample:"))
        top.addWidget(self.sample, 1)
        right = QVBoxLayout()
        right.addWidget(self.view, 1)
        right.addWidget(self.captions)
        left = QVBoxLayout()
        left.addWidget(prep_box)
        left.addWidget(box)
        left.addStretch(1)
        body = QHBoxLayout()
        body.addLayout(left)
        body.addLayout(right, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(hint(HELP))
        layout.addLayout(top)
        layout.addLayout(body, 1)
        ctx.datasets_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        current = self.dataset.currentData()
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for v in list_dataset_versions(self.ctx.workspace.datasets_dir):
            self.dataset.addItem(v.dataset_id, v.dataset_id)
        idx = self.dataset.findData(current)
        self.dataset.setCurrentIndex(idx if idx >= 0 else self.dataset.count() - 1)
        self.dataset.blockSignals(False)
        self._load_samples()

    def _version(self) -> DatasetVersion | None:
        dataset_id = self.dataset.currentData()
        return DatasetVersion.open(self.ctx.workspace.datasets_dir / dataset_id) if dataset_id else None

    def _load_samples(self) -> None:
        self.sample.clear()
        version = self._version()
        if version:
            for s in version.samples("train"):
                self.sample.addItem(s["stem"], s["stem"])

    def reset(self) -> None:
        defaults = AugmentConfig()
        for name, w in self.inputs.items():
            w.setValue(getattr(defaults, name))

    def preprocess(self) -> Preprocess:
        return Preprocess(resize_factor=round(self.resize.value(), 4), normalization=self.normalization.currentData())

    def config(self) -> AugmentConfig:
        """Current settings; raises AugmentConfigError with a readable message if invalid."""
        return AugmentConfig(**{name: w.value() for name, w in self.inputs.items()})

    def preview(self) -> None:
        version = self._version()
        if version is None or self.sample.currentData() is None:
            self.ctx.error(self, "Preview", "Create a dataset version in step 1 first.")
            return
        try:
            cfg = self.config()
            prep = self.preprocess()
        except (AugmentConfigError, PreprocessError) as exc:
            self.ctx.error(self, "Augmentation settings", str(exc))
            return
        sample = next(s for s in version.samples() if s["stem"] == self.sample.currentData())
        image = to_unit_float(load_image(version.image_path(sample)).array)
        labels = version.class_map.channel_lut()[load_mask_values(version.mask_path(sample))]
        original = labels.shape
        shape = prep.scaled_shape(*original)
        image, labels = resize_image(image, shape), resize_labels(labels, shape)
        grid, captions = augmentation_preview(image, labels, version.class_map, cfg, seed=self.seed.value())
        self.view.set_array(grid)
        size_note = (f"Resized from {original[0]}x{original[1]} to {shape[0]}x{shape[1]} px. "
                     if shape != original else "")
        norm_note = ("Shown before standardization (standardized values are not displayable). "
                     if prep.normalization != "dtype_range" else "")
        self.captions.setText(size_note + norm_note + "Columns: " + " | ".join(f"{i}: {', '.join(c)}" for i, c in enumerate(captions)) +
                              f".  Training additionally crops random {cfg.patch_size}x{cfg.patch_size} patches.")
