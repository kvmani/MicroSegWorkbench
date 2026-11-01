"""Widgets for the visual augmentation pipeline builder."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from .widgets import array_to_pixmap

ACCENT = "#2a78d6"
GEOMETRY_COLOR = "#1f5fae"
INTENSITY_COLOR = "#9a5b00"
MUTED = "#6b6b6b"


class Dots(QWidget):
    """'Fires in N of 10 patches' as ten dots (filled = fires)."""

    def __init__(self) -> None:
        super().__init__()
        self.fraction = 0.0
        self.setFixedHeight(14)
        self.setMinimumWidth(110)

    def set_fraction(self, fraction: float) -> None:
        self.fraction = max(0.0, min(1.0, fraction))
        self.setToolTip(f"Applied to about {round(self.fraction * 100)} % of training patches")
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        filled = round(self.fraction * 10)
        for i in range(10):
            c = QPointF(6 + i * 11, 7)
            if i < filled:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(ACCENT))
            else:
                p.setPen(QPen(QColor("#9a9a96"), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, 4, 4)
        p.end()


class Arrow(QWidget):
    """A right-pointing connector between pipeline blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(22)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        p.setPen(QPen(QColor("#8a8a86"), 2))
        p.drawLine(QPointF(2, y), QPointF(18, y))
        p.drawLine(QPointF(18, y), QPointF(12, y - 5))
        p.drawLine(QPointF(18, y), QPointF(12, y + 5))
        p.end()


class ClickableImage(QLabel):
    clicked = Signal()

    def __init__(self, size: int) -> None:
        super().__init__()
        self.box = size
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background: #f4f4f2; border: 1px solid #d0d0cc; }")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.has_image = False

    def set_array(self, arr: np.ndarray | None) -> None:
        if arr is None:
            self.clear()
            self.has_image = False
            return
        self.setPixmap(array_to_pixmap(arr).scaled(self.box - 2, self.box - 2, Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.FastTransformation))
        self.has_image = True

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)


class SliderSpin(QWidget):
    """A slider bound to an exact-value spin box (the spin box holds the value; the slider is the visual control)."""

    def __init__(self, spin: QSpinBox | QDoubleSpinBox, fmt: Callable[[float], str]) -> None:
        super().__init__()
        self.spin = spin
        self.fmt = fmt
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.steps = max(1, round((spin.maximum() - spin.minimum()) / spin.singleStep()))
        self.slider.setRange(0, self.steps)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(70)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label)
        row.addWidget(spin)
        self.slider.valueChanged.connect(self._from_slider)
        spin.valueChanged.connect(self._from_spin)
        self._from_spin()

    def set_exact_visible(self, visible: bool) -> None:
        """Advanced mode shows the exact-value box next to the slider."""
        self.spin.setVisible(visible)

    def _from_slider(self, pos: int) -> None:
        value = self.spin.minimum() + pos * self.spin.singleStep()
        if abs(value - self.spin.value()) > 1e-9:
            self.spin.setValue(value)

    def _from_spin(self, *_args) -> None:
        pos = round((self.spin.value() - self.spin.minimum()) / self.spin.singleStep())
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.value_label.setText(self.fmt(self.spin.value()))


def make_spin(lo: float, hi: float, step: float, value: float, integer: bool = False) -> QSpinBox | QDoubleSpinBox:
    spin: QSpinBox | QDoubleSpinBox = QSpinBox() if integer else QDoubleSpinBox()
    if isinstance(spin, QDoubleSpinBox):
        spin.setDecimals(3)
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setMaximumWidth(78)
    spin.setToolTip("Exact value")
    return spin


class StepCard(QFrame):
    """One pipeline block: on/off, kind badge, weakest/strongest thumbnails, probability and strength controls."""

    selected = Signal(str)
    THUMB = 60

    def __init__(self, step_id: str, title: str, kind: str, help_text: str,
                 prob: SliderSpin | None, strength: SliderSpin | None, thumb_labels: tuple[str, str]) -> None:
        super().__init__()
        self.step_id = step_id
        self.setObjectName("card")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(206)
        self.setToolTip(help_text)
        self.enable = QCheckBox(title)
        self.enable.setStyleSheet("QCheckBox { font-weight: 600; }")
        self.title = QLabel(title)
        self.title.setStyleSheet("QLabel { font-weight: 600; }")
        if kind == "crop":
            self.enable.setChecked(True)  # always on; shown as a plain title
            self.enable.hide()
        else:
            self.title.hide()
        badge_text = {"geometry": "moves image + mask", "intensity": "image only - mask unchanged",
                      "crop": "always - sets patch size", "plugin": "plugin - image only"}[kind]
        badge_color = GEOMETRY_COLOR if kind in ("geometry", "crop") else INTENSITY_COLOR
        self.badge = QLabel(badge_text)
        self.badge.setStyleSheet(f"QLabel {{ color: {badge_color}; font-size: 11px; }}")
        self.thumb_a = ClickableImage(self.THUMB)
        self.thumb_b = ClickableImage(self.THUMB)
        self.cap_a = QLabel(thumb_labels[0])
        self.cap_b = QLabel(thumb_labels[1])
        for cap in (self.cap_a, self.cap_b):
            cap.setStyleSheet(f"QLabel {{ color: {MUTED}; font-size: 10px; }}")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbs = QHBoxLayout()
        for t, c in ((self.thumb_a, self.cap_a), (self.thumb_b, self.cap_b)):
            col = QVBoxLayout()
            col.setSpacing(1)
            col.addWidget(t, 0, Qt.AlignmentFlag.AlignHCenter)
            col.addWidget(c)
            thumbs.addLayout(col)
            t.clicked.connect(lambda: self.selected.emit(self.step_id))
        self.dots = Dots()
        self.rate = QLabel("")
        self.rate.setStyleSheet(f"QLabel {{ color: {MUTED}; font-size: 11px; }}")
        self.prob = prob
        self.strength = strength
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(2)
        lay.addWidget(self.enable)
        lay.addWidget(self.title)
        lay.addWidget(self.badge)
        lay.addLayout(thumbs)
        small = "QLabel { color: #444; font-size: 11px; }"
        if prob is not None:
            lab = QLabel("How often")
            lab.setStyleSheet(small)
            lay.addWidget(lab)
            lay.addWidget(prob)
        if strength is not None:
            lab = QLabel("Patch size" if kind == "crop" else "How strong")
            lab.setStyleSheet(small)
            lay.addWidget(lab)
            lay.addWidget(strength)
        dots_row = QHBoxLayout()
        dots_row.addWidget(self.dots)
        dots_row.addWidget(self.rate, 1)
        lay.addLayout(dots_row)
        lay.addStretch(1)
        self.set_selected(False)

    def set_rate(self, fraction: float) -> None:
        self.dots.set_fraction(fraction)
        n = round(fraction * 10)
        self.rate.setText("never" if fraction <= 0 else "every patch" if fraction >= 1 else f"{n} in 10 patches")
        self.setProperty("off", fraction <= 0)
        self.set_selected(self.property("selected") is True)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", on)
        off = self.property("off") is True
        border = ACCENT if on else "#c9c9c4"
        bg = "#f6f6f4" if off else "#ffffff"
        self.setStyleSheet(f"QFrame#card {{ border: {2 if on else 1}px solid {border}; border-radius: 6px; "
                           f"background: {bg}; }}")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.selected.emit(self.step_id)
        super().mousePressEvent(event)


def group_frame(title: str, color: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("group")
    frame.setStyleSheet(f"QFrame#group {{ border: 1px dashed {color}; border-radius: 8px; }}")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(8, 6, 8, 8)
    head = QLabel(title)
    head.setStyleSheet(f"QLabel {{ color: {color}; font-weight: 600; }}")
    lay.addWidget(head)
    return frame, lay


def rect_tile_caption(text: str) -> QLabel:
    cap = QLabel(text)
    cap.setWordWrap(True)
    cap.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
    cap.setStyleSheet(f"QLabel {{ color: {MUTED}; font-size: 11px; }}")
    cap.setFixedWidth(120)
    return cap


__all__ = ["Arrow", "ClickableImage", "Dots", "SliderSpin", "StepCard", "group_frame", "make_spin",
           "rect_tile_caption", "QRectF"]
