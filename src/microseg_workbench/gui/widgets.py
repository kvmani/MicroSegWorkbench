"""Reusable Qt widgets: image view, line chart, background tasks, help banners."""

from __future__ import annotations

import math
import traceback
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QToolTip, QWidget

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e0"


def array_to_pixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w = arr.shape[:2]
        img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class ImageView(QLabel):
    """Shows an image scaled to fit, keeping pixels crisp (nearest neighbour)."""

    def __init__(self, placeholder: str = "No image") -> None:
        super().__init__(placeholder)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QLabel { background: #f4f4f2; border: 1px solid #d0d0cc; color: #6b6b6b; }")

    def set_array(self, arr: np.ndarray | None) -> None:
        self.set_pixmap(array_to_pixmap(arr) if arr is not None else None)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        if pixmap is None:
            self.clear()
            self.setText("No image")
        self._rescale()

    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None and not self._pixmap.isNull():
            self.setPixmap(self._pixmap.scaled(self.size() * 0.98, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.FastTransformation))


class LineChart(QWidget):
    """Minimal line chart: one y-axis, legend, markers, nearest-point hover tooltip."""

    def __init__(self, title: str, y_label: str, y_range: tuple[float, float] | None = None,
                 x_label: str = "epoch") -> None:
        super().__init__()
        self.title = title
        self.y_label = y_label
        self.x_label = x_label
        self.fixed_range = y_range
        self.series: dict[str, list[tuple[float, float]]] = {}
        self.setMinimumSize(320, 200)
        self.setMouseTracking(True)

    def clear(self) -> None:
        self.series = {}
        self.update()

    def add_point(self, name: str, x: float, y: float | None) -> None:
        """Add a point; a point with the same x replaces the old one (e.g. an epoch repeated after resume)."""
        if y is None or not math.isfinite(y):
            return
        pts = [pt for pt in self.series.setdefault(name, []) if pt[0] != float(x)]
        pts.append((float(x), float(y)))
        pts.sort()
        self.series[name] = pts
        self.update()

    def set_series(self, name: str, points: list[tuple[float, float | None]]) -> None:
        self.series[name] = sorted((float(x), float(y)) for x, y in points if y is not None and math.isfinite(y))
        self.update()

    def point_count(self) -> int:
        return sum(len(v) for v in self.series.values())

    def _bounds(self) -> tuple[float, float, float, float]:
        xs = [x for pts in self.series.values() for x, _ in pts] or [0.0, 1.0]
        ys = [y for pts in self.series.values() for _, y in pts] or [0.0, 1.0]
        x0, x1 = min(xs), max(max(xs), min(xs) + 1)
        if self.fixed_range:
            y0, y1 = self.fixed_range
        else:
            y0, y1 = 0.0, max(ys) * 1.1 or 1.0
        return x0, x1, y0, y1

    def _plot_rect(self) -> QRectF:
        return QRectF(52, 48, max(10, self.width() - 64), max(10, self.height() - 80))

    def _to_px(self, x: float, y: float) -> QPointF:
        r = self._plot_rect()
        x0, x1, y0, y1 = self._bounds()
        return QPointF(r.left() + (x - x0) / (x1 - x0) * r.width(), r.bottom() - (y - y0) / (y1 - y0) * r.height())

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        r = self._plot_rect()
        x0, x1, y0, y1 = self._bounds()
        font = QFont(self.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1))
        p.setFont(font)
        p.setPen(QColor(INK))
        p.drawText(QRectF(8, 4, self.width() - 16, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.title)
        for i in range(5):
            yv = y0 + (y1 - y0) * i / 4
            py = r.bottom() - r.height() * i / 4
            p.setPen(QPen(QColor(GRID), 1))
            p.drawLine(QPointF(r.left(), py), QPointF(r.right(), py))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(0, py - 8, r.left() - 6, 16), Qt.AlignmentFlag.AlignRight, f"{yv:.2f}")
        p.drawText(QRectF(r.left(), r.bottom() + 4, r.width(), 16), Qt.AlignmentFlag.AlignCenter, self.x_label)
        p.drawText(QRectF(r.left(), r.bottom() + 4, 40, 16), Qt.AlignmentFlag.AlignLeft, f"{x0:g}")
        p.drawText(QRectF(r.right() - 40, r.bottom() + 4, 40, 16), Qt.AlignmentFlag.AlignRight, f"{x1:g}")
        if not self.series:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Waiting for data")
        legend_x = r.right()
        for idx, (name, pts) in reversed(list(enumerate(self.series.items()))):
            if idx >= len(SERIES_COLORS):
                continue  # never cycle colours; callers limit the series count
            color = QColor(SERIES_COLORS[idx])
            path = QPainterPath()
            for j, (x, y) in enumerate(pts):
                pt = self._to_px(x, y)
                path.moveTo(pt) if j == 0 else path.lineTo(pt)
            p.setPen(QPen(color, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setBrush(color)
            p.setPen(QPen(QColor("#ffffff"), 2))
            if len(pts) <= 60:  # markers only while they stay readable
                for x, y in pts:
                    p.drawEllipse(self._to_px(x, y), 4, 4)
            if len(self.series) > 1:  # legend row under the title; a single series is named by the title
                width = p.fontMetrics().horizontalAdvance(name) + 22
                legend_x -= width + 8
                p.fillRect(QRectF(legend_x, 32, 12, 4), color)
                p.setPen(QColor(INK))
                p.drawText(QRectF(legend_x + 16, 24, width, 20), Qt.AlignmentFlag.AlignVCenter, name)
        p.end()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        best = None
        pos = event.position()
        for name, pts in self.series.items():
            for x, y in pts:
                d = (self._to_px(x, y) - pos).manhattanLength()
                if best is None or d < best[0]:
                    best = (d, name, x, y)
        if best and best[0] < 40:
            QToolTip.showText(event.globalPosition().toPoint(), f"{best[1]}\n{self.x_label} {best[2]:g}: {best[3]:.4f}", self)
        else:
            QToolTip.hideText()


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str, str)


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:  # reported to the UI, never swallowed
            self.signals.failed.emit(str(exc), traceback.format_exc())
        else:
            self.signals.done.emit(result)


_active: set[_Task] = set()


def run_background(fn: Callable[[], Any], on_done: Callable[[Any], None],
                   on_error: Callable[[str, str], None]) -> None:
    """Run ``fn`` on a worker thread; callbacks run on the UI thread."""
    task = _Task(fn)
    _active.add(task)
    task.setAutoDelete(False)

    def finish_ok(result: Any) -> None:
        _active.discard(task)
        on_done(result)

    def finish_err(message: str, details: str) -> None:
        _active.discard(task)
        on_error(message, details)

    task.signals.done.connect(finish_ok, Qt.ConnectionType.QueuedConnection)
    task.signals.failed.connect(finish_err, Qt.ConnectionType.QueuedConnection)
    QThreadPool.globalInstance().start(task)


def hint(text: str) -> QLabel:
    """A short explanatory banner shown at the top of each page."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setStyleSheet("QLabel { background: #eef4fb; border: 1px solid #c9dcf2; padding: 8px; color: #1f2d3d; }")
    return label
