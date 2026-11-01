"""Visual preprocessing: drag the excluded border on the image."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .widgets import array_to_pixmap

EDGES = ("top", "bottom", "left", "right")


class BorderEditor(QWidget):
    """Shows an image; the excluded border is shaded and its four edges can be dragged with the mouse.

    Values are in original image pixels. ``changed`` is emitted with (top, bottom, left, right).
    """

    changed = Signal(int, int, int, int)
    GRAB = 8  # screen pixels around an edge that start a drag
    MIN_KEEP = 8  # image pixels that must remain

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.pixmap = None
        self.img_h = self.img_w = 0
        self.border = [0, 0, 0, 0]
        self.dragging: str | None = None

    def set_image(self, rgb: np.ndarray | None, true_size: tuple[int, int] | None = None) -> None:
        """``rgb`` may be a reduced copy of a large image; ``true_size`` (h, w) gives the original pixel size."""
        if rgb is None:
            self.pixmap, self.img_h, self.img_w = None, 0, 0
        else:
            self.pixmap = array_to_pixmap(rgb)
            self.img_h, self.img_w = true_size or rgb.shape[:2]
        self.update()

    def set_border(self, top: int, bottom: int, left: int, right: int) -> None:
        self.border = [int(top), int(bottom), int(left), int(right)]
        self.update()

    # ---- geometry -----------------------------------------------------------------------------------------
    def _frame(self) -> tuple[QRectF, float]:
        if not self.img_w:
            return QRectF(), 1.0
        scale = min((self.width() - 20) / self.img_w, (self.height() - 34) / self.img_h)
        w, h = self.img_w * scale, self.img_h * scale
        return QRectF((self.width() - w) / 2, 6 + (self.height() - 34 - h) / 2, w, h), scale

    def _edge_positions(self) -> dict[str, float]:
        r, s = self._frame()
        t, b, lft, rgt = self.border
        return {"top": r.top() + t * s, "bottom": r.bottom() - b * s, "left": r.left() + lft * s,
                "right": r.right() - rgt * s}

    def _edge_at(self, pos: QPointF) -> str | None:
        if not self.img_w:
            return None
        r, _ = self._frame()
        e = self._edge_positions()
        inside_x = r.left() - self.GRAB <= pos.x() <= r.right() + self.GRAB
        inside_y = r.top() - self.GRAB <= pos.y() <= r.bottom() + self.GRAB
        best = None
        for name in EDGES:
            d = abs(pos.y() - e[name]) if name in ("top", "bottom") else abs(pos.x() - e[name])
            ok = inside_x if name in ("top", "bottom") else inside_y
            if ok and d <= self.GRAB and (best is None or d < best[0]):
                best = (d, name)
        return best[1] if best else None

    def drag_edge_to(self, name: str, image_coord: float) -> None:
        """Move one edge to an image coordinate (also used by tests)."""
        if not self.img_w:
            return  # nothing to drag until an image is shown
        t, b, lft, rgt = self.border
        if name == "top":
            t = int(round(min(max(0, image_coord), self.img_h - b - self.MIN_KEEP)))
        elif name == "bottom":
            b = int(round(min(max(0, self.img_h - image_coord), self.img_h - t - self.MIN_KEEP)))
        elif name == "left":
            lft = int(round(min(max(0, image_coord), self.img_w - rgt - self.MIN_KEEP)))
        else:
            rgt = int(round(min(max(0, self.img_w - image_coord), self.img_w - lft - self.MIN_KEEP)))
        if [t, b, lft, rgt] != self.border:
            self.border = [t, b, lft, rgt]
            self.update()
            self.changed.emit(t, b, lft, rgt)

    # ---- events -------------------------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.dragging = self._edge_at(event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self.dragging:
            r, s = self._frame()
            coord = (pos.y() - r.top()) / s if self.dragging in ("top", "bottom") else (pos.x() - r.left()) / s
            self.drag_edge_to(self.dragging, coord)
            return
        edge = self._edge_at(pos)
        if edge in ("top", "bottom"):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif edge:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.dragging = None

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#f4f4f2"))
        if self.pixmap is None:
            p.setPen(QColor("#6b6b6b"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")
            p.end()
            return
        r, s = self._frame()
        p.drawPixmap(r, self.pixmap, QRectF(self.pixmap.rect()))
        e = self._edge_positions()
        shade = QBrush(QColor(180, 35, 24, 110), Qt.BrushStyle.BDiagPattern)
        tint = QColor(255, 255, 255, 90)
        regions = [QRectF(r.left(), r.top(), r.width(), e["top"] - r.top()),
                   QRectF(r.left(), e["bottom"], r.width(), r.bottom() - e["bottom"]),
                   QRectF(r.left(), e["top"], e["left"] - r.left(), e["bottom"] - e["top"]),
                   QRectF(e["right"], e["top"], r.right() - e["right"], e["bottom"] - e["top"])]
        for reg in regions:
            if reg.width() > 0.5 and reg.height() > 0.5:
                p.fillRect(reg, tint)
                p.fillRect(reg, shade)
        p.setPen(QPen(QColor("#b42318"), 2, Qt.PenStyle.DashLine))
        p.drawRect(QRectF(QPointF(e["left"], e["top"]), QPointF(e["right"], e["bottom"])))
        p.setPen(QColor("#1f2d3d"))
        kept = (self.img_h - self.border[0] - self.border[1], self.img_w - self.border[2] - self.border[3])
        p.drawText(QRectF(r.left(), r.bottom() + 2, r.width(), 16), Qt.AlignmentFlag.AlignCenter,
                   f"analysed {kept[1]} x {kept[0]} px of {self.img_w} x {self.img_h}; drag the dashed edges")
        p.end()
