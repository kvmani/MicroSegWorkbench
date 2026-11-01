"""A clickable map of the whole workflow, coloured by each stage's status."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .workflow import COLORS, ICONS


class WorkflowMap(QWidget):
    clicked = Signal(int)  # node index

    def __init__(self) -> None:
        super().__init__()
        self.nodes: list[tuple[str, str, str]] = []  # (label, state, reason)
        self.current = -1
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._rects: list[QRectF] = []

    def set_nodes(self, nodes: list[tuple[str, str, str]], current: int) -> None:
        self.nodes, self.current = nodes, current
        self.update()

    def _layout(self) -> list[QRectF]:
        if not self.nodes:
            return []
        fm = QFontMetrics(self.font())
        widths = [fm.horizontalAdvance(f"{ICONS[s]}  {label}") + 20 for label, s, _ in self.nodes]
        gap = max(14.0, (self.width() - 4 - sum(widths)) / max(1, len(widths) - 1))
        gap = min(gap, 40.0)
        x = (self.width() - (sum(widths) + gap * (len(widths) - 1))) / 2
        rects = []
        for w in widths:
            rects.append(QRectF(x, 6, w, 28))
            x += w + gap
        return rects

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._rects = self._layout()
        for i, (r, (label, state, _reason)) in enumerate(zip(self._rects, self.nodes)):
            color = QColor(COLORS[state])
            if i:
                prev = self._rects[i - 1]
                y = r.center().y()
                p.setPen(QPen(QColor("#9a9a96"), 1.5))
                p.drawLine(QPointF(prev.right() + 3, y), QPointF(r.left() - 3, y))
                p.drawLine(QPointF(r.left() - 3, y), QPointF(r.left() - 8, y - 4))
                p.drawLine(QPointF(r.left() - 3, y), QPointF(r.left() - 8, y + 4))
            fill = QColor(color)
            fill.setAlpha(28 if state != "locked" else 10)
            p.setBrush(fill)
            p.setPen(QPen(color, 2.5 if i == self.current else 1.2))
            p.drawRoundedRect(r, 6, 6)
            font = QFont(self.font())
            font.setBold(i == self.current)
            p.setFont(font)
            p.setPen(color)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, f"{ICONS[state]}  {label}")
        p.end()

    def node_at(self, pos: QPointF) -> int:
        for i, r in enumerate(self._rects or self._layout()):
            if r.contains(pos):
                return i
        return -1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        i = self.node_at(event.position())
        self.setCursor(Qt.CursorShape.PointingHandCursor if i >= 0 else Qt.CursorShape.ArrowCursor)
        self.setToolTip(self.nodes[i][2] if i >= 0 else "")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        i = self.node_at(event.position())
        if i >= 0:
            self.clicked.emit(i)
