"""Visual widgets for the data step: split overview and class balance."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}  # validated categorical order
SPLIT_TITLES = {"train": "Training", "val": "Validation", "test": "Test"}
INK = "#1f2d3d"
MUTED = "#6b6b6b"


class SplitView(QWidget):
    """Three columns (training / validation / test); each specimen group is a tile in exactly one column."""

    def __init__(self) -> None:
        super().__init__()
        self.groups: dict[str, tuple[str, int]] = {}
        self.error = ""
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_plan(self, groups: dict[str, tuple[str, int]]) -> None:
        """``groups``: group name -> (split, number of images)."""
        self.groups, self.error = dict(groups), ""
        self.setToolTip("Each tile is one group (e.g. a specimen); a group never appears in two columns, so test "
                        "images are never near-copies of training images.")
        self.updateGeometry()
        self.update()

    def set_error(self, message: str) -> None:
        self.groups, self.error = {}, message
        self.update()

    def _layout(self, width: float) -> tuple[list[tuple[str, QRectF]], dict[str, list[tuple[str, QRectF]]], float]:
        col_w = (width - 16) / 3
        fm = QFontMetrics(self.font())
        heads, tiles, bottom = [], {}, 0.0
        for i, split in enumerate(("train", "val", "test")):
            x0 = i * (col_w + 8)
            heads.append((split, QRectF(x0, 0, col_w, 22)))
            x, y = x0 + 4, 28.0
            tiles[split] = []
            for g, (s, n) in sorted(self.groups.items()):
                if s != split:
                    continue
                w = fm.horizontalAdvance(f"{g}  {n}") + 16
                if x + w > x0 + col_w - 4 and x > x0 + 4:
                    x, y = x0 + 4, y + 26
                tiles[split].append((g, QRectF(x, y, w, 22)))
                x += w + 6
            bottom = max(bottom, y + 26)
        return heads, tiles, bottom

    def sizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        _h, _t, bottom = self._layout(max(300, self.width()))
        return QSize(300, int(max(120, bottom + 6)))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.error:
            p.setPen(QColor("#b42318"))
            p.drawText(self.rect().adjusted(4, 4, -4, -4), Qt.TextFlag.TextWordWrap, self.error)
            p.end()
            return
        heads, tiles, _ = self._layout(self.width())
        counts = {s: sum(n for g, (sp, n) in self.groups.items() if sp == s) for s in SPLIT_COLORS}
        for split, rect in heads:
            color = QColor(SPLIT_COLORS[split])
            p.fillRect(QRectF(rect.left(), rect.bottom() - 3, rect.width(), 3), color)
            p.setPen(QColor(INK))
            p.drawText(rect.adjusted(2, 0, 0, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{SPLIT_TITLES[split]}: {counts[split]} images")
            for g, r in tiles[split]:
                fill = QColor(color)
                fill.setAlpha(40)
                p.setBrush(fill)
                p.setPen(QPen(color, 1.5))
                p.drawRoundedRect(r, 5, 5)
                p.setPen(QColor(INK))
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, f"{g}  {self.groups[g][1]}")
            if not tiles[split]:
                p.setPen(QColor(MUTED))
                p.drawText(QRectF(rect.left() + 4, 28, rect.width(), 22), Qt.AlignmentFlag.AlignLeft, "none")
        p.end()


class ClassBars(QWidget):
    """Share of all pixels per class as horizontal bars, with the class colour as a swatch and text labels."""

    ROW = 22

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str, float, int, int]] = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_rows(self, rows: list[tuple[str, str, float, int, int]]) -> None:
        """rows: (name, colour, fraction of pixels, images containing it, total images)."""
        self.rows = rows
        self.setFixedHeight(self.ROW * len(rows) + 4 if rows else 0)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = QFontMetrics(self.font())
        name_w = max((fm.horizontalAdvance(r[0]) for r in self.rows), default=0) + 28
        text_w = max((fm.horizontalAdvance(f"{100 * r[2]:.2f}% of pixels, in {r[3]}/{r[4]} images")
                      for r in self.rows), default=0) + 12
        bar_w = max(40, self.width() - name_w - text_w - 8)
        for i, (name, color, frac, n_img, total) in enumerate(self.rows):
            y = i * self.ROW + 2
            p.setPen(QPen(QColor("#8a8a86"), 1))
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(2, y + 5, 12, 12), 2, 2)
            p.setPen(QColor(INK))
            p.drawText(QRectF(20, y, name_w - 20, self.ROW), Qt.AlignmentFlag.AlignVCenter, name)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#e8e8e4"))
            p.drawRoundedRect(QRectF(name_w, y + 6, bar_w, 10), 3, 3)
            p.setBrush(QColor("#2a78d6"))
            p.drawRoundedRect(QRectF(name_w, y + 6, max(2.0, bar_w * frac), 10), 3, 3)
            p.setPen(QColor(INK if frac >= 0.02 or i == 0 else "#9a5b00"))
            p.drawText(QRectF(name_w + bar_w + 8, y, text_w, self.ROW), Qt.AlignmentFlag.AlignVCenter,
                       f"{100 * frac:.2f}% of pixels, in {n_img}/{total} images")
        p.end()
