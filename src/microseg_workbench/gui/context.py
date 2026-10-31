"""Shared application state passed to every page."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox, QWidget

from ..core.workspace import Workspace


class AppContext(QObject):
    datasets_changed = Signal()
    runs_changed = Signal()
    status_message = Signal(str)

    def __init__(self, workspace: Workspace, interactive: bool = True) -> None:
        super().__init__()
        self.workspace = workspace.ensure()
        self.interactive = interactive  # False in automated tests: no modal dialogs
        self.last_error: str | None = None

    def set_workspace(self, workspace: Workspace) -> None:
        self.workspace = workspace.ensure()
        self.datasets_changed.emit()
        self.runs_changed.emit()

    def error(self, parent: QWidget, title: str, message: str, details: str = "") -> None:
        self.last_error = message
        self.status_message.emit(f"{title}: {message}")
        if self.interactive:
            box = QMessageBox(QMessageBox.Icon.Warning, title, message, parent=parent)
            if details:
                box.setDetailedText(details)
            box.exec()

    def confirm(self, parent: QWidget, title: str, question: str) -> bool:
        if not self.interactive:
            return True
        return QMessageBox.question(parent, title, question) == QMessageBox.StandardButton.Yes
