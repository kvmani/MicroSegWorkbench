"""Shared application state passed to every page."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox, QWidget

from ..core.workspace import Workspace


class AppContext(QObject):
    datasets_changed = Signal()
    runs_changed = Signal()
    status_message = Signal(str)
    workflow_changed = Signal()  # something that affects step status (checks, settings) changed
    mode_changed = Signal(bool)  # True = Advanced mode

    def __init__(self, workspace: Workspace, interactive: bool = True) -> None:
        super().__init__()
        self.workspace = workspace.ensure()
        self.interactive = interactive  # False in automated tests: no modal dialogs
        self.last_error: str | None = None
        self.advanced = bool(self._ui_settings().get("advanced", False))  # Beginner mode by default

    def _ui_settings(self) -> dict:
        import json

        try:
            return json.loads((self.workspace.root / "ui.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def set_advanced(self, on: bool) -> None:
        """Switch Beginner/Advanced mode (remembered in this workspace)."""
        from ..core.atomic import write_json_atomic

        self.advanced = on
        data = self._ui_settings()
        data["advanced"] = on
        write_json_atomic(self.workspace.root / "ui.json", data)
        self.mode_changed.emit(on)

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
