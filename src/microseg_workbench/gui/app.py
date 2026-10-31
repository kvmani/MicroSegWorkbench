"""Main window and entry point: ``msw-gui`` or ``python -m microseg_workbench gui``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import __version__
from ..core.workspace import Workspace
from .augment_page import AugmentPage
from .compare_page import ComparePage
from .context import AppContext
from .data_page import DataPage
from .help_page import HelpPage
from .results_page import ResultsPage
from .train_page import TrainPage

STEPS = ["1  Data", "2  Preprocess", "3  Training", "4  Results", "5  Compare", "?  Help"]


class MainWindow(QMainWindow):
    def __init__(self, workspace: Workspace, interactive: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(f"MicroSegWorkbench {__version__}")
        self.resize(1280, 820)
        self.ctx = AppContext(workspace, interactive=interactive)
        self.data_page = DataPage(self.ctx)
        self.augment_page = AugmentPage(self.ctx)
        self.train_page = TrainPage(self.ctx, self.augment_page.config, self.augment_page.preprocess)
        self.results_page = ResultsPage(self.ctx, resume=self._resume_run)
        self.compare_page = ComparePage(self.ctx)
        self.help_page = HelpPage()

        self.nav = QListWidget()
        self.nav.addItems(STEPS)
        self.nav.setFixedWidth(170)
        self.nav.setStyleSheet("QListWidget { font-size: 14px; } QListWidget::item { padding: 10px 6px; }")
        self.stack = QStackedWidget()
        for page in (self.data_page, self.augment_page, self.train_page, self.results_page, self.compare_page,
                     self.help_page):
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.workspace_label = QLabel()
        change = QPushButton("Change workspace...")
        change.clicked.connect(self._change_workspace)
        top = QHBoxLayout()
        top.addWidget(self.workspace_label, 1)
        top.addWidget(change)
        body = QHBoxLayout()
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top)
        layout.addLayout(body, 1)
        self.setCentralWidget(central)
        self.ctx.status_message.connect(lambda m: self.statusBar().showMessage(m, 15000))
        self._update_workspace_label()
        self.statusBar().showMessage("Start with step 1. New here? Open Help or press 'Create demo data'.")

    def go_to(self, step: int) -> None:
        self.nav.setCurrentRow(step)

    def _resume_run(self, run) -> bool:
        if self.train_page.resume(run):
            self.go_to(2)
            return True
        return False

    def _update_workspace_label(self) -> None:
        self.workspace_label.setText(f"<b>Workspace:</b> {self.ctx.workspace.root}")

    def _change_workspace(self) -> None:
        if self.train_page.is_running():
            self.ctx.error(self, "Workspace", "Wait for training to finish or cancel it first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose workspace folder", str(self.ctx.workspace.root))
        if folder:
            self.ctx.set_workspace(Workspace(Path(folder)))
            self._update_workspace_label()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.train_page.is_running() and not self.ctx.confirm(
                self, "Training is running", "Stop training and quit? The last completed epoch is kept."):
            event.ignore()
            return
        self.train_page.shutdown()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="msw-gui")
    parser.add_argument("--workspace")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(Workspace(Path(args.workspace)) if args.workspace else Workspace.default())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
