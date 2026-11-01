"""Main window and entry point: ``msw-gui`` or ``python -m microseg_workbench gui``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from ..core.workspace import Workspace
from .augment_page import AugmentPage
from .compare_page import ComparePage
from .context import AppContext
from .data_page import DataPage
from .help_page import HelpPage
from .results_page import ResultsPage
from .share_page import SharePage
from .train_page import TrainPage
from .workflow import COLORS, GUIDE, ICONS, step_statuses

STEPS = ["1  Data", "2  Preprocess", "3  Training", "4  Results", "5  Compare", "6  Share", "?  Help"]


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
        self.share_page = SharePage(self.ctx)
        self.help_page = HelpPage()

        self.nav = QListWidget()
        self.nav.addItems(STEPS)
        self.nav.setFixedWidth(190)
        self.nav.setStyleSheet("QListWidget { font-size: 14px; } QListWidget::item { padding: 10px 6px; }")
        self.stack = QStackedWidget()
        for page in (self.data_page, self.augment_page, self.train_page, self.results_page, self.compare_page,
                     self.share_page, self.help_page):
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self._page_changed)

        # footer: what to do here + Back / Next
        self.guide = QLabel()
        self.guide.setWordWrap(True)
        self.step_state = QLabel()
        self.step_state.setWordWrap(True)
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(lambda: self.go_to(max(0, self.nav.currentRow() - 1)))
        self.next_btn = QPushButton("Next")
        self.next_btn.setDefault(True)
        self.next_btn.clicked.connect(lambda: self.go_to(min(len(STEPS) - 1, self.nav.currentRow() + 1)))
        footer = QFrame()
        footer.setObjectName("footer")
        footer.setStyleSheet("QFrame#footer { background: #f6f6f4; border-top: 1px solid #d8d8d4; }")
        fl = QHBoxLayout(footer)
        texts = QVBoxLayout()
        texts.addWidget(self.guide)
        texts.addWidget(self.step_state)
        fl.addLayout(texts, 1)
        fl.addWidget(self.back_btn)
        fl.addWidget(self.next_btn)

        self.workspace_label = QLabel()
        change = QPushButton("Change workspace...")
        change.clicked.connect(self._change_workspace)
        self.mode_group = QButtonGroup(self)
        self.beginner_btn = QPushButton("Beginner")
        self.advanced_btn = QPushButton("Advanced")
        for b, on in ((self.beginner_btn, False), (self.advanced_btn, True)):
            b.setCheckable(True)
            self.mode_group.addButton(b)
            b.clicked.connect(lambda _c=False, on=on: self.ctx.set_advanced(on))
        self.beginner_btn.setToolTip("Presets and visual controls only.")
        self.advanced_btn.setToolTip("Show every option, with exact values.")
        (self.advanced_btn if self.ctx.advanced else self.beginner_btn).setChecked(True)
        top = QHBoxLayout()
        top.addWidget(self.workspace_label, 1)
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.beginner_btn)
        top.addWidget(self.advanced_btn)
        top.addSpacing(16)
        top.addWidget(change)
        right = QVBoxLayout()
        right.addWidget(self.stack, 1)
        right.addWidget(footer)
        body = QHBoxLayout()
        body.addWidget(self.nav)
        body.addLayout(right, 1)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top)
        layout.addLayout(body, 1)

        self.status_timer = QTimer(self)  # coalesces bursts of changes into one status refresh
        self.status_timer.setSingleShot(True)
        self.status_timer.setInterval(200)
        self.status_timer.timeout.connect(self.refresh_status)
        for sig in (self.ctx.datasets_changed, self.ctx.runs_changed, self.ctx.workflow_changed):
            sig.connect(self.status_timer.start)
        self.train_page.dataset.currentIndexChanged.connect(lambda _i: self.status_timer.start())
        self.train_page.device_info_changed = self.status_timer.start
        self.nav.setCurrentRow(0)
        self.setCentralWidget(central)
        self.ctx.status_message.connect(lambda m: self.statusBar().showMessage(m, 15000))
        self._update_workspace_label()
        self.statusBar().showMessage("Start with step 1. New here? Open Help or press 'Create demo data'.")

    def go_to(self, step: int) -> None:
        self.nav.setCurrentRow(step)

    def _page_changed(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        self.refresh_status()

    def refresh_status(self) -> None:
        """Update step icons, the footer and the training checklist from the current state."""
        self.statuses = step_statuses(self)
        for i, st in enumerate(self.statuses):
            item = self.nav.item(i)
            item.setText(f"{ICONS[st.state]}  {STEPS[i]}")
            item.setForeground(QColor(COLORS[st.state]))
            item.setToolTip(st.reason)
        row = max(0, self.nav.currentRow())
        title, guide = GUIDE[row]
        self.guide.setText(f"<b>What to do here:</b> {guide}")
        st = self.statuses[row]
        self.step_state.setText(f"<span style='color:{COLORS[st.state]}'>{ICONS[st.state]} {st.reason}</span>")
        self.back_btn.setEnabled(row > 0)
        nxt = row + 1
        self.next_btn.setVisible(nxt < len(STEPS))
        if nxt < len(STEPS):
            self.next_btn.setText(f"Next: {GUIDE[nxt][0]} \u2192")
        self.train_page.update_readiness(self.readiness())

    def readiness(self) -> list[tuple[str, str]]:
        """Checklist shown above 'Start training'."""
        from ..core.dataset import DatasetVersion
        from ..core.preprocess import PreprocessError

        items: list[tuple[str, str]] = []
        dataset_id = self.train_page.dataset.currentData()
        if not dataset_id:
            return [("block", "No dataset version yet - create one in step 1.")]
        version = DatasetVersion.open(self.ctx.workspace.datasets_dir / dataset_id)
        c = version.manifest["split_counts"]
        items.append(("ok" if c["train"] else "block",
                      f"Dataset {dataset_id}: {c['train']} training / {c['val']} validation / {c['test']} test images"))
        try:
            prep = self.augment_page.preprocess()
            for s in version.samples():
                prep.analysed_region(s["height"], s["width"])
            items.append(("ok", "Preprocessing fits every image"))
        except PreprocessError as exc:
            items.append(("block", f"Preprocessing: {exc} Fix it in step 2."))
        aug = self.augment_page
        if aug.last_check is None or aug.check_is_stale():
            items.append(("todo", "Augmentation not checked yet - recommended: Check these settings in step 2"))
        elif any(level == "warn" for level, _ in aug.last_check["findings"]):
            n = sum(1 for level, _ in aug.last_check["findings"] if level == "warn")
            items.append(("warn", f"Augmentation check found {n} thing(s) to review in step 2"))
        else:
            items.append(("ok", "Augmentation checked - no problems"))
        device = self.train_page.device_info.text()
        items.append(("ok" if device.startswith("Found") else "todo",
                      device.split(". ")[1] + "." if device.startswith("Found") and ". " in device else device))
        return items

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
            self.share_page.load_settings()
            self._update_workspace_label()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.train_page.is_running() and not self.ctx.confirm(
                self, "Training is running", "Stop training and quit? The last completed epoch is kept."):
            event.ignore()
            return
        self.train_page.shutdown()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--cli"]:
        # A packaged (frozen) executable re-launches itself with --cli to run training in a child process.
        from ..cli import main as cli_main

        return cli_main(argv[1:])
    from ..plugins import load_plugins

    load_plugins()
    parser = argparse.ArgumentParser(prog="msw-gui")
    parser.add_argument("--workspace")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(Workspace(Path(args.workspace)) if args.workspace else Workspace.default())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
