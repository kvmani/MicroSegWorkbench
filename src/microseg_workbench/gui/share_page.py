"""Step 6: collaborate through the shared office folder (publish, download, review)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ..collab.client import ServiceClient, ServiceError, update_service
from ..collab.settings import CollabSettings
from ..collab.share import DECISIONS, SharedStore, ShareError
from ..core.dataset import list_dataset_versions
from ..training.runs import list_runs
from .context import AppContext
from .widgets import hint, run_background

HELP = ("<b>Step 6 - Share.</b> The shared office folder is the single trusted place for datasets and models. "
        "<i>Publish</i> copies a dataset version or a finished run there (files are checked and never changed "
        "afterwards); <i>Download</i> brings a colleague's dataset or run to this computer for training, comparison "
        "or segmentation. A model becomes a deployment candidate only when <b>someone other than its publisher "
        "approves it</b>. The optional metadata service is updated automatically when configured.")
REVIEW_COLORS = {"approved": "#1a7f37", "rejected": "#b42318", "needs_changes": "#8a5a00", "unreviewed": "#6b6b6b"}


class _Progress(QObject):
    message = Signal(str)


class SharePage(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.store: SharedStore | None = None
        self.busy = False
        self.last_result: Any = None
        self._progress = _Progress()
        self._progress.message.connect(lambda m: self.ctx.status_message.emit(m))

        self.path = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        self.user = QLineEdit()
        self.service = QLineEdit()
        self.service.setPlaceholderText("optional, e.g. http://ml-vm:8765")
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        save = QPushButton("Save and connect")
        save.clicked.connect(self.save_settings)
        init = QPushButton("Initialise new shared folder")
        init.setToolTip("Only once, by an administrator, on an empty folder.")
        init.clicked.connect(self._init_share)
        self.conn = QLabel("Not connected.")
        self.conn.setWordWrap(True)
        settings_box = QGroupBox("Connection")
        form = QFormLayout(settings_box)
        prow = QHBoxLayout()
        prow.addWidget(self.path, 1)
        prow.addWidget(browse)
        form.addRow("Shared folder:", prow)
        form.addRow("Your name:", self.user)
        form.addRow("Metadata service:", self.service)
        form.addRow("Service token:", self.token)
        brow = QHBoxLayout()
        brow.addWidget(save)
        brow.addWidget(init)
        form.addRow(brow)
        form.addRow(self.conn)

        # datasets
        self.datasets = QTableWidget(0, 5)
        self.datasets.setHorizontalHeaderLabels(["Dataset version", "Images (train/val/test)", "Classes",
                                                 "Published by", "On this computer"])
        self._table_setup(self.datasets)
        self.local_dataset = QComboBox()
        pub_d = QPushButton("Publish")
        pub_d.clicked.connect(self.publish_dataset)
        get_d = QPushButton("Download selected")
        get_d.clicked.connect(self.download_dataset)
        dbox = QGroupBox("Shared datasets")
        dl = QVBoxLayout(dbox)
        dl.addWidget(self.datasets, 1)
        r = QHBoxLayout()
        r.addWidget(QLabel("Local version:"))
        r.addWidget(self.local_dataset, 1)
        r.addWidget(pub_d)
        r.addWidget(get_d)
        dl.addLayout(r)

        # runs
        self.runs = QTableWidget(0, 6)
        self.runs.setHorizontalHeaderLabels(["Run", "Review", "Test mIoU", "Dataset version", "Published by",
                                             "On this computer"])
        self._table_setup(self.runs)
        self.runs.itemSelectionChanged.connect(self._show_reviews)
        self.approved_only = QCheckBox("Approved models only")
        self.approved_only.toggled.connect(self.refresh_tables)
        self.local_run = QComboBox()
        pub_r = QPushButton("Publish")
        pub_r.clicked.connect(self.publish_run)
        get_r = QPushButton("Download selected")
        get_r.clicked.connect(self.download_run)
        rbox = QGroupBox("Shared runs (models)")
        rl = QVBoxLayout(rbox)
        rl.addWidget(self.approved_only)
        rl.addWidget(self.runs, 1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Local completed run:"))
        r2.addWidget(self.local_run, 1)
        r2.addWidget(pub_r)
        r2.addWidget(get_r)
        rl.addLayout(r2)

        # review
        self.history = QLabel("Select a shared run to see its reviews.")
        self.history.setWordWrap(True)
        self.history.setTextFormat(Qt.TextFormat.RichText)
        self.history.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.decision = QComboBox()
        for d in DECISIONS:
            self.decision.addItem(d.replace("_", " "), d)
        self.comment = QTextEdit()
        self.comment.setPlaceholderText("What did you check? Required when rejecting or requesting changes.")
        self.comment.setMaximumHeight(70)
        submit = QPushButton("Record review")
        submit.clicked.connect(self.submit_review)
        sync = QPushButton("Sync with service")
        sync.clicked.connect(self.sync_service)
        vbox = QGroupBox("Review selected run")
        vl = QVBoxLayout(vbox)
        vl.addWidget(self.history, 1)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Decision:"))
        r3.addWidget(self.decision)
        r3.addStretch(1)
        r3.addWidget(sync)
        vl.addLayout(r3)
        vl.addWidget(self.comment)
        vl.addWidget(submit)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(settings_box)
        lv.addWidget(vbox, 1)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(dbox, 1)
        rv.addWidget(rbox, 1)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 700])
        layout = QVBoxLayout(self)
        layout.addWidget(hint(HELP))
        layout.addWidget(splitter, 1)
        ctx.datasets_changed.connect(self.refresh_local)
        ctx.runs_changed.connect(self.refresh_local)
        self.load_settings()

    @staticmethod
    def _table_setup(table: QTableWidget) -> None:
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)

    # ----- settings ----------------------------------------------------------------------------------------
    def settings(self) -> CollabSettings:
        return CollabSettings(share_path=self.path.text().strip(), user=self.user.text().strip(),
                              service_url=self.service.text().strip(), service_token=self.token.text().strip())

    def load_settings(self) -> None:
        s = CollabSettings.load(self.ctx.workspace)
        self.path.setText(s.share_path)
        self.user.setText(s.user or s.user_name())
        self.service.setText(s.service_url)
        self.token.setText(s.service_token)
        self.refresh_local()
        if s.share_path:
            self.connect_share()

    def save_settings(self) -> None:
        self.settings().save(self.ctx.workspace)
        self.connect_share()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Shared office folder", self.path.text())
        if folder:
            self.path.setText(folder)

    def _init_share(self) -> None:
        if not self.path.text().strip():
            self.ctx.error(self, "Shared folder", "Enter the folder to initialise first.")
            return
        if self.ctx.confirm(self, "Initialise shared folder", f"Prepare {self.path.text()} as a new shared folder?"):
            SharedStore.create(Path(self.path.text().strip()))
            self.save_settings()

    def connect_share(self) -> None:
        settings = self.settings()
        try:
            self.store = SharedStore.open(Path(settings.share_path))
        except (ShareError, OSError) as exc:
            self.store = None
            self.conn.setText(f"<span style='color:#b42318'>Not connected: {exc}</span>")
            self.refresh_tables()
            return
        note = f"Connected to {self.store.root} as <b>{settings.user_name()}</b>."
        if settings.service_url:
            try:
                who = ServiceClient(settings.service_url, settings.service_token, timeout=5, retries=1).whoami()
                note += f" Metadata service: signed in as <b>{who}</b>."
            except ServiceError as exc:
                note += f" <span style='color:#8a5a00'>Metadata service unavailable ({exc}).</span>"
        self.conn.setText(note)
        self.refresh_tables()

    # ----- tables ------------------------------------------------------------------------------------------
    def refresh_local(self) -> None:
        ws = self.ctx.workspace
        self.local_dataset.clear()
        for v in list_dataset_versions(ws.datasets_dir):
            self.local_dataset.addItem(v.dataset_id, v.dataset_id)
        self.local_run.clear()
        for r in list_runs(ws.runs_dir):
            if r.state() == "completed" and not r.info().get("imported_from_share"):
                self.local_run.addItem(r.run_id, r.run_id)
        self.refresh_tables()

    def refresh_tables(self) -> None:
        keep_run, keep_dataset = self.selected_run_id(), self._selected(self.datasets)
        self.datasets.setRowCount(0)
        self.runs.setRowCount(0)
        if self.store is None:
            return
        ws = self.ctx.workspace
        datasets = self.store.list_datasets()
        self.datasets.setRowCount(len(datasets))
        for row, d in enumerate(datasets):
            c = d["split_counts"]
            local = "yes" if (ws.datasets_dir / d["dataset_id"]).exists() else "no"
            for col, v in enumerate([d["dataset_id"], f"{c['train']}/{c['val']}/{c['test']}",
                                     ", ".join(d["class_names"]), f"{d['published_by']} {d['published_at'][:10]}",
                                     local]):
                self.datasets.setItem(row, col, QTableWidgetItem(v))
        runs = self.store.list_runs()
        if self.approved_only.isChecked():
            runs = [r for r in runs if r["review"] == "approved"]
        self.runs.setRowCount(len(runs))
        for row, r in enumerate(runs):
            miou = r["summary"].get("mean_iou")
            local = "yes" if (ws.runs_dir / r["run_id"]).exists() else "no"
            values = [r["run_id"], r["review"].replace("_", " ").upper(), "n/a" if miou is None else f"{miou:.4f}",
                      r["dataset_id"], f"{r['published_by']} {r['published_at'][:10]}", local]
            for col, v in enumerate(values):
                item = QTableWidgetItem(v)
                if col == 1:
                    item.setForeground(QColor(REVIEW_COLORS.get(r["review"], "#2b2b2b")))
                self.runs.setItem(row, col, item)
        self.datasets.resizeColumnsToContents()
        self.runs.resizeColumnsToContents()
        for table, keep in ((self.runs, keep_run), (self.datasets, keep_dataset)):
            for row in range(table.rowCount()):
                if keep and table.item(row, 0).text() == keep:
                    table.selectRow(row)

    def _selected(self, table: QTableWidget) -> str | None:
        rows = table.selectionModel().selectedRows() if table.selectionModel() else []
        item = table.item(rows[0].row(), 0) if rows else None
        return item.text() if item else None

    def selected_run_id(self) -> str | None:
        return self._selected(self.runs)

    def _show_reviews(self, run_id: str | None = None) -> None:
        run_id = run_id or self.selected_run_id()
        if not run_id or not self.store:
            return
        reviews = self.store.reviews(run_id)
        if not reviews:
            self.history.setText(f"<b>{run_id}</b>: no reviews yet (unreviewed).")
            return
        lines = [f"<b>{run_id}</b>:"]
        for r in reviews:
            color = REVIEW_COLORS.get(r["decision"], "#2b2b2b")
            lines.append(f"<span style='color:{color}'><b>{r['decision'].replace('_', ' ')}</b></span> by "
                         f"{r['reviewer']} on {r['reviewed_at'][:16].replace('T', ' ')} - {r['comment'] or ''}")
        self.history.setText("<br>".join(lines))

    # ----- actions -----------------------------------------------------------------------------------------
    def _run(self, label: str, fn: Callable[[], Any], after: Callable[[Any], None] | None = None) -> None:
        if self.store is None:
            self.ctx.error(self, label, "Connect to the shared folder first.")
            return
        if self.busy:
            self.ctx.error(self, label, "Another transfer is still running.")
            return
        self.busy = True
        self.ctx.status_message.emit(f"{label}...")

        def done(result: Any) -> None:
            self.busy = False
            self.last_result = result
            note = update_service(self.settings(), self.store) if self.settings().service_url else ""
            self.ctx.status_message.emit(f"{label}: done. {note}")
            if after:
                after(result)
            self.refresh_tables()

        def failed(message: str, details: str) -> None:
            self.busy = False
            self.ctx.error(self, label, message, details)

        run_background(fn, done, failed)

    def publish_dataset(self) -> None:
        from ..core.dataset import DatasetVersion

        dataset_id = self.local_dataset.currentData()
        if not dataset_id:
            self.ctx.error(self, "Publish dataset", "There is no local dataset version.")
            return
        version = DatasetVersion.open(self.ctx.workspace.datasets_dir / dataset_id)
        store, user = self.store, self.settings().user_name()
        self._run("Publish dataset", lambda: store.publish_dataset(version, user=user,
                                                                  progress=self._progress.message.emit))

    def publish_run(self) -> None:
        from ..training.runs import RunDir

        run_id = self.local_run.currentData()
        if not run_id:
            self.ctx.error(self, "Publish run", "There is no completed local run to publish.")
            return
        run = RunDir(self.ctx.workspace.runs_dir / run_id)
        store, user = self.store, self.settings().user_name()
        self._run("Publish run", lambda: store.publish_run(run, user=user, progress=self._progress.message.emit))

    def download_dataset(self) -> None:
        dataset_id = self._selected(self.datasets)
        if not dataset_id:
            self.ctx.error(self, "Download", "Select a shared dataset first.")
            return
        store, ws = self.store, self.ctx.workspace
        self._run("Download dataset", lambda: store.pull_dataset(dataset_id, ws, self._progress.message.emit),
                  lambda _r: self.ctx.datasets_changed.emit())

    def download_run(self) -> None:
        run_id = self.selected_run_id()
        if not run_id:
            self.ctx.error(self, "Download", "Select a shared run first.")
            return
        store, ws = self.store, self.ctx.workspace

        def after(_r: Any) -> None:
            self.ctx.datasets_changed.emit()
            self.ctx.runs_changed.emit()

        self._run("Download run", lambda: store.pull_run(run_id, ws, progress=self._progress.message.emit), after)

    def submit_review(self) -> None:
        run_id = self.selected_run_id()
        if not run_id or self.store is None:
            self.ctx.error(self, "Review", "Select a shared run first.")
            return
        try:
            self.store.add_review(run_id, self.decision.currentData(), self.comment.toPlainText(),
                                  reviewer=self.settings().user_name())
        except ShareError as exc:
            self.ctx.error(self, "Review", str(exc))
            return
        note = update_service(self.settings(), self.store)
        self.comment.clear()
        self.ctx.status_message.emit(f"Review recorded for {run_id}. {note}")
        self.refresh_tables()
        self._show_reviews(run_id)

    def sync_service(self) -> None:
        if not self.settings().service_url or self.store is None:
            self.ctx.error(self, "Sync", "Configure the shared folder and the metadata service first.")
            return
        store, settings = self.store, self.settings()
        self._run("Sync with service", lambda: ServiceClient(settings.service_url, settings.service_token)
                  .sync_from_share(store))
