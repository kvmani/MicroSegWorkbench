"""Step 4: review runs, test-set metrics, per-class results and overlays."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QTextEdit, QLabel, QListWidget, QListWidgetItem, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..training.runs import RunDir, list_runs
from .context import AppContext
from .widgets import ImageView, hint, run_background

HELP = ("<b>Step 4 - Results.</b> Each training run keeps its exact settings, dataset version, software versions "
        "and checkpoints. Metrics here come from the <i>test</i> split, which the model never saw during training "
        "or model selection. IoU and Dice: 1 is perfect, 0 is no overlap; 'n/a' means the class occurs in neither "
        "truth nor prediction. Results are <b>unreviewed</b> until a scientist approves them.")
UNREVIEWED = ("<span style='color:#8a5a00'><b>&#9888; UNREVIEWED</b> - not approved for scientific use or "
              "deployment.</span>")


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


class ResultsPage(QWidget):
    def __init__(self, ctx: AppContext, resume: Callable[[RunDir], bool] | None = None) -> None:
        super().__init__()
        self.ctx = ctx
        self.resume_callback = resume
        self.last_prediction: dict | None = None
        self.predicting = False
        self.runs = QListWidget()
        self.runs.currentItemChanged.connect(self._show_run)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        self.open_btn = QPushButton("Open run folder")
        self.open_btn.clicked.connect(self._open_folder)
        self.resume_btn = QPushButton("Resume training")
        self.resume_btn.setToolTip("Continue a cancelled, failed or interrupted run from its last completed epoch.")
        self.resume_btn.clicked.connect(self._resume)
        self.report_btn = QPushButton("Export report")
        self.report_btn.setToolTip("Write a self-contained HTML report (settings, scores, curves, overlays, provenance) "
                                   "that can be emailed or printed.")
        self.report_btn.clicked.connect(self.export_report)
        self.last_report: Path | None = None
        self.predict_btn = QPushButton("Segment new images...")
        self.predict_btn.setToolTip("Apply this run's best model to a folder of new micrographs.")
        self.predict_btn.clicked.connect(self._predict_dialog)

        self.banner = QLabel("")
        self.diagnosis = QLabel("")
        self.diagnosis.setWordWrap(True)
        self.diagnosis.setTextFormat(Qt.TextFormat.RichText)
        self.diagnosis.setStyleSheet("QLabel { background: #f6f6f4; border: 1px solid #d8d8d4; padding: 6px; }")
        self.details = QLabel("Select a run.")
        self.details.setWordWrap(True)
        self.metrics = QTableWidget(0, 7)
        self.metrics.setHorizontalHeaderLabels(["Class", "IoU", "Dice", "Precision", "Recall", "True pixels",
                                                "Predicted pixels"])
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history = QTableWidget(0, 4)
        self.history.setHorizontalHeaderLabels(["Epoch", "Training loss", "Validation loss", "Validation mIoU (features)"])
        self.history.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.overlay_pick = QComboBox()
        self.overlay_pick.currentIndexChanged.connect(self._show_overlay)
        self.overlay = ImageView("Test image | ground truth | prediction")
        overlays = QWidget()
        ov = QVBoxLayout(overlays)
        ov.addWidget(self.overlay_pick)
        ov.addWidget(self.overlay, 1)
        self.stats_class = QComboBox()
        self.stats_um = QLineEdit()
        self.stats_um.setPlaceholderText("micrometres per pixel (optional)")
        self.stats_btn = QPushButton("Compute feature statistics")
        self.stats_btn.setToolTip("Area fraction, feature count, size and orientation distributions of ground truth "
                                  "vs prediction, computed with HydrideSegmentation's analysis functions.")
        self.stats_btn.clicked.connect(self.compute_feature_stats)
        self.stats_note = QLabel("")
        self.stats_note.setWordWrap(True)
        self.stats_table = QTableWidget(0, 7)
        self.stats_table.setHorizontalHeaderLabels(["Test image", "Area fraction (truth)", "Area fraction (pred)",
                                                    "Features (truth)", "Features (pred)", "Orientation KS",
                                                    "Size KS"])
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats = QWidget()
        sv = QVBoxLayout(stats)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Class:"))
        srow.addWidget(self.stats_class)
        srow.addWidget(self.stats_um, 1)
        srow.addWidget(self.stats_btn)
        sv.addLayout(srow)
        sv.addWidget(self.stats_note)
        sv.addWidget(self.stats_table, 1)
        self.stats_busy = False
        tabs = QTabWidget()
        tabs.addTab(self.metrics, "Per-class test metrics")
        tabs.addTab(overlays, "Test overlays")
        tabs.addTab(self.history, "Training history")
        tabs.addTab(stats, "Feature statistics")
        obs = QWidget()
        ol = QVBoxLayout(obs)
        self.notes_view = QLabel("")
        self.notes_view.setWordWrap(True)
        self.notes_view.setTextFormat(Qt.TextFormat.RichText)
        self.notes_view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.note_edit = QTextEdit()
        self.note_edit.setPlaceholderText("What did you look at, what works, what does not, what is this model for? "
                                          "Notes are kept with the run, cannot be edited later, and are published with "
                                          "it.")
        self.note_edit.setMaximumHeight(90)
        add_note_btn = QPushButton("Add observation")
        add_note_btn.clicked.connect(self.add_observation)
        ol.addWidget(QLabel("Observations (newest last):"))
        ol.addWidget(self.notes_view, 1)
        ol.addWidget(self.note_edit)
        ol.addWidget(add_note_btn)
        tabs.addTab(obs, "Observations")
        self.card_view = QTextEdit()
        self.card_view.setReadOnly(True)
        tabs.addTab(self.card_view, "Model card")
        self.result_tabs = tabs

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("Training runs (newest first):"))
        lv.addWidget(self.runs, 1)
        row = QHBoxLayout()
        row.addWidget(refresh)
        row.addWidget(self.open_btn)
        lv.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(self.resume_btn)
        row2.addWidget(self.predict_btn)
        lv.addLayout(row2)
        lv.addWidget(self.report_btn)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.banner)
        rv.addWidget(self.details)
        rv.addWidget(self.diagnosis)
        rv.addWidget(tabs, 1)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 750])
        layout = QVBoxLayout(self)
        layout.addWidget(hint(HELP))
        layout.addWidget(splitter, 1)
        ctx.runs_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        from .compare_page import network_name

        current = self.runs.currentItem().data(256) if self.runs.currentItem() else None
        self.runs.clear()
        for run in list_runs(self.ctx.workspace.runs_dir):
            state = run.state()
            item = QListWidgetItem(f"{run.run_id}\n   {state} - {network_name(run)} - "
                                   f"{run.info().get('dataset', {}).get('dataset_id', '?')}")
            item.setData(256, str(run.root))
            self.runs.addItem(item)
            if str(run.root) == current:
                self.runs.setCurrentItem(item)
        if self.runs.currentItem() is None and self.runs.count():
            self.runs.setCurrentRow(0)

    def selected_run(self) -> RunDir | None:
        item = self.runs.currentItem()
        from pathlib import Path

        return RunDir(Path(item.data(256))) if item else None

    def _show_run(self, *_args) -> None:
        run = self.selected_run()
        self.metrics.setRowCount(0)
        self.history.setRowCount(0)
        self.overlay_pick.clear()
        self.overlay.set_pixmap(None)
        self.stats_class.clear()
        self.stats_table.setRowCount(0)
        self.stats_note.setText("")
        self.resume_btn.setEnabled(bool(run and run.resumable()))
        self.predict_btn.setEnabled(bool(run and (run.checkpoints / "best.pt").exists()))
        if run is None:
            return
        info, status = run.info(), run.status()
        cfg = info.get("config", {})
        for c in info.get("class_map", {}).get("classes", [])[1:]:
            self.stats_class.addItem(c["name"])
        idx = self.stats_class.findText("hydride")
        if idx >= 0:
            self.stats_class.setCurrentIndex(idx)
        evaluation_path = run.evaluation / "evaluation.json"
        report = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None
        self.banner.setText(self._review_banner(run) if report else "")
        lines = [
            f"<b>State:</b> {run.state()} - {status.get('message', '')}"
            + (" <i>This run can be resumed.</i>" if run.resumable() else ""),
            f"<b>Dataset:</b> {info.get('dataset', {}).get('dataset_id')} &nbsp; <b>Preset:</b> {cfg.get('preset')} "
            f"&nbsp; <b>Epochs:</b> {cfg.get('epochs')} &nbsp; <b>Seed:</b> {cfg.get('seed')}",
            f"<b>Device:</b> {info.get('device', {}).get('name')} ({info.get('device', {}).get('kind')}) &nbsp; "
            f"<b>Software:</b> {info.get('tool', {}).get('id')} {info.get('tool', {}).get('version')}, "
            f"torch {info.get('environment', {}).get('torch')}",
        ]
        if report and report.get("metrics"):
            m = report["metrics"]
            lines.append(f"<b>{report['split'].title()} split</b> ({len(report['images'])} images, checkpoint from "
                         f"epoch {report['checkpoint_epoch']}): mean IoU {_f(m['mean_iou'])}, feature mean IoU "
                         f"{_f(m['mean_iou_foreground'])}, pixel accuracy {_f(m['pixel_accuracy'])}")
            if report.get("note"):
                lines.append(f"<i>{report['note']}</i>")
            self.metrics.setRowCount(len(m["per_class"]))
            for r, c in enumerate(m["per_class"]):
                for col, v in enumerate([c["name"], _f(c["iou"]), _f(c["dice"]), _f(c["precision"]),
                                         _f(c["recall"]), str(c["true_pixels"]), str(c["predicted_pixels"])]):
                    self.metrics.setItem(r, col, QTableWidgetItem(v))
            self.metrics.resizeColumnsToContents()
            for stem in report["images"]:
                self.overlay_pick.addItem(stem)
        elif status.get("state") == "completed":
            lines.append("No evaluation found.")
        self.details.setText("<br>".join(lines))
        from ..training.diagnostics import diagnose_run

        self._show_notes(run)
        self._show_card(run)

        marks = {"ok": ("&#10003;", "#1a7f37"), "info": ("&#8505;", "#1f5fae"), "warn": ("&#9888;", "#9a5b00"),
                 "block": ("&#10007;", "#b42318")}
        diag = diagnose_run(run)
        self.diagnosis.setText("<b>Run check</b><br>" + "<br>".join(
            f"<span style='color:{marks[lv][1]}'>{marks[lv][0]}</span> {msg}" for lv, msg in diag))
        epochs = run.epoch_history()
        self.history.setRowCount(len(epochs))
        for r, e in enumerate(epochs):
            for col, v in enumerate([str(e["epoch"]), _f(e["train_loss"]), _f(e["val_loss"]),
                                     _f(e["val_mean_iou_foreground"])]):
                self.history.setItem(r, col, QTableWidgetItem(v))

    def _show_card(self, run: RunDir) -> None:
        import html as _html

        from ..training.model_card import build_model_card

        if not (run.checkpoints / "best.pt").exists():
            self.card_view.setHtml("<p>No trained model yet.</p>")
            return
        c = build_model_card(run)
        net, data, res, inp = c["network"], c["training_data"], c["results"], c["input"]
        prof = inp.get("training_images") or {}
        bright = prof.get("mean_brightness") or {}

        def row(k: str, v: object) -> str:
            return f"<tr><th align='left'>{_html.escape(k)}</th><td>{_html.escape(str(v))}</td></tr>"

        rows = [
            row("Network", f"{net.get('title')} ({net.get('architecture')}, {net.get('parameters'):,} parameters)"
                if net.get("parameters") else net.get("title")),
            row("Defined in", net.get("defined_in")),
            row("Classes", ", ".join(f"{x['value']}={x['name']}" for x in (c["classes"] or {}).get("classes", []))),
            row("Input", f"{inp.get('channels')} channel(s), {', '.join(prof.get('dtypes', []))}, normalization "
                         f"{inp.get('normalization')}, resize x{inp.get('resize_factor')}, border "
                         f"{inp.get('exclude_border')}"),
            row("Training image brightness", f"{bright.get('min', 0):.2f}-{bright.get('max', 0):.2f}" if bright else "-"),
            row("Training data", f"{data.get('dataset_id')} - " + ", ".join(
                f"{label} {(data.get('split_counts') or {}).get(key, 0)}"
                for key, label in (("train", "training"), ("val", "validation"), ("test", "test")))),
            row("Annotation source", ((data.get("annotation_source") or {}).get("tool") or {}).get("id", "folder")
                + (" - contains unapproved masks" if data.get("contains_unapproved_annotations") else "")),
            row("Test results", f"{res.get('split')} split: mean IoU {_f4(res.get('mean_iou'))}, feature IoU "
                                f"{_f4(res.get('mean_iou_foreground'))}"),
            row("Review", _plain(self._review_banner(run))),
            row("Software", f"{(c['training'].get('tool') or {}).get('id')} {(c['training'].get('tool') or {}).get('version')}"
                            f", torch {(c['training'].get('environment') or {}).get('torch')}"),
            row("Checkpoint SHA-256", (c["checkpoint"] or {}).get("sha256")),
            row("Observations", len(c["observations"])),
            row("How to use", c["how_to_use"]),
        ]
        self.card_view.setHtml("<table cellpadding='3'>" + "".join(rows) + "</table><p style='color:#6b6b6b'>The full "
                               "card is saved as model_card.json in the run folder and published with the run.</p>")

    def _show_notes(self, run: RunDir) -> None:
        import html as _html

        from ..training.model_card import read_notes

        notes = read_notes(run)
        self.notes_view.setText("<br>".join(
            f"<b>{_html.escape(n['author'])}</b> <span style='color:#6b6b6b'>{n['time'][:16].replace('T', ' ')} - "
            f"{n['kind']}</span><br>{_html.escape(n['text'])}" for n in notes) or "No observations yet.")

    def add_observation(self) -> None:
        from ..collab.settings import CollabSettings
        from ..training.model_card import NoteError, add_note

        run = self.selected_run()
        if run is None:
            self.ctx.error(self, "Observation", "Select a run first.")
            return
        try:
            add_note(run, self.note_edit.toPlainText(), CollabSettings.load(self.ctx.workspace).user_name())
        except NoteError as exc:
            self.ctx.error(self, "Observation", str(exc))
            return
        self.note_edit.clear()
        self._show_notes(run)
        self._show_card(run)
        self.ctx.status_message.emit(f"Observation saved with {run.run_id}.")

    def export_report(self) -> None:
        import re

        from ..training.report import write_report

        run = self.selected_run()
        if run is None or not (run.evaluation / "evaluation.json").exists():
            self.ctx.error(self, "Export report", "Select a completed run first.")
            return
        review = re.sub("<[^>]+>", "", self._review_banner(run)).replace("&#9888;", "").replace("&#10003;", "") \
            .replace("&#10007;", "").strip()
        self.last_report = write_report(run, review or "unreviewed")
        self.ctx.status_message.emit(f"Report written to {self.last_report}")
        if self.ctx.interactive:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_report)))

    def _review_banner(self, run: RunDir) -> str:
        """Review status from the shared folder when connected; otherwise (or if unpublished) UNREVIEWED."""
        from ..collab.settings import CollabSettings
        from ..collab.share import SharedStore, ShareError

        settings = CollabSettings.load(self.ctx.workspace)
        if not settings.share_path:
            return UNREVIEWED
        try:
            store = SharedStore.open(Path(settings.share_path))
            store.run_record(run.run_id)
            reviews = store.reviews(run.run_id)
        except (ShareError, OSError):
            return UNREVIEWED + " <span style='color:#6b6b6b'>(not published on the shared folder)</span>"
        if not reviews:
            return UNREVIEWED + " <span style='color:#6b6b6b'>(published; awaiting review)</span>"
        last = reviews[-1]
        who = f"{last['reviewer']} on {last['reviewed_at'][:10]}"
        if last["decision"] == "approved":
            return (f"<span style='color:#1a7f37'><b>&#10003; APPROVED</b> by {who}</span> - "
                    f"{last.get('comment', '')}")
        label = "CHANGES REQUESTED" if last["decision"] == "needs_changes" else "REJECTED"
        return f"<span style='color:#b42318'><b>&#10007; {label}</b> by {who}</span> - {last.get('comment', '')}"

    def _show_overlay(self) -> None:
        run = self.selected_run()
        stem = self.overlay_pick.currentText()
        if run and stem:
            self.overlay.set_pixmap(QPixmap(str(run.evaluation / f"{stem}.png")))

    def _open_folder(self) -> None:
        run = self.selected_run()
        if run:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.root)))

    def _resume(self) -> None:
        run = self.selected_run()
        if run and self.resume_callback and self.resume_callback(run):
            self.ctx.status_message.emit(f"Resumed {run.run_id}; see step 3 for progress.")

    def _predict_dialog(self) -> None:
        run = self.selected_run()
        if not run:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder with new micrographs")
        if not folder:
            return
        out = QFileDialog.getSaveFileName(self, "Name of the new results folder",
                                          str(Path(folder).parent / f"predictions-{run.run_id}"))[0]
        if out:
            self.predict(run, Path(folder), Path(out))

    def predict(self, run: RunDir, input_folder: Path, output_folder: Path) -> None:
        from ..training.predict import predict_folder

        self.predicting = True
        self.predict_btn.setEnabled(False)
        self.ctx.status_message.emit(f"Segmenting images in {input_folder}...")

        def done(report: dict) -> None:
            self.predicting = False
            self.last_prediction = report
            self.predict_btn.setEnabled(True)
            skipped = f"; {len(report['errors'])} skipped" if report["errors"] else ""
            if report.get("warnings"):
                skipped += f"; {len(report['warnings'])} warning(s) - see predictions.json"
                self.ctx.error(self, "Check these images", "; ".join(f"{w['file']}: {w['warning']}"
                                                                     for w in report["warnings"][:5]))
            self.ctx.status_message.emit(f"{len(report['images'])} image(s) segmented into {output_folder}{skipped}. "
                                         "Results are unreviewed.")
            if self.ctx.interactive:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_folder)))

        def failed(message: str, details: str) -> None:
            self.predicting = False
            self.predict_btn.setEnabled(True)
            self.ctx.error(self, "Segment new images", message, details)

        run_background(lambda: predict_folder(run, input_folder, output_folder), done, failed)

    def compute_feature_stats(self) -> None:
        from ..integrations.hydrideseg import analyse_run, availability

        run = self.selected_run()
        if run is None or not (run.evaluation / "evaluation.json").exists():
            self.ctx.error(self, "Feature statistics", "Select a completed run first.")
            return
        ok, message = availability()
        if not ok:
            self.stats_note.setText(f"<span style='color:#8a5a00'>{message}</span>")
            self.ctx.error(self, "Feature statistics", message)
            return
        try:
            um = float(self.stats_um.text()) if self.stats_um.text().strip() else None
        except ValueError:
            self.ctx.error(self, "Feature statistics", "Micrometres per pixel must be a number.")
            return
        cls, datasets_dir = self.stats_class.currentText(), self.ctx.workspace.datasets_dir
        self.stats_busy = True
        self.stats_btn.setEnabled(False)

        def done(report: dict) -> None:
            self.stats_busy = False
            self.stats_btn.setEnabled(True)
            rows = report["per_image_distances"]
            self.stats_table.setRowCount(len(rows))
            for r, (stem, d) in enumerate(rows.items()):
                values = [stem, f"{d['mask_area_fraction_gt']:.4f}", f"{d['mask_area_fraction_pred']:.4f}",
                          f"{d['hydride_count_gt']:.0f}", f"{d['hydride_count_pred']:.0f}",
                          f"{d['hydride_orientation_ks']:.3f}", f"{d['hydride_size_ks']:.3f}"]
                for c, v in enumerate(values):
                    self.stats_table.setItem(r, c, QTableWidgetItem(v))
            self.stats_table.resizeColumnsToContents()
            self.stats_note.setText(f"{report['split'].title()} split, class '{cls}'. Computed with "
                                    f"{report['computed_with']}. KS = Kolmogorov-Smirnov distance between truth and "
                                    "prediction distributions (0 = identical). Saved in the run's evaluation folder. "
                                    "Unreviewed.")

        def failed(message: str, details: str) -> None:
            self.stats_busy = False
            self.stats_btn.setEnabled(True)
            self.ctx.error(self, "Feature statistics", message, details)

        run_background(lambda: analyse_run(run.root, datasets_dir, cls, um), done, failed)


def _f4(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def _plain(html_text: str) -> str:
    import html as _html
    import re

    return _html.unescape(re.sub(r"<[^>]+>", "", html_text)).replace("\u2713", "").replace("\u26a0", "").strip()
