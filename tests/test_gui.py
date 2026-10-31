"""Offscreen walk-through of the desktop workflow, including a real training subprocess."""

import time

import pytest

pytest.importorskip("torch")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)  # also skips if Qt system libs are missing

from microseg_workbench.core.workspace import Workspace  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(app, condition, timeout=120.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for the UI")


def test_full_workflow(app, tmp_path):
    from microseg_workbench.gui.app import MainWindow

    win = MainWindow(Workspace(tmp_path / "ws"), interactive=False)
    win.show()
    data = win.data_page

    # step 1: demo data, validation, split preview, dataset version
    data.create_demo_data()
    wait_until(app, lambda: data.report is not None and not data.busy)
    assert data.report.ok, data.report.issues
    assert data.table.rowCount() == 20
    assert data.preview.has_image()
    assert "groups" in data.split_preview.text()
    data.create_version()
    wait_until(app, lambda: data.last_version_id is not None and not data.busy)
    assert data.last_version_id.startswith("demo-")

    # step 2: augmentation preview from the new dataset
    win.go_to(1)
    aug = win.augment_page
    assert aug.dataset.currentData() == data.last_version_id
    aug.preview()
    assert aug.view.has_image() and "Columns" in aug.captions.text()
    aug.inputs["patch_size"].setValue(24)
    assert aug.config().patch_size == 24
    aug.inputs["patch_size"].setValue(64)

    # step 3: tiny real training in a subprocess
    win.go_to(2)
    train = win.train_page
    assert train.dataset.currentData() == data.last_version_id
    train.epochs.setValue(3)
    train.device.setCurrentIndex(train.device.findData("cpu"))
    train.start()
    assert train.is_running()
    wait_until(app, lambda: train.final_state is not None, timeout=300)
    assert train.final_state == "completed", (train.stderr, train.ctx.last_error)
    assert train.loss_chart.point_count() == 6
    assert train.iou_chart.point_count() == 3
    assert train.pred_view.has_image() and train.truth_view.has_image()
    assert "measured" in train.throughput.text() or train.ram_bar.bar.value() >= 0
    assert train.ram_bar.bar.format() != "not available"

    # step 4: results show the unreviewed test evaluation
    win.go_to(3)
    res = win.results_page
    assert res.runs.count() == 1
    assert "UNREVIEWED" in res.banner.text()
    assert res.metrics.rowCount() == 3
    assert res.history.rowCount() == 3
    assert res.overlay.has_image()
    win.close()


def test_errors_are_reported_not_crashing(app, tmp_path):
    from microseg_workbench.gui.app import MainWindow

    win = MainWindow(Workspace(tmp_path / "ws"), interactive=False)
    win.train_page.start()
    assert "dataset version" in win.ctx.last_error
    win.data_page.folder.setText(str(tmp_path / "nope"))
    win.data_page.check_folder()
    wait_until(app, lambda: win.data_page.report is not None)
    assert not win.data_page.report.ok
    assert not win.data_page.create_btn.isEnabled()
    win.augment_page.preview()
    assert "step 1" in win.ctx.last_error
    win.close()


def test_resume_compare_predict_and_simulated_gpu(app, tmp_path, monkeypatch):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow
    from microseg_workbench.training.monitor import SIMULATE_ENV

    monkeypatch.setenv(SIMULATE_ENV, "1")  # inherited by the training subprocess
    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=4, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    train, results, compare = win.train_page, win.results_page, win.compare_page
    train.device.setCurrentIndex(train.device.findData("cpu"))
    train.epochs.setValue(25)

    # start, cancel after the first epoch, resume from the Results page
    train.start()
    wait_until(app, lambda: len(train.epochs_seen) >= 1)
    train.cancel()
    wait_until(app, lambda: train.final_state is not None)
    assert train.final_state == "cancelled", train.stderr
    first_run = train.run
    results.refresh()
    assert results.selected_run().run_id == first_run.run_id
    assert results.resume_btn.isEnabled()
    results.resume_btn.click()
    assert win.nav.currentRow() == 2 and train.is_running()
    wait_until(app, lambda: train.final_state is not None, timeout=300)
    assert train.final_state == "completed", train.stderr
    assert [e["epoch"] for e in first_run.epoch_history()] == list(range(1, 26))
    assert train.loss_chart.point_count() == 50  # repeated epochs replaced, not duplicated
    assert "SIMULATED" in train.sim_label.text() and "SIMULATED" in train.vram_bar.bar.format()

    # a second, different run, then compare
    train.epochs.setValue(2)
    train.seed.setValue(7)
    train.start()
    wait_until(app, lambda: train.final_state is not None, timeout=300)
    assert train.final_state == "completed"
    compare.refresh()
    compare.set_checked([first_run.run_id, train.run.run_id])
    assert compare.table.columnCount() == 2
    assert compare.diff.rowCount() >= 2  # epochs and seed differ
    assert len(compare.chart.series) == 2
    assert compare.image_pick.count() > 0

    # segment new images with the first run
    results.refresh()
    results.predict(first_run, tmp_path / "data", tmp_path / "pred")
    wait_until(app, lambda: not results.predicting)
    assert results.last_prediction and len(results.last_prediction["images"]) == 8
    win.close()


def test_preprocess_controls_flow_into_training_config(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    aug = win.augment_page
    aug.resize.setValue(0.5)
    aug.normalization.setCurrentIndex(aug.normalization.findData("per_image_standardize"))
    aug.preview()
    assert "Resized from 64x64 to 32x32" in aug.captions.text()
    cfg = win.train_page.build_config()
    assert cfg.resize_factor == 0.5 and cfg.normalization == "per_image_standardize"
    win.close()


def test_share_page_two_colleagues(app, tmp_path):
    from microseg_workbench.collab.share import SharedStore
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow
    from microseg_workbench.training.config import TrainConfig
    from microseg_workbench.training.trainer import run_training

    SharedStore.create(tmp_path / "share")
    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    wa, wb = Workspace(tmp_path / "alice").ensure(), Workspace(tmp_path / "bob").ensure()
    version = create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                                     wa.datasets_dir, "d")
    run_training(TrainConfig(dataset_id=version.dataset_id, epochs=1, base_channels=4, depth=2, patches_per_image=1,
                             device="cpu"), wa, run_id="alice-run")

    alice = MainWindow(wa, interactive=False).share_page
    alice.path.setText(str(tmp_path / "share"))
    alice.user.setText("alice")
    alice.save_settings()
    assert "Connected" in alice.conn.text()
    alice.publish_dataset()
    wait_until(app, lambda: not alice.busy)
    alice.publish_run()
    wait_until(app, lambda: not alice.busy)
    assert alice.runs.rowCount() == 1 and alice.runs.item(0, 1).text() == "UNREVIEWED"
    alice.runs.selectRow(0)
    alice.decision.setCurrentIndex(alice.decision.findData("approved"))
    alice.submit_review()
    assert "someone other" in alice.ctx.last_error

    bob_win = MainWindow(wb, interactive=False)
    bob = bob_win.share_page
    bob.path.setText(str(tmp_path / "share"))
    bob.user.setText("bob")
    bob.save_settings()
    bob.runs.selectRow(0)
    bob.download_run()
    wait_until(app, lambda: not bob.busy)
    assert (wb.runs_dir / "alice-run").exists() and (wb.datasets_dir / version.dataset_id).exists()
    assert bob_win.results_page.runs.count() == 1  # results page refreshed via signal
    bob.runs.selectRow(0)
    bob.decision.setCurrentIndex(bob.decision.findData("approved"))
    bob.comment.setPlainText("Checked overlays")
    bob.submit_review()
    assert bob.runs.item(0, 1).text() == "APPROVED"
    bob.approved_only.setChecked(True)
    assert bob.runs.rowCount() == 1
    assert "approved" in bob.history.text()
    bob_results = bob_win.results_page
    bob_results.refresh()
    assert "APPROVED" in bob_results.banner.text() and "bob" in bob_results.banner.text()


def test_import_onlineannotator_export_in_data_page(app, tmp_path):
    from microseg_workbench.gui.app import MainWindow
    from oa_fixture import make_bundle

    splits = {f"S{s}_{i}": ("train" if s <= 2 else "val" if s == 3 else "test") for s in range(1, 5) for i in (1, 2)}
    bundle = make_bundle(tmp_path / "zr_export.zip", splits=splits, unapproved=("S1_2",))
    win = MainWindow(Workspace(tmp_path / "ws"), interactive=False)
    data = win.data_page
    data.import_bundle(bundle)
    wait_until(app, lambda: data.report is not None and not data.busy)
    assert data.report.ok and data.split_source.currentData() == "annotator"
    assert "NOT approved" in data.ctx.last_error  # warning surfaced to the user
    assert "train</b>: 4 images" in data.split_preview.text()
    data.create_version()
    wait_until(app, lambda: data.last_version_id is not None and not data.busy)
    from microseg_workbench.core.dataset import DatasetVersion

    v = DatasetVersion.open(win.ctx.workspace.datasets_dir / data.last_version_id)
    assert v.manifest["split_config"]["source"] == "annotator" and v.manifest["contains_unapproved_annotations"]
    win.close()


def test_feature_statistics_tab(app, tmp_path, monkeypatch):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow
    from microseg_workbench.integrations import hydrideseg
    from microseg_workbench.training.config import TrainConfig
    from microseg_workbench.training.trainer import run_training

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    v = create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                               ws.datasets_dir, "d")
    run_training(TrainConfig(dataset_id=v.dataset_id, epochs=1, base_channels=4, depth=2, patches_per_image=1,
                             device="cpu"), ws, run_id="r")
    win = MainWindow(ws, interactive=False)
    res = win.results_page
    assert res.stats_class.currentText() == "hydride"
    if not hydrideseg.availability()[0]:
        res.compute_feature_stats()
        assert "not available" in res.ctx.last_error
        return
    res.compute_feature_stats()
    wait_until(app, lambda: not res.stats_busy)
    assert res.stats_table.rowCount() == len(v.samples("test"))
    assert "HydrideSegmentation" in res.stats_note.text()


def test_frozen_launch_command_and_cli_dispatch(tmp_path, monkeypatch, capsys):
    import sys

    from microseg_workbench.gui.app import main
    from microseg_workbench.gui.train_page import worker_command

    prog, args = worker_command(["runs"])
    assert prog == sys.executable and args[:2] == ["-m", "microseg_workbench"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    prog, args = worker_command(["runs"])
    assert args == ["--cli", "runs"]
    assert main(["--cli", "--workspace", str(tmp_path / "ws"), "datasets"]) == 0


def test_border_controls_and_class_balance(app, tmp_path):
    from microseg_workbench.gui.app import MainWindow

    win = MainWindow(Workspace(tmp_path / "ws"), interactive=False)
    data = win.data_page
    data.create_demo_data()
    wait_until(app, lambda: data.report is not None and not data.busy)
    assert "share of all pixels" in data.summary.text() and "Rare classes (pore)" in data.summary.text()
    data.create_version()
    wait_until(app, lambda: data.last_version_id is not None and not data.busy)
    aug = win.augment_page
    aug.border[1].setValue(16)
    aug.preview()
    assert "analysed region 80x96" in aug.captions.text()
    assert win.train_page.build_config().exclude_border == (0, 16, 0, 0)
    aug.border[0].setValue(90)
    aug.preview()
    assert "leaves only" in win.ctx.last_error
    win.close()
