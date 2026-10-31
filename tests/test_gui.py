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
