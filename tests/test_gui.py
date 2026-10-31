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
    assert "measured" in train.resources.text()

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
