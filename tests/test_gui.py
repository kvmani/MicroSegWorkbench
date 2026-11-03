"""Offscreen walk-through of the desktop workflow, including a real training subprocess."""

import time

import pytest

pytest.importorskip("torch")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)  # also skips if Qt system libs are missing

from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.runs import RunDir  # noqa: E402

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
    assert all(t.has_image for t in aug.tiles) and "patch" in aug.captions.text()
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
    assert "Run check" in res.diagnosis.text()
    res.add_observation()  # empty: refused
    assert "Write the observation" in res.ctx.last_error
    res.note_edit.setPlainText("Pore boundaries a bit soft.")
    res.add_observation()
    assert "Pore boundaries" in res.notes_view.text() and not res.note_edit.toPlainText()
    card = res.card_view.toPlainText()  # the card tab follows the new observation
    assert "Observations\t1" in card or "Observations 1" in card.replace("\n", " ")
    assert "{'" not in card and "training " in card
    res.export_report()
    assert res.last_report is not None and "UNREVIEWED" in res.last_report.read_text(encoding="utf-8")
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
    aug.resize_factor.setValue(0.5)
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
    alice.publish_run()  # refused: no observations yet
    assert "observations" in alice.ctx.last_error
    alice.publish_note.setPlainText("Hydrides look right on the overlays.")
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
    assert "approved" in bob.history.text() and "Hydrides look right" in bob.history.text()
    bob_results = bob_win.results_page
    bob_results.refresh()
    assert "APPROVED" in bob_results.banner.text() and "bob" in bob_results.banner.text()
    assert "APPROVED by bob" in bob_results.card_view.toPlainText()  # card tab agrees with the banner
    assert "unreviewed" not in bob_results.details.text().lower()
    bob_results.result_tabs.setCurrentIndex(bob_results.result_tabs.count() - 1)
    card = bob_results.card_view.toPlainText()
    assert "U-Net" in card and version.dataset_id in card and "Checkpoint SHA-256" in card

    # A third colleague fetches the approved model and segments new images in one step.
    wc = Workspace(tmp_path / "carol").ensure()
    carol_win = MainWindow(wc, interactive=False)
    carol = carol_win.share_page
    carol.path.setText(str(tmp_path / "share"))
    carol.user.setText("carol")
    carol.save_settings()
    carol.runs.selectRow(0)
    out = tmp_path / "carol-pred"
    carol.download_and_use(carol.selected_run_id(), tmp_path / "data", out)
    wait_until(app, lambda: not carol.busy and (out / "predictions.json").exists()
               and not carol_win.results_page.predicting)
    assert carol_win.stack.currentWidget() is carol_win.results_page
    assert (wc.runs_dir / "alice-run" / "checkpoints" / "best.pt").exists()


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
    assert "Rare classes (pore)" in data.summary.text()
    assert [r[0] for r in data.class_bars.rows] == ["matrix", "hydride", "pore"]
    assert abs(sum(r[2] for r in data.class_bars.rows) - 1) < 1e-6
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


def test_repeat_with_seeds_queue_and_compare_summary(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    train = win.train_page
    train.device.setCurrentIndex(train.device.findData("cpu"))
    train.epochs.setValue(1)
    train.repeats.setValue(3)
    train.start()
    wait_until(app, lambda: len(train.series_runs) == 3 and train.final_state == "completed"
               and not train.is_running(), timeout=300)
    seeds = [RunDir(ws.runs_dir / r).info()["config"]["seed"] for r in train.series_runs]
    assert seeds == [1234, 1235, 1236]
    compare = win.compare_page
    compare.refresh()
    compare.set_checked(train.series_runs)
    assert "differ only by random seed" in compare.warnings.text() and "&plusmn;" in compare.warnings.text()
    win.close()


def test_plugin_operation_appears_in_augmentation_page(app, tmp_path):
    from microseg_workbench.core.augment import PHOTOMETRIC_OPS, register_photometric_op
    from microseg_workbench.gui.app import MainWindow

    register_photometric_op("sharpen_test", lambda img, rng, p: img, {}, "test")
    try:
        win = MainWindow(Workspace(tmp_path / "ws"), interactive=False)
        aug = win.augment_page
        check, prob = aug.extra["sharpen_test"]
        check.setChecked(True)
        prob.setValue(0.3)
        aug.inputs["blur_max_sigma"].setValue(1.0)
        cfg = aug.config()
        assert cfg.extra_ops == ({"name": "sharpen_test", "probability": 0.3},) and cfg.blur_max_sigma == 1.0
        win.close()
    finally:
        PHOTOMETRIC_OPS.pop("sharpen_test", None)


def test_visual_augmentation_builder(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=96))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    aug = win.augment_page
    aug.preview()
    assert all(t.has_image for t in aug.tiles)
    assert all(c.thumb_a.has_image and c.thumb_b.has_image for c in aug.cards.values())
    # presets drive the blocks; the matching preset button is highlighted
    aug.use_preset("gentle")
    assert aug.config().rot90 == 0 and not aug.cards["rot90"].enable.isChecked()
    assert aug.preset_buttons["gentle"].isChecked()
    # switching a block on restores a sensible value; off sets it to zero; the dots follow
    aug.cards["rot90"].enable.setChecked(True)
    assert aug.config().rot90 == 0.5 and aug.cards["rot90"].dots.fraction == 0.5
    assert not aug.preset_buttons["gentle"].isChecked()  # no longer a preset
    aug.cards["blur_max_sigma"].enable.setChecked(True)
    assert aug.config().blur_max_sigma == 1.0
    aug.cards["blur_max_sigma"].enable.setChecked(False)
    assert aug.config().blur_max_sigma == 0
    # sliders and exact values stay in sync; advanced mode shows the exact boxes
    aug.sliders["hflip"].slider.setValue(aug.sliders["hflip"].steps)
    assert aug.config().hflip == 1.0 and "100 %" in aug.sliders["hflip"].value_label.text()
    assert not aug.inputs["hflip"].isVisibleTo(aug)
    aug.set_advanced(True)
    assert aug.inputs["hflip"].isVisibleTo(aug)
    # step-by-step follows one patch through the pipeline; selecting a block shows its effect
    aug.preview()
    aug.show_steps(2)
    assert aug.steps_layout.count() >= 3
    aug.select_step("brightness")
    assert aug.detail_view.has_image() and "Brightness" in aug.detail_title.text()
    # sanity check runs the real pipeline and reports
    aug.run_check()
    wait_until(app, lambda: not aug.checking)
    assert aug.last_check and aug.rates.rowCount() >= len(aug.cards) - len(aug.extra)
    # recipes
    aug.save_recipe("mine")
    aug.use_preset("none")
    aug.load_recipe("mine")
    assert aug.config().hflip == 1.0 and aug.recipes.currentText() == "mine"
    # training picks the builder's settings up
    assert win.train_page.build_config().augment.hflip == 1.0
    win.close()


def test_guided_workflow_frame(app, tmp_path):
    from microseg_workbench.gui.app import MainWindow
    from microseg_workbench.gui.workflow import ATTENTION, DONE, LOCKED, TODO

    ws = Workspace(tmp_path / "ws")
    win = MainWindow(ws, interactive=False)
    win.refresh_status()
    states = [s.state for s in win.statuses]
    assert states[0] == TODO and states[1] == LOCKED and states[2] == LOCKED and states[3] == LOCKED
    assert "Start here" in win.step_state.text() and "Next: Preprocess" in win.next_btn.text()
    assert not win.train_page.start_btn.isEnabled() and "No dataset version" in win.train_page.ready.text()
    # Beginner mode (default) hides expert options; Advanced shows them and is remembered per workspace
    assert not win.train_page.lr.isVisibleTo(win.train_page) and win.train_page.epochs.isVisibleTo(win.train_page)
    assert not win.data_page.seed.isVisibleTo(win.data_page)
    win.advanced_btn.click()
    assert win.train_page.lr.isVisibleTo(win.train_page) and win.data_page.seed.isVisibleTo(win.data_page)
    win.ctx.set_advanced(False)
    assert win.beginner_btn.isChecked() and not win.advanced_btn.isChecked()
    win.ctx.set_advanced(True)
    assert win.advanced_btn.isChecked()
    assert win.augment_page.inputs["hflip"].isVisibleTo(win.augment_page)
    win.close()
    assert MainWindow(ws, interactive=False).ctx.advanced is True

    win = MainWindow(ws, interactive=False)
    data = win.data_page
    data.create_demo_data()
    wait_until(app, lambda: data.report is not None and not data.busy)
    data.create_version()
    wait_until(app, lambda: data.last_version_id is not None and not data.busy)
    win.refresh_status()
    states = [s.state for s in win.statuses]
    assert states[0] == DONE and states[1] == TODO and states[2] == TODO
    win.next_btn.click()
    assert win.nav.currentRow() == 1
    aug = win.augment_page
    aug.use_preset("standard")
    aug.run_check()
    wait_until(app, lambda: not aug.checking)
    win.refresh_status()
    assert win.statuses[1].state == DONE
    aug.cards["blur_max_sigma"].enable.setChecked(True)  # settings changed after the check
    win.refresh_status()
    assert win.statuses[1].state == TODO and "changed" in win.statuses[1].reason
    aug.inputs["blur_max_sigma"].setValue(4.0)
    aug.run_check()
    wait_until(app, lambda: not aug.checking)
    win.refresh_status()
    assert win.statuses[1].state == ATTENTION
    assert "to review in step 2" in win.train_page.ready.text() and win.train_page.start_btn.isEnabled()
    aug.border[0].setValue(500)  # preprocessing that cannot work blocks the start button
    win.refresh_status()
    assert not win.train_page.start_btn.isEnabled() and "leaves only" in win.train_page.ready.text()
    win.close()


def test_visual_preprocessing(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=96))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    aug = win.augment_page
    aug.main_tabs.setCurrentIndex(0)
    aug.preview()
    ed = aug.border_editor
    assert (ed.img_h, ed.img_w) == (96, 96)
    ed.drag_edge_to("bottom", 80)  # drag the bottom edge up by 16 px
    assert aug.border[1].value() == 16 and aug.preprocess().exclude_border == (0, 16, 0, 0)
    ed.drag_edge_to("top", 200)  # cannot drag past the other edge: at least 8 px stay
    assert aug.border[0].value() == 96 - 16 - 8
    aug.border[0].setValue(0)
    aug.refresh_preprocess()
    assert ed.border == [0, 16, 0, 0] and "96 x 80 px" in aug.size_info.text()
    aug.resize_factor.setValue(0.25)
    aug.refresh_preprocess()
    assert aug.size_before.has_image() and aug.size_after.has_image()
    assert "would be thinner" in aug.size_info.text() and "hydride" in aug.size_info.text()  # 2-3 px features break up
    aug.resize_factor.setValue(1.0)
    aug.refresh_preprocess()
    assert "thinner" not in aug.size_info.text() and "hydride about" in aug.size_info.text()
    win.close()


def test_visual_data_step(app, tmp_path):
    import numpy as np
    from PIL import Image

    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    folder = tmp_path / "data"
    write_synthetic_dataset(folder, SyntheticSpec(specimens=5, images_per_specimen=2, size=64))
    bad = np.asarray(Image.open(folder / "S03_0002_mask.png")).copy()
    bad[0, 0] = 9
    Image.fromarray(bad).save(folder / "S03_0002_mask.png")
    win = MainWindow(Workspace(tmp_path / "ws"), interactive=False)
    data = win.data_page
    data.folder.setText(str(folder))
    data.check_folder()
    wait_until(app, lambda: data.report is not None and not data.busy and data.thumbs_ready)
    assert data.gallery.count() == 10 and all(not data.gallery.item(i).icon().isNull() for i in range(10))
    bad_row = next(i for i, p in enumerate(data.report.pairs) if p.stem == "S03_0002")
    assert data.gallery.currentRow() == bad_row  # the first problem is selected and shown
    assert data.gallery.item(bad_row).text().startswith("\u26a0") and "not declared" in data.gallery.item(bad_row).toolTip()
    assert data.split_view.error  # split waits until the folder is fixed
    data.gallery.setCurrentRow(0)
    assert data._selected_row() == 0  # gallery and list stay in sync
    issue = next(data.issues.item(i) for i in range(data.issues.count()) if "S03_0002" in data.issues.item(i).text())
    data._issue_clicked(issue)
    assert data._selected_row() == bad_row
    data.opacity.setValue(0)
    assert data.preview.has_image()
    Image.fromarray(np.where(bad == 9, 0, bad).astype(np.uint8)).save(folder / "S03_0002_mask.png")
    data.check_folder()
    wait_until(app, lambda: data.report is not None and data.report.ok and not data.busy)
    plan = data.split_view.groups
    assert set(plan) == {"S01", "S02", "S03", "S04", "S05"} and {s for s, _ in plan.values()} == {"train", "val", "test"}
    assert all(n == 2 for _, n in plan.values()) and not data.split_view.error
    win.close()


def test_workflow_map_and_run_summary(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MAP_NODES, MainWindow
    from microseg_workbench.gui.workflow import DONE, LOCKED

    ws = Workspace(tmp_path / "ws").ensure()
    win = MainWindow(ws, interactive=False)
    win.refresh_status()
    assert [n[0] for n in win.map.nodes] == [n[0] for n in MAP_NODES]
    assert win.map.nodes[1][1] == LOCKED and win.map.nodes[4][1] == LOCKED
    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    version = create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                                     ws.datasets_dir, "d")
    win.ctx.datasets_changed.emit()
    win.refresh_status()
    states = [n[1] for n in win.map.nodes]
    assert states[1] == DONE and states[2] == DONE and states[5] == LOCKED
    win.open_map_node(2)  # Preprocess node opens step 2, tab A
    assert win.nav.currentRow() == 1 and win.augment_page.main_tabs.currentIndex() == 0
    win.open_map_node(3)
    assert win.augment_page.main_tabs.currentIndex() == 1
    win.open_map_node(4)
    assert win.nav.currentRow() == 2 and win.map.current == 4
    train = win.train_page
    train.device.setCurrentIndex(train.device.findData("cpu"))
    train.refresh_estimate()
    wait_until(app, lambda: train.estimate is not None and not train.estimating)
    est = train.estimate
    n_train = version.manifest["split_counts"]["train"]
    assert est["patches_per_epoch"] == n_train * 8 and est["steps_per_epoch"] == -(-n_train * 8 // 8)
    assert est["total_seconds"] > 0
    assert "This run:" in train.summary_card.text() and "about" in train.summary_card.text()
    win.close()


def test_network_choice_reaches_training(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    train = win.train_page
    assert train.network.isVisibleTo(train)  # visible in Beginner mode too
    assert {train.network.itemData(i) for i in range(train.network.count())} >= {"unet", "segformer"}
    train.network.setCurrentIndex(train.network.findData("segformer"))
    assert "transformer" in train.network_note.text()
    assert train.build_config().architecture == "segformer"
    train.refresh_estimate()
    wait_until(app, lambda: train.estimate is not None and not train.estimating)
    assert "SegFormer" in train.summary_card.text()
    win.close()


def test_same_image_several_models_tab(app, tmp_path):
    from microseg_workbench.core.augment import AugmentConfig
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow
    from microseg_workbench.training.config import TrainConfig
    from microseg_workbench.training.trainer import run_training

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    v = create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                               ws.datasets_dir, "d")
    for arch in ("unet", "segformer"):
        run_training(TrainConfig(dataset_id=v.dataset_id, architecture=arch, epochs=1, base_channels=4, depth=2,
                                 patches_per_image=1, device="cpu", augment=AugmentConfig(patch_size=32)), ws,
                     run_id=arch)
    win = MainWindow(ws, interactive=False)
    cmp = win.compare_page
    cmp.refresh()
    cmp.run_same_image()  # nothing ticked
    assert "Tick one or more" in win.ctx.last_error
    cmp.set_checked(["unet", "segformer"])
    labels = [cmp.runs.item(i).text() for i in range(cmp.runs.count())]
    assert any("SegFormer-lite" in t for t in labels) and any("U-Net" in t for t in labels)  # networks named
    assert cmp.si_sample.currentText().endswith("(test)")  # test images listed first
    cmp.run_same_image()
    wait_until(app, lambda: not cmp.si_busy)
    assert cmp.si_result and cmp.si_grid.count() == 5  # image, truth, two models, disagreement
    assert {cmp.si_grid.getItemPosition(i)[0] for i in range(5)} == {0}  # all on one row
    assert cmp.si_table.rowCount() == 2 and "Feature IoU (this image)" in [
        cmp.si_table.horizontalHeaderItem(c).text() for c in range(cmp.si_table.columnCount())]
    cmp.use_image_file(v.image_path(v.samples()[0]))  # any image file, no ground truth
    cmp.run_same_image()
    wait_until(app, lambda: not cmp.si_busy)
    assert cmp.si_grid.count() == 4
    win.close()


def test_estimate_follows_settings_changed_while_estimating(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    train = win.train_page
    train.device.setCurrentIndex(train.device.findData("cpu"))
    train.epochs.setValue(3)
    train.refresh_estimate()
    assert train.estimating
    train.epochs.setValue(7)  # changed while the first estimate is still running
    train.refresh_estimate()
    wait_until(app, lambda: train.estimate is not None and not train.estimating)
    assert train.estimate["epochs"] == 7 and "7 epochs" in train.summary_card.text()
    win.close()


def test_run_card_matches_the_started_run(app, tmp_path):
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow

    write_synthetic_dataset(tmp_path / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64))
    ws = Workspace(tmp_path / "ws").ensure()
    create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                           ws.datasets_dir, "d")
    win = MainWindow(ws, interactive=False)
    train = win.train_page
    train.device.setCurrentIndex(train.device.findData("cpu"))
    train.epochs.setValue(2)
    train.network.setCurrentIndex(train.network.findData("segformer"))
    train.refresh_estimate()
    wait_until(app, lambda: train.estimate is not None and not train.estimating)
    assert "SegFormer" in train.summary_card.text()
    train.network.setCurrentIndex(train.network.findData("unet"))  # changed, then Start before a new estimate
    train.start()
    assert "U-Net" in train.summary_card.text() and "SegFormer" not in train.summary_card.text()
    wait_until(app, lambda: train.final_state is not None, timeout=300)
    assert "SegFormer" not in train.summary_card.text()
    win.close()


def test_blocked_dataset_is_explained_everywhere(app, tmp_path):
    import numpy as np
    from PIL import Image

    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
    from microseg_workbench.gui.app import MainWindow
    from microseg_workbench.gui.workflow import ATTENTION

    folder = tmp_path / "data"
    write_synthetic_dataset(folder, SyntheticSpec(specimens=5, images_per_specimen=2, size=64, seed=5))
    for m in folder.glob("*_mask.png"):  # pores only in S03
        if not m.name.startswith("S03"):
            arr = np.asarray(Image.open(m)).copy()
            arr[arr == 2] = 0
            Image.fromarray(arr).save(m)
    win = MainWindow(Workspace(tmp_path / "ws"), interactive=False)
    win.resize(1440, 900)
    win.show()
    data = win.data_page
    data.folder.setText(str(folder))
    data.check_folder()
    wait_until(app, lambda: data.report is not None and data.report.ok and not data.busy)
    data.seed.setValue(1)  # S03 lands outside the training split
    data.create_version()
    wait_until(app, lambda: data.last_version_id is not None and not data.busy)
    app.processEvents()
    assert "does not occur in any training image" in data.result.text()
    assert data.result.height() >= data.result.heightForWidth(data.result.width())  # not clipped
    win.go_to(2)
    win.refresh_status()
    assert not win.train_page.start_btn.isEnabled()
    assert win.statuses[2].state == ATTENTION and "Cannot start yet" in win.step_state.text()
    win.close()
