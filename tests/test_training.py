"""Real tiny CPU training (no mocked success)."""

import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from microseg_workbench.core.augment import AugmentConfig  # noqa: E402
from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.config import TrainConfig, TrainConfigError, preset_config  # noqa: E402
from microseg_workbench.training.device import DeviceError, choose_device  # noqa: E402
from microseg_workbench.training.engine import load_checkpoint, predict_labels  # noqa: E402
from microseg_workbench.training.runs import RunDir  # noqa: E402
from microseg_workbench.training.trainer import run_training  # noqa: E402

pytestmark = pytest.mark.training


@pytest.fixture
def workspace_with_dataset(synthetic_folder, tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    version = create_dataset_version(report, SplitConfig(seed=4), ws.datasets_dir, "t")
    return ws, version


def tiny(dataset_id, **kw):
    base = dict(epochs=3, batch_size=4, base_channels=4, depth=2, patches_per_image=2, seed=99, device="cpu",
                num_threads=1, fixed_samples=2)
    base.update(kw)
    return TrainConfig(dataset_id=dataset_id, **base)


def epoch_events(run):
    return [e for e in run.read_events() if e["type"] == "epoch"]


def test_training_learns_and_writes_complete_run(workspace_with_dataset):
    ws, version = workspace_with_dataset
    cfg = tiny(version.dataset_id, epochs=10, base_channels=8, patches_per_image=6)
    run = run_training(cfg, ws)
    assert run.status()["state"] == "completed"
    epochs = epoch_events(run)
    assert [e["epoch"] for e in epochs] == list(range(1, 11))
    assert epochs[-1]["train_loss"] < 0.6 * epochs[0]["train_loss"]  # it really learned
    assert epochs[-1]["resources"]["simulated"] is False
    info = json.loads(run.run_json.read_text())
    assert info["dataset"]["dataset_id"] == version.dataset_id
    assert info["environment"]["torch"] == torch.__version__
    # fixed-sample predictions per epoch
    stems = info["fixed_samples"]["stems"]
    assert len(stems) == 2
    for e in (1, 10):
        for stem in stems:
            assert (run.samples / f"epoch_{e:03d}" / f"{stem}.png").exists()
    report = json.loads((run.evaluation / "evaluation.json").read_text())
    assert report["review_status"] == "unreviewed"
    assert report["split"] == "test"
    assert set(report["images"]) == {s["stem"] for s in version.samples("test")}
    assert report["metrics"]["mean_iou_foreground"] > 0.3
    # accuracy details: consistent with the confusion matrix, one row per test image, CSV for Excel
    acc = report["accuracy"]
    assert len(acc["per_image"]) == len(report["images"]) and acc["boundary_tolerance_px"] == 1
    for c, m in zip(acc["per_class"], report["metrics"]["per_class"]):
        total = report["metrics"]["evaluated_pixels"]
        assert c["area_fraction_true"] == pytest.approx(m["true_pixels"] / total)
        assert c["area_fraction_pred"] == pytest.approx(m["predicted_pixels"] / total)
    assert acc["mean_boundary_f1_foreground"] is not None and 0 <= acc["mean_boundary_f1_foreground"] <= 1
    assert all(abs(sum(r) - 1) < 1e-9 for r in acc["confusion_row_normalized"] if r[0] is not None)
    csv_lines = (run.evaluation / "per_image.csv").read_text().splitlines()
    assert len(csv_lines) == 1 + len(report["images"])
    from microseg_workbench.training.report import build_report

    assert "Boundaries and area fractions" in build_report(run)
    pred_mask = np.asarray(Image.open(run.evaluation / f"{report['images'][0]}_pred_mask.png"))
    assert set(np.unique(pred_mask)) <= set(version.class_map.values)
    model, ckpt = load_checkpoint(run.checkpoints / "best.pt")
    assert ckpt["dataset_id"] == version.dataset_id
    assert predict_labels(model, np.zeros((37, 50, 1), np.float32), torch.device("cpu")).shape == (37, 50)


def test_same_seed_reproduces_losses(workspace_with_dataset):
    ws, version = workspace_with_dataset
    a = run_training(tiny(version.dataset_id), ws)
    b = run_training(tiny(version.dataset_id), ws)
    c = run_training(tiny(version.dataset_id, seed=100), ws)
    losses = lambda r: [e["loss"] for e in r.read_events() if e["type"] == "step"]  # noqa: E731
    assert losses(a) == losses(b)
    assert losses(a) != losses(c)


def test_validation_only_training_warns_about_generalization(synthetic_folder, tmp_path):
    from microseg_workbench.training.diagnostics import diagnose_run
    from microseg_workbench.training.report import build_report

    ws = Workspace(tmp_path / "ws").ensure()
    version = create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX),
                                     SplitConfig(train=0.8, val=0.2, test=0), ws.datasets_dir, "no-test")
    run = run_training(tiny(version.dataset_id, epochs=1), ws)
    report = json.loads((run.evaluation / "evaluation.json").read_text())
    assert report["split"] == "val" and report["metrics"] is not None
    findings = diagnose_run(run)
    assert not any(lv == "ok" for lv, _ in findings)
    assert any(lv == "warn" and "independent test" in message for lv, message in findings)
    assert "No independent test result" in build_report(run)


def test_cancel_stops_gracefully(workspace_with_dataset):
    ws, version = workspace_with_dataset
    holder = {}

    def on_event(event):
        if event["type"] == "start":
            holder["run"] = True
        if event["type"] == "epoch" and event["epoch"] == 1:
            RunDir(ws.runs_dir / "cancel-me").request_cancel()

    run = run_training(tiny(version.dataset_id, epochs=5), ws, run_id="cancel-me", on_event=on_event)
    assert run.status()["state"] == "cancelled"
    assert len(epoch_events(run)) == 1
    assert (run.checkpoints / "last.pt").exists()


def test_failure_is_recorded_and_raised(tmp_path):
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(FileNotFoundError):
        run_training(tiny("missing-dataset"), ws, run_id="bad")
    run = RunDir(ws.runs_dir / "bad")
    assert run.status()["state"] == "failed"
    assert run.read_events()[-1]["state"] == "failed"


def test_corrupted_dataset_is_refused(workspace_with_dataset):
    ws, version = workspace_with_dataset
    version.mask_path(version.samples()[0]).write_bytes(b"not a png")
    with pytest.raises(ValueError, match="integrity"):
        run_training(tiny(version.dataset_id), ws)


def test_device_and_config_validation():
    if not torch.cuda.is_available():
        with pytest.raises(DeviceError, match="no usable NVIDIA GPU"):
            choose_device("cuda")
        device, _, note = choose_device("auto")
        assert device.type == "cpu" and "CPU" in note
    with pytest.raises(TrainConfigError):
        TrainConfig(dataset_id="x", epochs=0)
    with pytest.raises(TrainConfigError, match="divisible"):
        preset_config("standard", "x", depth=4, augment=AugmentConfig(patch_size=24))


def test_event_tail_skips_partial_lines(tmp_path):
    from microseg_workbench.training.runs import EventTail

    path = tmp_path / "events.jsonl"
    tail = EventTail(path)
    assert tail.read_new() == []
    path.write_text('{"a": 1}\n{"b": ')
    assert tail.read_new() == [{"a": 1}]
    with open(path, "a") as fh:
        fh.write('2}\n')
    assert tail.read_new() == [{"b": 2}]
    assert tail.read_new() == []
