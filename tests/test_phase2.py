"""Phase 2: resume, interruption detection, out-of-memory handling, telemetry, batch-size probing."""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training import monitor  # noqa: E402
from microseg_workbench.training.config import TrainConfig, preset_config  # noqa: E402
from microseg_workbench.training.device import is_out_of_memory, probe_batch_size  # noqa: E402
from microseg_workbench.training.model import UNet  # noqa: E402
from microseg_workbench.training.runs import RunDir  # noqa: E402
from microseg_workbench.training.trainer import (  # noqa: E402
    ResumeError, TrainingOutOfMemory, resume_training, run_training,
)

pytestmark = pytest.mark.training


@pytest.fixture
def ws_version(synthetic_folder, tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    return ws, create_dataset_version(report, SplitConfig(seed=4), ws.datasets_dir, "t")


def tiny(dataset_id, **kw):
    base = dict(epochs=4, batch_size=4, base_channels=4, depth=2, patches_per_image=2, seed=5, device="cpu",
                num_threads=1, fixed_samples=1)
    base.update(kw)
    return TrainConfig(dataset_id=dataset_id, **base)


def cancel_after(ws, run_id, epoch):
    def on_event(event):
        if event["type"] == "epoch" and event["epoch"] == epoch:
            RunDir(ws.runs_dir / run_id).request_cancel()
    return on_event


def test_resume_continues_exactly_like_uninterrupted_run(ws_version):
    ws, version = ws_version
    full = run_training(tiny(version.dataset_id), ws, run_id="full")
    part = run_training(tiny(version.dataset_id), ws, run_id="part", on_event=cancel_after(ws, "part", 2))
    assert part.state() == "cancelled" and part.resumable()
    resumed = resume_training(ws, "part")
    assert resumed.state() == "completed"
    hist_full = {e["epoch"]: e["train_loss"] for e in full.epoch_history()}
    hist_part = {e["epoch"]: e["train_loss"] for e in resumed.epoch_history()}
    assert hist_full == hist_part
    a = torch.load(full.checkpoints / "last.pt", weights_only=True)["model_state"]
    b = torch.load(resumed.checkpoints / "last.pt", weights_only=True)["model_state"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [e["type"] for e in resumed.read_events()].count("resume") == 1
    assert resumed.info()["resumes"][0]["from_epoch"] == 2
    with pytest.raises(ResumeError, match="already completed"):
        resume_training(ws, "part")


def test_resume_without_checkpoint_is_refused(ws_version):
    ws, version = ws_version

    def cancel_immediately(event):
        if event["type"] == "start":
            RunDir(ws.runs_dir / "early").request_cancel()

    run_training(tiny(version.dataset_id), ws, run_id="early", on_event=cancel_immediately)
    with pytest.raises(ResumeError, match="No checkpoint"):
        resume_training(ws, "early")


def test_dead_process_is_reported_as_interrupted(tmp_path):
    run = RunDir(tmp_path / "r").create()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    run.write_status(state="running", epoch=2, pid=dead.pid)
    assert run.state() == "interrupted"
    run.write_status(state="running", epoch=2)  # this (alive) process
    assert run.state() == "running"


def test_out_of_memory_gets_actionable_message(ws_version, monkeypatch):
    ws, version = ws_version

    def boom(self, x):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(UNet, "forward", boom)
    with pytest.raises(TrainingOutOfMemory, match="smaller batch size"):
        run_training(tiny(version.dataset_id), ws, run_id="oom")
    st = RunDir(ws.runs_dir / "oom").status()
    assert st["state"] == "failed" and st["error_type"] == "out_of_memory"


def test_probe_batch_size_halves_until_it_fits():
    tried = []

    def step(batch):
        tried.append(batch)
        if batch > 10:
            raise torch.OutOfMemoryError("CUDA out of memory")

    batch, attempts = probe_batch_size(step, 64)
    assert batch == 8 and tried == [64, 32, 16, 8]
    assert [a["ok"] for a in attempts] == [False, False, False, True]
    with pytest.raises(ValueError):
        probe_batch_size(lambda b: (_ for _ in ()).throw(ValueError("other")), 8)
    assert is_out_of_memory(MemoryError()) and not is_out_of_memory(ValueError("x"))


def test_nvidia_smi_parsing():
    text = "0, NVIDIA RTX A4000, 87, 5120, 16376, 64, 112.5\n1, Tesla T4, [N/A], 10, 15360, 40, [Not Supported]\n"
    gpus = monitor.parse_nvidia_smi(text)
    assert gpus[0]["utilization_percent"] == 87 and gpus[0]["memory_total_mb"] == 16376
    assert gpus[1]["utilization_percent"] is None and gpus[1]["power_w"] is None
    assert all(g["simulated"] is False for g in gpus)


def test_resource_events_real_and_simulated_gpu_flagged(ws_version, monkeypatch):
    ws, version = ws_version
    monkeypatch.setenv(monitor.SIMULATE_ENV, "1")
    monkeypatch.setattr(monitor, "_query_nvml", lambda: None)
    monkeypatch.setattr(monitor, "_query_nvidia_smi", lambda: None)
    run = run_training(tiny(version.dataset_id, epochs=1), ws)
    epoch = run.epoch_history()[0]["resources"]
    assert epoch["simulated"] is False  # CPU/RAM are measured
    assert epoch["simulated_gpu"] is True and epoch["gpus"][0]["simulated"] is True
    assert "process_rss_mb" in epoch


def test_auto_batch_and_precision_on_cpu(ws_version):
    ws, version = ws_version
    run = run_training(tiny(version.dataset_id, epochs=1, batch_size=0), ws)
    assert run.info()["resolved"]["batch_size"] == 8
    assert run.info()["resolved"]["mixed_precision"] is False
    with pytest.raises(ValueError, match="Mixed precision needs"):
        run_training(tiny(version.dataset_id, mixed_precision="on"), ws)


def test_presets_are_valid():
    for name in ("quick_cpu", "standard", "large_gpu"):
        cfg = preset_config(name, "x")
        assert cfg.augment.patch_size % 2 ** cfg.depth == 0
    assert preset_config("large_gpu", "x").augment.patch_size == 256


def test_lr_schedule_is_defined_for_any_length():
    from microseg_workbench.training.trainer import lr_schedule

    for total in (1, 2, 3, 10, 1000):
        f = lr_schedule(total)
        values = [f(i) for i in range(total)]
        assert all(0.0 < v <= 1.0 for v in values)
    f = lr_schedule(100)
    assert f(9) == pytest.approx(1.0) and f(99) < 0.1


def test_early_stopping_stops_and_still_evaluates(ws_version, monkeypatch):
    import json

    from microseg_workbench.training import trainer as trainer_mod

    ws, version = ws_version
    real_evaluate = trainer_mod.evaluate

    def flat_after_first(model, samples, device, names, prep=None, keep=None, on_prediction=None):
        metrics, loss, preds = real_evaluate(model, samples, device, names, prep, keep, on_prediction)
        if metrics is not None and getattr(samples, "records", None) and samples.records[0]["split"] == "val":
            metrics = dict(metrics, mean_iou_foreground=0.5)  # validation never improves after epoch 1
        return metrics, loss, preds

    monkeypatch.setattr(trainer_mod, "evaluate", flat_after_first)
    run = run_training(tiny(version.dataset_id, epochs=10, early_stopping_patience=2), ws)
    assert [e["epoch"] for e in run.epoch_history()] == [1, 2, 3]
    assert run.status()["stopped_early"] == 3 and "Stopped early" in run.status()["message"]
    assert json.loads((run.evaluation / "evaluation.json").read_text())["checkpoint_epoch"] == 1
