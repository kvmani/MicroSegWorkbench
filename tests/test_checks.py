"""Beginner cross-checks: dataset health before training and run diagnostics after."""

import json
from pathlib import Path

import pytest

from microseg_workbench.core.dataset import DatasetVersion
from microseg_workbench.core.health import dataset_health
from microseg_workbench.training.diagnostics import diagnose_run
from microseg_workbench.training.runs import RunDir

CLASSES = {"schema": "microseg-workbench.classes/1", "mask_encoding": "indexed",
           "classes": [{"value": 0, "name": "matrix", "color": "#000000"},
                       {"value": 1, "name": "hydride", "color": "#e6194b"},
                       {"value": 2, "name": "pore", "color": "#4363d8"}]}


def version(samples, **extra):
    manifest = {"class_map": CLASSES, "samples": samples, **extra}
    return DatasetVersion(Path("/nonexistent"), manifest)


def sample(split, group, px):
    return {"split": split, "group": group, "class_pixels": {str(k): v for k, v in px.items()}}


def good_samples():
    out = []
    for g, split in [("A", "train"), ("B", "train"), ("C", "train"), ("D", "val"), ("E", "test"), ("F", "test")]:
        out += [sample(split, g, {0: 900, 1: 80, 2: 20}) for _ in range(2)]
    return out


def levels(findings):
    return {lv for lv, _ in findings}


def test_healthy_dataset_is_ok():
    f = dataset_health(version(good_samples()))
    assert f[0][0] == "ok" and "block" not in levels(f) and "warn" not in levels(f)


def test_class_missing_from_training_blocks():
    s = [x if x["split"] != "train" else sample("train", x["group"], {0: 980, 1: 20, 2: 0}) for x in good_samples()]
    f = dataset_health(version(s))
    assert any(lv == "block" and "'pore'" in m and "training" in m for lv, m in f)


def test_inverted_masks_tiny_splits_single_group_and_unapproved():
    s = [sample("train", "A", {0: 100, 1: 900, 2: 5}), sample("val", "B", {0: 100, 1: 900, 2: 5}),
         sample("test", "C", {0: 100, 1: 900, 2: 5})]
    f = dataset_health(version(s, contains_unapproved_annotations=True))
    text = " ".join(m for _, m in f)
    assert "inverted" in text and "Only 1 training image" in text and "single image" in text
    assert "one group (C)" in text and "not approved" in text


def make_run(tmp_path, epochs, per_class=None, status=None):
    run = RunDir(tmp_path / "r").create()
    run.write_status(**(status or {"state": "completed"}))
    for e in epochs:
        run.append_event({"type": "epoch", **e})
    if per_class is not None:
        (run.evaluation / "evaluation.json").write_text(json.dumps({"metrics": {
            "mean_iou_foreground": per_class.pop("_fg", 0.8), "per_class": per_class["rows"]},
            "split": "test", "ground_truth": "as in dataset version"}))
    return run


def ep(i, train, val, iou):
    return {"epoch": i, "train_loss": train, "val_loss": val, "val_mean_iou_foreground": iou}


def test_overfitting_is_explained(tmp_path):
    epochs = [ep(1, 1.0, 0.8, 0.3), ep(2, 0.6, 0.5, 0.6), ep(3, 0.4, 0.4, 0.7), ep(4, 0.3, 0.45, 0.7),
              ep(5, 0.2, 0.55, 0.68), ep(6, 0.1, 0.7, 0.66)]
    f = diagnose_run(make_run(tmp_path, epochs))
    assert any(lv == "warn" and "Overfitting" in m and "epoch 3" in m for lv, m in f)


def test_still_improving_dead_class_and_val_test_gap(tmp_path):
    epochs = [ep(i, 1.0 / i, 1.0 / i, 0.2 + 0.1 * i) for i in range(1, 7)]
    rows = [{"name": "matrix", "iou": 0.99, "true_pixels": 9000},
            {"name": "hydride", "iou": 0.6, "true_pixels": 800},
            {"name": "pore", "iou": 0.0, "true_pixels": 300}]
    f = diagnose_run(make_run(tmp_path, epochs, {"rows": rows, "_fg": 0.3}))
    text = " ".join(m for _, m in f)
    assert "Still improving" in text and "never finds class 'pore'" in text and "much lower" in text
    assert "ok" not in levels(f)


@pytest.mark.parametrize("split", ["val", "unknown"])
def test_diagnosis_does_not_call_non_test_scores_test_results(tmp_path, split):
    run = make_run(tmp_path, [ep(i, 1 / i, 1 / i, 0.8) for i in range(1, 5)])
    (run.evaluation / "evaluation.json").write_text(json.dumps({
        "split": split, "metrics": {"mean_iou_foreground": 0.7, "per_class": [
            {"name": "matrix", "true_pixels": 9000, "iou": 0.9},
            {"name": "hydride", "true_pixels": 300, "iou": 0.6}]}}))
    findings = diagnose_run(run)
    assert "ok" not in levels(findings)
    assert any(lv == "warn" and "independent test" in message for lv, message in findings)
    assert not any("test pixels" in message or "healthy (test" in message for _, message in findings)


def test_overfitting_does_not_claim_best_iou_checkpoint_is_at_lowest_loss(tmp_path):
    history = [ep(1, 1.0, 0.5, 0.3), ep(2, 0.7, 0.3, 0.4), ep(3, 0.5, 0.4, 0.5),
               ep(4, 0.3, 0.5, 0.6), ep(5, 0.2, 0.6, 0.7)]
    findings = diagnose_run(make_run(tmp_path, history))
    warning = next(m for lv, m in findings if lv == "warn" and "Overfitting" in m)
    assert "foreground IoU" in warning
    assert "from before" not in warning


@pytest.mark.training
def test_training_records_health_and_diagnostics(synthetic_folder, tmp_path, capsys):
    pytest.importorskip("torch")
    from microseg_workbench.cli import main

    ws = str(tmp_path / "ws")
    assert main(["--workspace", ws, "prepare", str(synthetic_folder), "--name", "d"]) == 0
    out = capsys.readouterr().out
    assert "No problems found in the dataset" in out or "INFO" in out
    dataset = next((tmp_path / "ws" / "datasets").iterdir()).name
    cfg = tmp_path / "c.json"
    cfg.write_text('{"epochs": 2, "base_channels": 4, "patches_per_image": 1, "device": "cpu"}')
    assert main(["--workspace", ws, "train", dataset, "--config", str(cfg), "--run-id", "r"]) == 0
    assert "Run check:" in capsys.readouterr().out
    run = RunDir(tmp_path / "ws" / "runs" / "r")
    assert json.loads((run.evaluation / "diagnostics.json").read_text())["findings"]
    assert any(e["type"] == "dataset_health" for e in run.read_events())
    assert main(["--workspace", ws, "check-run", "r"]) == 0
    from microseg_workbench.training.report import build_report

    assert "Run check" in build_report(run)
