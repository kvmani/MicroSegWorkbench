"""Run comparison, prediction on new images and tiled inference."""

import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from microseg_workbench.cli import main  # noqa: E402
from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.compare import compare_runs, comparison_csv  # noqa: E402
from microseg_workbench.training.config import TrainConfig  # noqa: E402
from microseg_workbench.training.engine import load_checkpoint, predict_probabilities  # noqa: E402
from microseg_workbench.training.predict import PredictionError, predict_folder  # noqa: E402
from microseg_workbench.training.trainer import run_training  # noqa: E402

pytestmark = pytest.mark.training


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset

    root = tmp_path_factory.mktemp("cp")
    write_synthetic_dataset(root / "data", SyntheticSpec(specimens=4, images_per_specimen=2, size=64, seed=1))
    ws = Workspace(root / "ws").ensure()
    report = validate_folder(root / "data", DEFAULT_GROUP_REGEX)
    v1 = create_dataset_version(report, SplitConfig(seed=1), ws.datasets_dir, "a")
    v2 = create_dataset_version(report, SplitConfig(seed=2), ws.datasets_dir, "a")
    cfg = dict(epochs=2, batch_size=4, base_channels=4, depth=2, patches_per_image=1, device="cpu", fixed_samples=1)
    r1 = run_training(TrainConfig(dataset_id=v1.dataset_id, seed=1, **cfg), ws, run_id="r1")
    r2 = run_training(TrainConfig(dataset_id=v1.dataset_id, seed=2, learning_rate=1e-3, **cfg), ws, run_id="r2")
    r3 = run_training(TrainConfig(dataset_id=v2.dataset_id, seed=1, **cfg), ws, run_id="r3")
    return root, ws, (r1, r2, r3)


def test_compare_same_dataset_lists_differences(trained):
    _, _, (r1, r2, _) = trained
    c = compare_runs([r1, r2])
    assert c["comparable"] and not c["warnings"]
    assert set(c["differences"]) == {"seed", "learning_rate"}
    assert c["runs"][0]["test_miou"] is not None
    assert c["runs"][0]["epochs_done"] == 2
    csv_text = comparison_csv(c)
    assert csv_text.splitlines()[0].startswith("run_id,") and "test_iou_hydride" in csv_text


def test_compare_different_datasets_warns(trained):
    _, _, (r1, _, r3) = trained
    c = compare_runs([r1, r3])
    assert not c["comparable"]
    assert any("different test sets" in w for w in c["warnings"])


def test_predict_folder_writes_class_values_and_is_atomic(trained, tmp_path):
    root, ws, (r1, _, _) = trained
    new = tmp_path / "new"
    new.mkdir()
    for p in sorted((root / "data").glob("S01_*.png"))[:2]:
        if not p.stem.endswith("_mask"):
            (new / p.name).write_bytes(p.read_bytes())
    Image.fromarray(np.zeros((40, 40, 3), np.uint8) + np.array([1, 2, 3], np.uint8)).save(new / "rgb.png")
    report = predict_folder(r1, new, tmp_path / "out", device="cpu")
    assert report["review_status"] == "unreviewed"
    assert [e["file"] for e in report["errors"]] == ["rgb.png"]  # channel mismatch reported, not guessed
    stem = report["images"][0]["file"].rsplit(".", 1)[0]
    mask = np.asarray(Image.open(tmp_path / "out" / f"{stem}_pred_mask.png"))
    assert set(np.unique(mask)) <= {0, 1, 2}
    assert json.loads((tmp_path / "out" / "predictions.json").read_text())["checkpoint_sha256"]
    with pytest.raises(PredictionError, match="already exists"):
        predict_folder(r1, new, tmp_path / "out", device="cpu")
    assert not any(p.name.startswith(".staging") for p in tmp_path.iterdir())


def test_tiled_inference_matches_whole_image(trained):
    _, _, (r1, _, _) = trained
    model, _ = load_checkpoint(r1.checkpoints / "best.pt")
    image = np.random.default_rng(0).random((150, 170, 1), dtype=np.float32)
    whole = predict_probabilities(model, image, torch.device("cpu"), tile=4096)
    tiled = predict_probabilities(model, image, torch.device("cpu"), tile=96, overlap=32)
    assert tiled.shape == whole.shape
    assert torch.allclose(tiled.sum(0), torch.ones(150, 170), atol=1e-5)
    agree = (tiled.argmax(0) == whole.argmax(0)).float().mean().item()
    assert agree > 0.9  # tiles see less context at their borders; overlap averaging keeps them close


def test_cli_compare_and_predict(trained, tmp_path, capsys):
    root, ws, _ = trained
    assert main(["--workspace", str(ws.root), "compare", "r1", "r3", "--csv", str(tmp_path / "c.csv")]) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and (tmp_path / "c.csv").exists()
    assert main(["--workspace", str(ws.root), "predict", "r1", str(root / "data"), "--out", str(tmp_path / "p"),
                 "--device", "cpu"]) == 0
    assert main(["--workspace", str(ws.root), "compare", "nope"]) == 1


def test_seed_variability_summary(trained):
    from microseg_workbench.training.compare import seed_variability

    _, ws, (r1, r2, r3) = trained
    assert compare_runs([r1, r2])["seed_summary"] is None  # lr differs too
    fake = [{"seed": i, "test_miou": v, "test_miou_fg": v, "test_pixel_accuracy": 1.0,
             "test_class_iou": {"a": v, "b": None}} for i, v in enumerate([0.5, 0.7, 0.9])]
    s = seed_variability(fake)
    assert s["test_miou"]["mean"] == pytest.approx(0.7) and s["test_miou"]["sd"] == pytest.approx(0.2)
    assert s["test_class_iou"]["b"] is None and s["test_class_iou"]["a"]["n"] == 3


def test_cli_repeat_seeds(trained, tmp_path, capsys):
    _, ws, (r1, _, _) = trained
    cfg = tmp_path / "c.json"
    cfg.write_text('{"epochs": 1, "base_channels": 4, "patches_per_image": 1, "device": "cpu"}')
    dataset = r1.info()["dataset"]["dataset_id"]
    assert main(["--workspace", str(ws.root), "train", dataset, "--config", str(cfg), "--repeat-seeds", "3",
                 "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "Over 3 seeds" in out and "+/-" in out


def test_html_report_is_self_contained(trained, tmp_path):
    from microseg_workbench.training.report import write_report

    _, ws, (r1, _, _) = trained
    path = write_report(r1)
    page = path.read_text(encoding="utf-8")
    assert "<script" not in page and "http://" not in page and "https://" not in page
    assert "data:image/png;base64," in page and "<svg" in page
    assert "Review status: unreviewed" in page and r1.info()["dataset"]["dataset_id"] in page
    assert main(["--workspace", str(ws.root), "report", r1.run_id]) == 0
