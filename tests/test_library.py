"""Fetching and running trained models: the model library, auto-download and input checks."""

import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from microseg_workbench.cli import main  # noqa: E402
from microseg_workbench.collab.settings import CollabSettings  # noqa: E402
from microseg_workbench.collab.share import SharedStore  # noqa: E402
from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.config import TrainConfig  # noqa: E402
from microseg_workbench.training.library import list_models  # noqa: E402
from microseg_workbench.training.model_card import add_note  # noqa: E402
from microseg_workbench.training.multi_predict import compare_models_on_image  # noqa: E402
from microseg_workbench.training.predict import predict_folder  # noqa: E402
from microseg_workbench.training.runs import RunDir  # noqa: E402
from microseg_workbench.training.trainer import input_warnings, run_training  # noqa: E402

pytestmark = pytest.mark.training


@pytest.fixture
def shared(synthetic_folder, tmp_path):
    """Alice trains and publishes run 'pub'; she also keeps an unpublished run 'mine'. Bob has an empty workspace."""
    store = SharedStore.create(tmp_path / "share")
    wa = Workspace(tmp_path / "wa").ensure()
    wb = Workspace(tmp_path / "wb").ensure()
    v = create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX), SplitConfig(seed=4),
                               wa.datasets_dir, "t")
    cfg = TrainConfig(dataset_id=v.dataset_id, epochs=1, base_channels=4, depth=2, patches_per_image=1, device="cpu")
    pub = run_training(cfg, wa, run_id="pub")
    run_training(cfg, wa, run_id="mine")
    store.publish_dataset(v, user="alice")
    add_note(pub, "boundaries a bit thick", author="alice")
    store.publish_run(pub, user="alice")
    store.add_review("pub", "approved", "ok", reviewer="carol")
    for ws in (wa, wb):
        CollabSettings(share_path=str(store.root)).save(ws)
    return store, wa, wb, v


def test_training_records_what_the_images_looked_like(shared):
    _, wa, _, _ = shared
    run = RunDir(wa.runs_dir / "pub")
    profile = run.info()["input_profile"]
    assert profile["dtypes"] == ["uint8"] and profile["images_measured"] > 0
    b = profile["mean_brightness"]
    assert 0 <= b["min"] <= b["mean"] <= b["max"] <= 1
    card = json.loads((run.root / "model_card.json").read_text())
    assert card["input"]["training_images"] == profile


def test_library_lists_local_and_shared_models_approved_first(shared):
    store, wa, wb, v = shared
    rows = {r["run_id"]: r for r in list_models(wa, store)}
    assert rows["pub"]["local"] and rows["pub"]["shared"] and rows["pub"]["review"] == "approved"
    assert rows["mine"]["local"] and not rows["mine"]["shared"] and rows["mine"]["review"] == "not published"
    assert [r["run_id"] for r in list_models(wa, store)][0] == "pub"
    bob = list_models(wb, store)
    assert [r["run_id"] for r in bob] == ["pub"]
    assert not bob[0]["local"] and bob[0]["network"] == "U-Net (convolutional)"
    assert bob[0]["dataset_id"] == v.dataset_id and bob[0]["test_miou_fg"] is not None
    assert list_models(wb) == []  # no share configured: nothing


def test_cli_models_and_predict_downloads_a_shared_model(shared, synthetic_folder, tmp_path, capsys):
    _, _, wb, _ = shared
    assert main(["--workspace", str(wb.root), "models"]) == 0
    out = capsys.readouterr().out
    assert "pub" in out and "[APPROVED]" in out and "shared only" in out
    out_dir = tmp_path / "pred"
    assert main(["--workspace", str(wb.root), "predict", "pub", str(synthetic_folder), "--out", str(out_dir),
                 "--device", "cpu"]) == 0
    assert "Downloading model pub" in capsys.readouterr().out
    assert (wb.runs_dir / "pub" / "checkpoints" / "best.pt").exists()
    assert RunDir(wb.runs_dir / "pub").info()["imported_from_share"]["published_by"] == "alice"
    report = json.loads((out_dir / "predictions.json").read_text())
    assert report["images"] and report["warnings"] == []  # same kind of images as training: no warnings
    assert report["model_card_sha256"]
    assert main(["--workspace", str(wb.root), "predict", "nope", str(synthetic_folder), "--out",
                 str(tmp_path / "x")]) == 1


def test_predict_without_share_explains(tmp_path, synthetic_folder, capsys):
    ws = Workspace(tmp_path / "lonely").ensure()
    assert main(["--workspace", str(ws.root), "predict", "r", str(synthetic_folder), "--out",
                 str(tmp_path / "p")]) == 1
    assert "no shared folder is configured" in capsys.readouterr().err


def test_input_warnings_for_unlike_images():
    profile = {"dtypes": ["uint8"], "mean_brightness": {"min": 0.3, "max": 0.5, "mean": 0.4, "std": 0.05}}
    ok = np.full((8, 8), 100, np.uint8)
    assert input_warnings(profile, ok, ok / 255.0) == []
    bright = np.full((8, 8), 250, np.uint8)
    assert "brighter" in input_warnings(profile, bright, bright / 255.0)[0]
    deep = np.full((8, 8), 30000, np.uint16)
    notes = input_warnings(profile, deep, deep / 65535.0)
    assert len(notes) == 1 and "uint16" in notes[0]
    assert input_warnings(None, bright, bright / 255.0) == []  # older runs: no profile, no warnings


def test_predict_and_compare_warn_about_unlike_images(shared, synthetic_folder, tmp_path, capsys):
    _, wa, _, _ = shared
    src = next(p for p in sorted(synthetic_folder.glob("*.png")) if not p.stem.endswith("_mask"))
    arr = np.asarray(Image.open(src))
    folder = tmp_path / "new"
    folder.mkdir()
    Image.fromarray((arr.astype(np.uint16) * 257)).save(folder / "deep.png")
    Image.fromarray(np.full_like(arr, 252)).save(folder / "bright.png")
    report = predict_folder(RunDir(wa.runs_dir / "pub"), folder, tmp_path / "out", device="cpu")
    files = {w["file"] for w in report["warnings"]}
    assert files == {"deep.png", "bright.png"} and len(report["images"]) == 2  # warned, still predicted
    result = compare_models_on_image([RunDir(wa.runs_dir / "pub")], folder / "bright.png")
    assert result["models"][0]["input_warnings"]
    capsys.readouterr()
    assert main(["--workspace", str(wa.root), "predict", "pub", str(folder), "--out", str(tmp_path / "o2"),
                 "--device", "cpu"]) == 0
    assert "WARNING bright.png" in capsys.readouterr().out
