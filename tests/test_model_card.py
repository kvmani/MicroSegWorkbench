"""Observations, the model card and provenance carried into predictions."""

import json

import pytest

torch = pytest.importorskip("torch")

from microseg_workbench.cli import main  # noqa: E402
from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.config import TrainConfig  # noqa: E402
from microseg_workbench.training.model_card import NoteError, add_note, read_notes  # noqa: E402
from microseg_workbench.training.predict import predict_folder  # noqa: E402
from microseg_workbench.training.trainer import run_training  # noqa: E402

pytestmark = pytest.mark.training


@pytest.fixture
def trained(synthetic_folder, tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    v = create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX), SplitConfig(seed=4),
                               ws.datasets_dir, "t")
    run = run_training(TrainConfig(dataset_id=v.dataset_id, epochs=2, base_channels=4, depth=2, patches_per_image=1,
                                   device="cpu"), ws, run_id="r")
    return ws, v, run


def test_model_card_has_everything_needed_to_trust_and_reuse(trained):
    ws, v, run = trained
    card = json.loads((run.root / "model_card.json").read_text())
    assert card["schema"] == "microseg-workbench.model-card/1" and card["status"] == "completed"
    assert card["network"]["architecture"] == "unet" and card["network"]["parameters"] > 0
    assert [c["name"] for c in card["classes"]["classes"]] == ["matrix", "hydride", "pore"]
    assert card["input"]["normalization"] == "dtype_range" and card["input"]["channels"] == 1
    assert card["training_data"]["dataset_id"] == v.dataset_id and card["training_data"]["manifest_sha256"]
    assert card["training"]["environment"]["torch"] == torch.__version__ and card["training"]["config"]["seed"]
    assert card["results"]["split"] == "test" and card["results"]["per_class"]
    assert card["diagnostics"] and card["checkpoint"]["sha256"] and "msw predict r" in card["how_to_use"]
    assert card["review_status"] == "unreviewed"


def test_notes_are_append_only_and_update_the_card(trained):
    _, _, run = trained
    with pytest.raises(NoteError):
        add_note(run, "   ", "alice")
    add_note(run, "Pores look slightly too small.", "alice")
    add_note(run, "Correction: only in S03.", "bob")
    notes = read_notes(run)
    assert [n["author"] for n in notes] == ["alice", "bob"] and all(n["time"] for n in notes)
    card = json.loads((run.root / "model_card.json").read_text())
    assert [o["text"] for o in card["observations"]] == [n["text"] for n in notes]


def test_predictions_name_the_model_card(trained, synthetic_folder, tmp_path):
    _, _, run = trained
    out = predict_folder(run, synthetic_folder, tmp_path / "pred", device="cpu")
    assert out["model_card_sha256"] and out["architecture"] == "unet" and out["checkpoint_sha256"]


def test_cli_note_and_card(trained, capsys):
    ws, _, _ = trained
    assert main(["--workspace", str(ws.root), "note", "r", "Good on hydrides"]) == 0
    assert "Good on hydrides" in capsys.readouterr().out
    assert main(["--workspace", str(ws.root), "card", "r"]) == 0
    assert "microseg-workbench.model-card/1" in capsys.readouterr().out
