"""Representative user workflow through the CLI entry point."""

import json

import pytest

from microseg_workbench.cli import main

pytest.importorskip("torch")


def test_cli_end_to_end(tmp_path, capsys):
    ws = str(tmp_path / "ws")
    data = tmp_path / "data"
    assert main(["--workspace", ws, "demo-data", "--out", str(data), "--specimens", "4", "--per-specimen", "2",
                 "--size", "64"]) == 0
    assert main(["--workspace", ws, "validate", str(data)]) == 0
    assert main(["--workspace", ws, "prepare", str(data), "--name", "cli"]) == 0
    out = capsys.readouterr().out
    dataset_id = next(line.split(": ")[1] for line in out.splitlines() if line.startswith("Dataset version"))
    assert main(["--workspace", ws, "preview-aug", dataset_id, "--out", str(tmp_path / "p.png")]) == 0
    assert (tmp_path / "p.png").stat().st_size > 0
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"epochs": 2, "base_channels": 4, "depth": 2, "patches_per_image": 1,
                               "device": "cpu", "fixed_samples": 1}))
    assert main(["--workspace", ws, "train", dataset_id, "--config", str(cfg), "--run-id", "r1"]) == 0
    assert main(["--workspace", ws, "evaluate", "r1"]) == 0
    assert main(["--workspace", ws, "runs"]) == 0
    out = capsys.readouterr().out
    assert "UNREVIEWED" in out and "r1  completed" in out


def test_cli_refuses_invalid_folder(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["--workspace", str(tmp_path / "ws"), "prepare", str(tmp_path / "empty"), "--name", "x"]) == 1
    assert "class_map" in capsys.readouterr().out
