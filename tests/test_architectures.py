"""Plug-and-play networks: registry, transformer training, checkpoints, plugins and clear errors."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from microseg_workbench.core.augment import AugmentConfig  # noqa: E402
from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.architectures import (  # noqa: E402
    ARCHITECTURES, UnknownArchitecture, build_model,
)
from microseg_workbench.training.compare import compare_runs  # noqa: E402
from microseg_workbench.training.config import TrainConfig  # noqa: E402
from microseg_workbench.training.engine import load_checkpoint, predict_labels  # noqa: E402
from microseg_workbench.training.model import ModelConfig  # noqa: E402
from microseg_workbench.training.trainer import run_training  # noqa: E402

pytestmark = pytest.mark.training


@pytest.fixture
def ws_version(synthetic_folder, tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    return ws, create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX), SplitConfig(seed=4),
                                      ws.datasets_dir, "t")


@pytest.mark.parametrize("arch,depth", [("unet", 2), ("segformer", 2), ("segformer", 4)])
def test_every_builtin_network_maps_input_to_class_logits(arch, depth):
    model = build_model(ModelConfig(1, 3, 8, depth, arch))
    x = torch.zeros(2, 1, 64, 64)
    assert model(x).shape == (2, 3, 64, 64) and 64 % model.multiple == 0
    assert model.cfg.architecture == arch


def test_transformer_trains_reproducibly_and_is_comparable(ws_version):
    ws, version = ws_version
    common = dict(dataset_id=version.dataset_id, epochs=3, batch_size=4, base_channels=8, depth=2,
                  patches_per_image=2, device="cpu", num_threads=1, seed=3, augment=AugmentConfig(patch_size=32))
    seg_a = run_training(TrainConfig(architecture="segformer", **common), ws)
    seg_b = run_training(TrainConfig(architecture="segformer", **common), ws)
    unet = run_training(TrainConfig(architecture="unet", **common), ws)
    losses = lambda r: [e["loss"] for e in r.read_events() if e["type"] == "step"]  # noqa: E731
    assert losses(seg_a) == losses(seg_b)
    hist = seg_a.epoch_history()
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert seg_a.info()["model"]["architecture"] == "segformer"
    assert seg_a.info()["model"]["architecture_title"].startswith("SegFormer")
    c = compare_runs([seg_a, unet])
    assert c["differences"]["architecture"] == ["segformer", "unet"] and c["comparable"]
    model, meta = load_checkpoint(seg_a.checkpoints / "best.pt")
    assert type(model).__name__ == "SegFormerLite"
    assert predict_labels(model, np.zeros((37, 51, 1), np.float32), torch.device("cpu")).shape == (37, 51)


def test_patch_size_must_suit_the_network(ws_version):
    ws, version = ws_version
    cfg = TrainConfig(dataset_id=version.dataset_id, architecture="segformer", depth=4, epochs=1, device="cpu",
                      augment=AugmentConfig(patch_size=48))
    with pytest.raises(ValueError, match="multiple of 32"):
        run_training(cfg, ws)


def test_unknown_network_and_plugin_network(tmp_path, monkeypatch, ws_version):
    from microseg_workbench import plugins

    with pytest.raises(UnknownArchitecture, match="not available"):
        build_model(ModelConfig(1, 2, 8, 2, "doesnotexist"))
    plugin = tmp_path / "tiny_net.py"
    plugin.write_text(
        "import torch\n"
        "from microseg_workbench.training.architectures import register_architecture\n"
        "def build(cfg):\n"
        "    return torch.nn.Sequential(torch.nn.Conv2d(cfg.in_channels, cfg.base_channels, 3, padding=1),\n"
        "        torch.nn.ReLU(), torch.nn.Conv2d(cfg.base_channels, cfg.num_classes, 1))\n"
        "register_architecture('tinyconv', 'Tiny conv net', 'two convolutions', build, lambda cfg: 1)\n")
    monkeypatch.setenv(plugins.ENV, str(plugin))
    monkeypatch.setattr(plugins, "_loaded", {})
    try:
        model = build_model(ModelConfig(1, 3, 4, 1, "tinyconv"))  # loads the plugin on demand
        assert model(torch.zeros(1, 1, 10, 10)).shape == (1, 3, 10, 10)
        ws, version = ws_version
        run = run_training(TrainConfig(dataset_id=version.dataset_id, architecture="tinyconv", epochs=1,
                                       base_channels=4, depth=1, device="cpu", patches_per_image=1,
                                       augment=AugmentConfig(patch_size=32)), ws)
        info = json.loads(run.run_json.read_text())
        assert info["model"]["architecture_module"] == "msw_plugin_tiny_net"
        assert str(plugin) in info["plugins"]["loaded_plugins"]
    finally:
        ARCHITECTURES.pop("tinyconv", None)


def test_cli_lists_networks(capsys):
    from microseg_workbench.cli import main

    assert main(["architectures"]) == 0
    out = capsys.readouterr().out
    assert "unet" in out and "segformer" in out
