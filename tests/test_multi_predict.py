"""Several trained models on the same image."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from microseg_workbench.cli import main  # noqa: E402
from microseg_workbench.core.augment import AugmentConfig  # noqa: E402
from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.config import TrainConfig  # noqa: E402
from microseg_workbench.training.multi_predict import (  # noqa: E402
    NOT_ANALYSED, CompareError, compare_models_on_image, load_truth, render_comparison,
)
from microseg_workbench.training.trainer import run_training  # noqa: E402

pytestmark = pytest.mark.training


@pytest.fixture(scope="module")
def three_models(tmp_path_factory):
    from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset

    root = tmp_path_factory.mktemp("mp")
    write_synthetic_dataset(root / "data", SyntheticSpec(specimens=3, images_per_specimen=2, size=64, seed=2))
    ws = Workspace(root / "ws").ensure()
    v = create_dataset_version(validate_folder(root / "data", DEFAULT_GROUP_REGEX), SplitConfig(), ws.datasets_dir, "d")
    common = dict(dataset_id=v.dataset_id, epochs=2, base_channels=4, depth=2, patches_per_image=1, device="cpu",
                  augment=AugmentConfig(patch_size=32))
    runs = [run_training(TrainConfig(architecture="unet", **common), ws, run_id="unet"),
            run_training(TrainConfig(architecture="segformer", **common), ws, run_id="segformer"),
            run_training(TrainConfig(architecture="unet", exclude_border=(0, 8, 0, 0), **common), ws, run_id="border")]
    return root, ws, v, runs


def test_models_compared_on_one_image_with_ground_truth(three_models):
    _, _, v, runs = three_models
    s = v.samples("test")[0]
    truth = load_truth(v.mask_path(s), v.class_map)
    result = compare_models_on_image(runs, v.image_path(s), truth, v.class_map)
    assert [m["run_id"] for m in result["models"]] == ["unet", "segformer", "border"]
    assert all(m["metrics"] and m["metrics"]["evaluated_pixels"] == 56 * 64 for m in result["models"])
    border = result["models"][2]["values"]
    assert (border[-8:] == NOT_ANALYSED).all() and not result["analysed_by_all"][-8:].any()
    assert result["agreement"][0][0] == 1.0 and 0 <= result["agreement"][0][1] <= 1
    assert result["disagreement"].shape == truth.shape and result["same_classes"]
    assert all(abs(sum(m["fractions"].values()) - 1) < 1e-6 for m in result["models"])
    tiles = render_comparison(result, truth)
    assert [t[0].split("\n")[0] for t in tiles][:2] == ["image", "ground truth"] and "disagree" in tiles[-1][0]


def test_any_image_without_truth_and_clear_errors(three_models, tmp_path):
    _, _, v, runs = three_models
    rgb = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((40, 50, 3), np.uint8) + np.array([1, 2, 3], np.uint8)).save(rgb)
    with pytest.raises(CompareError, match="expects 1-channel"):
        compare_models_on_image(runs[:1], rgb)
    gray = tmp_path / "g.png"
    Image.fromarray(np.full((40, 50), 128, np.uint8)).save(gray)
    result = compare_models_on_image(runs[:2], gray)
    assert result["models"][0]["values"].shape == (40, 50) and "metrics" not in result["models"][0]
    with pytest.raises(CompareError, match="at least one"):
        compare_models_on_image([], gray)


def test_cli_compare_image(three_models, tmp_path, capsys):
    root, ws, v, _ = three_models
    s = v.samples("test")[0]
    out = tmp_path / "side.png"
    assert main(["--workspace", str(ws.root), "compare-image", str(v.image_path(s)), "unet", "segformer",
                 "--mask", str(v.mask_path(s)), "--dataset", v.dataset_id, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "feature IoU" in text and "agreement" in text and out.stat().st_size > 0
