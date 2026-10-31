import json

import numpy as np
import pytest
from PIL import Image

from microseg_workbench.core.preprocess import (
    Preprocess, PreprocessError, normalize, resize_image, resize_labels,
)


def test_standardize_and_constant_image():
    img = np.random.default_rng(0).random((20, 30, 1), dtype=np.float32)
    z = normalize(img, "per_image_standardize")
    assert abs(float(z.mean())) < 1e-5 and abs(float(z.std()) - 1) < 1e-3
    flat = normalize(np.full((5, 5, 1), 0.3, np.float32), "per_image_standardize")
    assert np.isfinite(flat).all() and float(np.abs(flat).max()) == 0.0
    assert normalize(img, "dtype_range") is img


def test_resize_keeps_class_values_and_shape():
    labels = np.random.default_rng(1).integers(0, 3, (37, 51)).astype(np.uint8)
    small = resize_labels(labels, Preprocess(0.5).scaled_shape(37, 51))
    assert small.shape == (18, 26) and set(np.unique(small)) <= {0, 1, 2}
    img = np.random.default_rng(1).random((37, 51, 1), dtype=np.float32)
    assert resize_image(img, (18, 26)).shape == (18, 26, 1)
    assert resize_image(img, (37, 51)) is img
    with pytest.raises(PreprocessError):
        Preprocess(resize_factor=0)
    with pytest.raises(PreprocessError):
        Preprocess(normalization="magic")


def test_old_checkpoint_metadata_defaults():
    assert Preprocess.from_meta({"normalization": "dtype_range"}) == Preprocess()


@pytest.mark.training
def test_resized_standardized_training_scores_at_original_resolution(synthetic_folder, tmp_path):
    pytest.importorskip("torch")
    from microseg_workbench.core.dataset import create_dataset_version, validate_folder
    from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
    from microseg_workbench.core.workspace import Workspace
    from microseg_workbench.training.config import TrainConfig
    from microseg_workbench.training.predict import predict_folder
    from microseg_workbench.training.trainer import run_training

    ws = Workspace(tmp_path / "ws").ensure()
    version = create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX), SplitConfig(seed=4),
                                     ws.datasets_dir, "t")
    from microseg_workbench.core.augment import AugmentConfig

    cfg = TrainConfig(dataset_id=version.dataset_id, epochs=3, batch_size=4, base_channels=4, depth=2,
                      patches_per_image=2, device="cpu", resize_factor=0.5, normalization="per_image_standardize",
                      augment=AugmentConfig(patch_size=24))
    run = run_training(cfg, ws)
    report = json.loads((run.evaluation / "evaluation.json").read_text())
    n_test = len(version.samples("test"))
    assert report["metrics"]["evaluated_pixels"] == n_test * 64 * 64  # original resolution
    assert report["preprocess"] == {"resize_factor": 0.5, "normalization": "per_image_standardize"}
    stem = report["images"][0]
    assert np.asarray(Image.open(run.evaluation / f"{stem}_pred_mask.png")).shape == (64, 64)
    epochs = run.epoch_history()
    assert epochs[-1]["train_loss"] < epochs[0]["train_loss"]
    out = predict_folder(run, synthetic_folder, tmp_path / "pred", device="cpu")
    assert out["images"][0]["height"] == 64 and out["preprocess"]["resize_factor"] == 0.5
