import numpy as np
import pytest
from PIL import Image

from microseg_workbench.core.augment import IGNORE_INDEX, AugmentConfig, AugmentConfigError, augment


def _pair(h=40, w=48):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=(h, w)).astype(np.uint8)
    image = (labels.astype(np.float32) / 4.0)[:, :, None]  # image encodes the label exactly
    return image, labels


@pytest.mark.parametrize("seed", range(12))
def test_geometric_transforms_keep_alignment(seed):
    image, labels = _pair()
    cfg = AugmentConfig(patch_size=32, rotate_prob=1.0, rotate_max_deg=30, photometric_prob=0.0)
    img2, lab2, _ = augment(image, labels, cfg, np.random.default_rng(seed),
                            image_resample=Image.Resampling.NEAREST)
    assert img2.shape[:2] == lab2.shape == (32, 32)
    valid = lab2 != IGNORE_INDEX
    np.testing.assert_array_equal(img2[..., 0][valid], lab2[valid].astype(np.float32) / 4.0)
    assert set(np.unique(lab2)) <= {0, 1, 2, IGNORE_INDEX}


def test_photometric_changes_image_only_and_is_reproducible():
    image, labels = _pair()
    cfg = AugmentConfig(patch_size=32, hflip=0, vflip=0, rot90=0, photometric_prob=1.0,
                        brightness=0.2, contrast=0.3, gamma=0.3, noise_std=0.05)
    a = augment(image, labels, cfg, np.random.default_rng(5))
    b = augment(image, labels, cfg, np.random.default_rng(5))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[0].min() >= 0 and a[0].max() <= 1
    assert a[2] == b[2]


def test_small_image_padding_is_ignored_not_background():
    image, labels = _pair(20, 20)
    cfg = AugmentConfig(patch_size=32, hflip=0, vflip=0, rot90=0, photometric_prob=0)
    _, lab, _ = augment(image, labels, cfg, np.random.default_rng(0))
    assert (lab == IGNORE_INDEX).sum() == 32 * 32 - 20 * 20


def test_invalid_config_messages():
    with pytest.raises(AugmentConfigError, match="probability"):
        AugmentConfig(hflip=1.5)
    with pytest.raises(AugmentConfigError, match="multiple of 8"):
        AugmentConfig(patch_size=30)
    with pytest.raises(AugmentConfigError, match="Unknown"):
        AugmentConfig.from_dict({"flip": 1})


def test_blur_preserves_shape_mean_and_labels():
    from microseg_workbench.core.augment import gaussian_blur

    image, labels = _pair()
    blurred = gaussian_blur(image, 1.5)
    assert blurred.shape == image.shape and abs(float(blurred.mean() - image.mean())) < 0.01
    assert float(blurred.std()) < float(image.std())
    assert gaussian_blur(image, 0) is image
    cfg = AugmentConfig(patch_size=32, hflip=0, vflip=0, rot90=0, photometric_prob=1.0, brightness=0, contrast=0,
                        blur_max_sigma=2.0)
    _, lab, ops = augment(image, labels, cfg, np.random.default_rng(1), crop=False)
    np.testing.assert_array_equal(lab, labels)
    assert any(o.startswith("blur") for o in ops)


def test_registered_photometric_op_is_seeded_and_cannot_touch_labels():
    from microseg_workbench.core.augment import PHOTOMETRIC_OPS, register_photometric_op

    register_photometric_op("invert_test", lambda img, rng, p: p["scale"] * (1.0 - img) + rng.normal(0, 0.01, img.shape),
                            {"scale": 1.0}, "test op")
    try:
        image, labels = _pair()
        cfg = AugmentConfig.from_dict({"patch_size": 32, "photometric_prob": 0.0,
                                       "extra_ops": [{"name": "invert_test", "probability": 1.0}]})
        a = augment(image, labels, cfg, np.random.default_rng(3))
        b = augment(image, labels, cfg, np.random.default_rng(3))
        np.testing.assert_array_equal(a[0], b[0])
        assert "invert_test" in a[2]
        assert cfg.to_dict()["extra_ops"] == [{"name": "invert_test", "probability": 1.0}]
        with pytest.raises(AugmentConfigError, match="Unknown augmentation"):
            AugmentConfig(extra_ops=({"name": "nope", "probability": 1.0},))
        register_photometric_op("crop_test", lambda img, rng, p: img[1:], {}, "")
        bad = AugmentConfig(patch_size=32, extra_ops=({"name": "crop_test", "probability": 1.0},))
        with pytest.raises(AugmentConfigError, match="changed the image shape"):
            augment(image, labels, bad, np.random.default_rng(0))
    finally:
        PHOTOMETRIC_OPS.pop("invert_test", None)
        PHOTOMETRIC_OPS.pop("crop_test", None)


@pytest.mark.parametrize("seed", range(6))
def test_windowed_rotation_on_large_image_is_aligned_and_fully_valid(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, size=(300, 260)).astype(np.uint8)
    image = (labels.astype(np.float32) / 4.0)[:, :, None]
    cfg = AugmentConfig(patch_size=64, rotate_prob=1.0, rotate_max_deg=45, photometric_prob=0.0)
    img2, lab2, ops = augment(image, labels, cfg, np.random.default_rng(seed), image_resample=Image.Resampling.NEAREST)
    assert lab2.shape == (64, 64) and any(o.startswith("rotate") and "deg" in o for o in ops)
    assert not (lab2 == IGNORE_INDEX).any()  # window >= patch*sqrt(2): no rotation corners inside the patch
    np.testing.assert_array_equal(img2[..., 0], lab2.astype(np.float32) / 4.0)
