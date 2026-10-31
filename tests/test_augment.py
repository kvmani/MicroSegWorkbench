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
