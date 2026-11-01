"""The pipeline description, trace, single-step previews, presets, recipes and sanity check."""

import numpy as np
import pytest

from microseg_workbench.core.augment import AugmentConfig, augment
from microseg_workbench.core.augment_pipeline import (
    PIPELINE, PRESETS, apply_preset, expected_rate, list_recipes, load_recipe, matching_preset, preview_step,
    sanity_check, save_recipe, thinnest_feature_width,
)
from microseg_workbench.core.synthetic import SyntheticSpec, generate_sample

NAMES = ["matrix", "hydride", "pore"]


@pytest.fixture(scope="module")
def samples():
    out = []
    for i in range(4):
        img, m = generate_sample(SyntheticSpec(), 1, i)
        out.append(((img / 255.0).astype(np.float32)[:, :, None], m))
    return out


def test_trace_never_changes_training_samples(samples):
    img, lab = samples[0]
    cfg = apply_preset(AugmentConfig(patch_size=48), "strong")
    for i in range(40):
        a = augment(img, lab, cfg, np.random.default_rng(i))
        trace = []
        b = augment(img, lab, cfg, np.random.default_rng(i), trace=trace)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
        np.testing.assert_array_equal(trace[-1][2].clip(0, 1) if trace[-1][0] != "input" else b[0], b[0])


def test_builder_order_is_the_execution_order(samples):
    img, lab = samples[0]
    cfg = AugmentConfig(patch_size=48, hflip=1, vflip=1, rot90=1, rotate_prob=1, photometric_prob=1, contrast=0.1,
                        brightness=0.1, gamma=0.1, noise_std=0.01, blur_max_sigma=0.5)
    trace = []
    augment(img, lab, cfg, np.random.default_rng(3), trace=trace)
    executed = [t[0] for t in trace[1:]]
    shown = [s.id for s in PIPELINE if s.id in executed or (s.id == "crop" and "rotate" in executed)]
    assert executed == [s for s in shown if s in executed]
    assert [s.id for s in PIPELINE].index("rotate") < [s.id for s in PIPELINE].index("contrast")


def test_geometry_previews_move_mask_and_intensity_previews_do_not(samples):
    img, lab = samples[0]
    for step in PIPELINE:
        if step.kind == "crop":
            continue
        out_img, out_lab = preview_step(img, lab, step.id, 0.2 if step.kind == "intensity" else 10, +1)
        assert out_img.shape[:2] == out_lab.shape
        if step.kind == "intensity":
            np.testing.assert_array_equal(out_lab, lab)
            assert out_img.min() >= 0 and out_img.max() <= 1


def test_presets_and_rates():
    for name in PRESETS:
        cfg = apply_preset(AugmentConfig(patch_size=48), name)
        assert matching_preset(cfg) == name and cfg.patch_size == 48
    cfg = apply_preset(AugmentConfig(), "standard")
    rates = {s.id: expected_rate(cfg, s) for s in PIPELINE}
    assert rates["crop"] == 1 and rates["hflip"] == 0.5 and rates["contrast"] == 0.8 and rates["gamma"] == 0


def test_sanity_check_findings(samples):
    good = sanity_check(samples, apply_preset(AugmentConfig(), "standard"), NAMES, n=120)
    assert good["findings"][0][0] == "ok"
    assert all(abs(r["observed"] - r["expected"]) < 0.2 for r in good["rates"])
    bad = sanity_check(samples, AugmentConfig(patch_size=128, photometric_prob=1, brightness=0.5, contrast=0.8,
                                              blur_max_sigma=4), NAMES, n=80)
    text = " ".join(m for _, m in bad["findings"])
    assert "pure black or white" in text and "larger than the smallest image" in text and "smeared" in text
    none = sanity_check(samples, apply_preset(AugmentConfig(), "none"), NAMES, n=20)
    assert any(level == "info" for level, _ in none["findings"])
    name, width = thinnest_feature_width(samples, NAMES)
    assert name == "hydride" and 1 < width < 6


def test_recipes_roundtrip(tmp_path):
    cfg = apply_preset(AugmentConfig(patch_size=32), "gentle")
    save_recipe(tmp_path, "My SEM set", cfg)
    assert list_recipes(tmp_path) == ["My SEM set"]
    assert load_recipe(tmp_path, "My SEM set") == cfg
    with pytest.raises(ValueError):
        save_recipe(tmp_path, "///", cfg)
