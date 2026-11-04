"""Boundary F1, area fractions and per-image scores: hand-checked cases."""

import numpy as np
import pytest

from microseg_workbench.core.accuracy import AccuracyAccumulator, boundary, dilate, per_image_csv
from microseg_workbench.core.augment import IGNORE_INDEX


def square(size=20, top=5, left=5, side=8, value=1):
    m = np.zeros((size, size), np.uint8)
    m[top:top + side, left:left + side] = value
    return m


def test_boundary_is_the_ring_and_image_edges_do_not_count():
    m = square().astype(bool)
    b = boundary(m)
    assert b.sum() == 4 * 8 - 4  # ring of an 8x8 square
    full = np.ones((6, 6), bool)
    assert not boundary(full).any()


def test_dilate_square_neighbourhood():
    m = np.zeros((9, 9), bool)
    m[4, 4] = True
    assert dilate(m, 2).sum() == 25 and dilate(m, 0).sum() == 1


def test_perfect_prediction_scores_one_and_fractions_match():
    t = square()
    acc = AccuracyAccumulator(["matrix", "hydride"])
    acc.add("a", t, t.copy())
    r = acc.result()
    c = r["per_class"][1]
    assert c["boundary_f1"] == 1.0 and c["area_fraction_true"] == c["area_fraction_pred"] == 64 / 400
    assert c["area_fraction_error"] == 0 and c["per_image_area_fraction_mae"] == 0
    assert r["per_image"][0]["mean_iou_foreground"] == 1.0


def test_one_pixel_shift_is_within_tolerance_but_three_is_not():
    t = square()
    near = AccuracyAccumulator(["m", "h"], tolerance=2)
    near.add("a", t, np.roll(t, 1, axis=1))
    assert near.result()["per_class"][1]["boundary_f1"] == 1.0
    far = AccuracyAccumulator(["m", "h"], tolerance=2)
    far.add("a", t, np.roll(t, 3, axis=1))
    f1 = far.result()["per_class"][1]["boundary_f1"]
    assert 0 < f1 < 1  # the vertical edges are 3 px off; the horizontal ones still match


def test_missing_class_gives_zero_and_absent_class_gives_none():
    t = square()
    acc = AccuracyAccumulator(["m", "h", "pore"])
    acc.add("a", t, np.zeros_like(t))
    r = acc.result()
    assert r["per_class"][1]["boundary_f1"] == 0.0 and r["per_class"][1]["boundary_recall"] == 0.0
    assert r["per_class"][1]["area_fraction_error"] == pytest.approx(-64 / 400)
    assert r["per_class"][2]["boundary_f1"] is None and r["per_class"][2]["area_fraction_true"] == 0.0


def test_ignored_pixels_are_not_scored_and_do_not_create_boundaries():
    t = square()
    t[:, :3] = IGNORE_INDEX  # an ignored strip away from the square
    t[5:13, 9] = IGNORE_INDEX  # an ignored line through the square
    acc = AccuracyAccumulator(["m", "h"])
    pred = square()
    acc.add("a", t, pred)
    r = acc.result()
    assert r["per_image"][0]["evaluated_pixels"] == 400 - 60 - 8
    assert r["per_class"][1]["boundary_f1"] == 1.0  # the ignored line is not a boundary error


def test_per_image_worst_first_confusion_and_csv():
    t = square()
    acc = AccuracyAccumulator(["matrix", "hydride"])
    acc.add("good", t, t.copy())
    acc.add("bad", t, np.zeros_like(t))
    r = acc.result(confusion=[[3, 1], [0, 0]])
    assert [i["stem"] for i in r["per_image"]] == ["bad", "good"]
    assert r["confusion_row_normalized"] == [[0.75, 0.25], [None, None]]
    assert r["per_class"][1]["per_image_area_fraction_mae"] == pytest.approx(64 / 400 / 2)
    text = per_image_csv(r, ["matrix", "hydride"])
    lines = text.strip().splitlines()
    assert lines[0].startswith("image,evaluated_pixels") and lines[1].startswith("bad,400")
    assert "16" in lines[2]  # 64/400 = 16 % hydride


def test_accuracy_findings_in_plain_words():
    from microseg_workbench.training.diagnostics import _accuracy_findings

    acc = {"boundary_tolerance_px": 2, "per_class": [
        {"name": "matrix", "boundary_f1": 0.9, "area_fraction_true": 0.9, "area_fraction_pred": 0.86,
         "area_fraction_error": -0.04},
        {"name": "hydride", "boundary_f1": 0.4, "area_fraction_true": 0.05, "area_fraction_pred": 0.08,
         "area_fraction_error": 0.03},
        {"name": "pore", "boundary_f1": 0.9, "area_fraction_true": 0.05, "area_fraction_pred": 0.051,
         "area_fraction_error": 0.001}],
        "per_image": [{"stem": "bad", "mean_iou_foreground": 0.4}, {"stem": "a", "mean_iou_foreground": 0.9},
                      {"stem": "b", "mean_iou_foreground": 0.95}]}
    metrics = {"per_class": [{"name": "matrix", "iou": 0.9}, {"name": "hydride", "iou": 0.7},
                             {"name": "pore", "iou": 0.9}]}
    found = _accuracy_findings(acc, metrics)
    text = " ".join(m for _, m in found)
    assert ("warn", ) == tuple(lv for lv, m in found if "overestimates" in m)  # hydride only, not pore or matrix
    assert "hydride" in text and "8.00 % predicted vs 5.00 % true" in text
    assert "Boundaries of 'hydride'" in text and "Test image bad" in text
    assert _accuracy_findings(None, metrics) == []
