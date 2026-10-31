import numpy as np
import pytest

from microseg_workbench.core.augment import IGNORE_INDEX
from microseg_workbench.core.metrics import confusion_matrix, metrics_from_confusion


def test_hand_computed_example():
    truth = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
    pred = np.array([[0, 1, 1, 1], [0, 2, 0, 1]])
    cm = confusion_matrix(truth, pred, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 3, 0], [1, 0, 1]]
    m = metrics_from_confusion(cm, ["bg", "a", "b"])
    # class a: TP=3, FP=1, FN=0 -> IoU 3/4, Dice 6/7
    a = m["per_class"][1]
    assert a["iou"] == pytest.approx(0.75)
    assert a["dice"] == pytest.approx(6 / 7)
    assert a["precision"] == pytest.approx(0.75)
    assert a["recall"] == pytest.approx(1.0)
    # bg: TP=2, FP=1, FN=1 -> 0.5 ; b: TP=1 FP=0 FN=1 -> 0.5
    assert m["per_class"][0]["iou"] == pytest.approx(0.5)
    assert m["per_class"][2]["iou"] == pytest.approx(0.5)
    assert m["mean_iou"] == pytest.approx((0.5 + 0.75 + 0.5) / 3)
    assert m["mean_iou_foreground"] == pytest.approx((0.75 + 0.5) / 2)
    assert m["pixel_accuracy"] == pytest.approx(6 / 8)


def test_absent_class_is_none_not_zero_or_one():
    truth = np.zeros((3, 3), int)
    pred = np.zeros((3, 3), int)
    m = metrics_from_confusion(confusion_matrix(truth, pred, 2), ["bg", "feature"])
    assert m["per_class"][1]["iou"] is None
    assert m["per_class"][1]["dice"] is None
    assert m["mean_iou"] == pytest.approx(1.0)
    assert m["mean_iou_foreground"] is None


def test_ignore_pixels_excluded():
    truth = np.array([[1, IGNORE_INDEX], [0, IGNORE_INDEX]])
    pred = np.array([[1, 0], [0, 1]])
    cm = confusion_matrix(truth, pred, 2)
    assert cm.sum() == 2
    assert metrics_from_confusion(cm, ["bg", "f"])["pixel_accuracy"] == 1.0


def test_out_of_range_prediction_rejected():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros((2, 2), int), np.full((2, 2), 3), 2)
