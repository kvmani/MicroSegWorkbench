"""Optional HydrideSegmentation adapter: contract checks against known geometry (skipped if unavailable)."""

import numpy as np
import pytest

from microseg_workbench.integrations import hydrideseg

available, why = hydrideseg.availability()
needs_hs = pytest.mark.skipif(not available, reason=why)


def test_unavailable_is_reported_clearly(monkeypatch):
    monkeypatch.setenv(hydrideseg.ENV, "/nonexistent/path")
    monkeypatch.setattr(hydrideseg, "_load", lambda: (_ for _ in ()).throw(
        hydrideseg.HydrideSegUnavailable("HydrideSegmentation analysis is not available")))
    ok, message = hydrideseg.availability()
    assert not ok and "not available" in message


@needs_hs
def test_orientation_and_area_of_known_bars():
    horizontal = np.zeros((100, 100), np.uint8)
    for y in (20, 50, 80):
        horizontal[y:y + 3, 10:90] = 1
    stats = hydrideseg.feature_statistics(horizontal)
    assert len(stats["orientations_deg"]) == 3
    assert all(abs(a) < 5 for a in stats["orientations_deg"])
    vertical = horizontal.T.copy()
    assert all(abs(a - 90) < 5 for a in hydrideseg.feature_statistics(vertical)["orientations_deg"])
    assert stats["source"]["package"] == "HydrideSegmentation"
    d = hydrideseg.distribution_distances(horizontal, horizontal)
    assert d["mask_area_fraction_abs_error"] == 0 and d["hydride_orientation_ks"] == 0
    d2 = hydrideseg.distribution_distances(horizontal, vertical)
    assert d2["hydride_orientation_wasserstein"] > 60


@needs_hs
@pytest.mark.training
def test_run_analysis_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from microseg_workbench.cli import main

    ws = str(tmp_path / "ws")
    data = str(tmp_path / "d")
    assert main(["--workspace", ws, "demo-data", "--out", data, "--specimens", "3", "--per-specimen", "2",
                 "--size", "64"]) == 0
    assert main(["--workspace", ws, "prepare", data, "--name", "h"]) == 0
    dataset = next((tmp_path / "ws" / "datasets").iterdir()).name
    cfg = tmp_path / "c.json"
    cfg.write_text('{"epochs": 2, "base_channels": 4, "patches_per_image": 1, "device": "cpu"}')
    assert main(["--workspace", ws, "train", dataset, "--config", str(cfg), "--run-id", "r", "--quiet"]) == 0
    assert main(["--workspace", ws, "hydride-stats", "r", "--um-per-px", "0.25"]) == 0
    assert (tmp_path / "ws" / "runs" / "r" / "evaluation" / "hydride_analysis_hydride.json").exists()
    assert main(["--workspace", ws, "predict", "r", data, "--out", str(tmp_path / "p"), "--device", "cpu"]) == 0
    assert main(["--workspace", ws, "hydride-stats", "--predictions", str(tmp_path / "p")]) == 0
    assert main(["--workspace", ws, "hydride-stats", "r", "--class", "nope"]) == 1
