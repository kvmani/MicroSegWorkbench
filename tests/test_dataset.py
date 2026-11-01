import json
import shutil

import numpy as np
import pytest
from PIL import Image

from microseg_workbench.core.atomic import remove_stale_staging
from microseg_workbench.core.dataset import DatasetVersion, create_dataset_version, validate_folder
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
from microseg_workbench.core.synthetic import SyntheticSpec, generate_sample


def codes(report):
    return {i.code for i in report.issues}


def test_synthetic_generation_is_deterministic(tmp_path):
    a, m = generate_sample(SyntheticSpec(), 1, 2)
    b, n = generate_sample(SyntheticSpec(), 1, 2)
    assert np.array_equal(a, b) and np.array_equal(m, n)
    assert set(np.unique(m)) <= {0, 1, 2} and (m == 1).any()


def test_valid_synthetic_folder(synthetic_folder):
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    assert report.ok, report.issues
    assert len(report.pairs) == 15
    assert {p.group for p in report.pairs} == {"S01", "S02", "S03", "S04", "S05"}


def test_detects_pairing_shape_value_and_duplicate_problems(synthetic_folder):
    (synthetic_folder / "S01_0002_mask.png").unlink()
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(synthetic_folder / "S02_0001_mask.png")
    bad = np.asarray(Image.open(synthetic_folder / "S03_0001_mask.png")).copy()
    bad[0, 0] = 7
    Image.fromarray(bad).save(synthetic_folder / "S03_0001_mask.png")
    shutil.copy(synthetic_folder / "S04_0001.png", synthetic_folder / "S04_0009.png")
    shutil.copy(synthetic_folder / "S04_0001_mask.png", synthetic_folder / "S04_0009_mask.png")
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(synthetic_folder / "orphan_mask.png")
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    assert {"image_without_mask", "shape_mismatch", "mask_values", "duplicate_image",
            "mask_without_image"} <= codes(report)
    assert not report.ok


def test_colour_mask_and_missing_classes_are_errors(synthetic_folder):
    Image.fromarray(np.zeros((64, 64, 3), np.uint8) + np.array([255, 0, 0], np.uint8)).save(
        synthetic_folder / "S01_0001_mask.png")
    (synthetic_folder / "classes.json").unlink()
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    assert {"unreadable", "class_map"} <= codes(report)


def test_binary_encoding_maps_255_to_class_1_and_rejects_others(tmp_path):
    folder = tmp_path / "bin"
    folder.mkdir()
    for i in range(4):
        Image.fromarray(np.full((16, 16), 100 + i, np.uint8)).save(folder / f"g{i}_a.png")
        m = np.zeros((16, 16), np.uint8)
        m[:4] = 255
        Image.fromarray(m).save(folder / f"g{i}_a_mask.png")
    (folder / "classes.json").write_text(json.dumps({
        "schema": "microseg-workbench.classes/1", "mask_encoding": "binary_0_255",
        "classes": [{"value": 0, "name": "bg", "color": "#000000"}, {"value": 1, "name": "hydride", "color": "#ff0000"}]}))
    report = validate_folder(folder, DEFAULT_GROUP_REGEX)
    assert report.ok, report.issues
    assert report.pairs[0].class_pixels == {0: 192, 1: 64}
    m = np.zeros((16, 16), np.uint8)
    m[0, 0] = 128
    Image.fromarray(m).save(folder / "g0_a_mask.png")
    assert "mask_values" in codes(validate_folder(folder, DEFAULT_GROUP_REGEX))


def test_dataset_version_is_immutable_content_addressed_and_leakage_free(synthetic_folder, tmp_path):
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    out = tmp_path / "datasets"
    crashed = out / ".staging-crashed-1234"  # leftover of an interrupted run: its owner process is gone
    crashed.mkdir(parents=True)
    import json as _json
    import platform
    import subprocess
    import sys

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (crashed / ".msw-owner.json").write_text(_json.dumps({"pid": dead.pid, "host": platform.node(), "created": 0}))
    v1 = create_dataset_version(report, SplitConfig(seed=1), out, "demo")
    assert not (out / ".staging-crashed-1234").exists()
    assert v1.verify() == []
    groups = v1.manifest["groups"]
    assert all(len(splits) == 1 for splits in groups.values())
    assert set(v1.manifest["split_counts"]) == {"train", "val", "test"}
    assert all(v1.manifest["split_counts"][s] > 0 for s in ("train", "val", "test"))
    # stored masks hold exactly the source labels
    s = v1.samples()[0]
    src = np.asarray(Image.open(synthetic_folder / f"{s['stem']}_mask.png"))
    np.testing.assert_array_equal(np.asarray(Image.open(v1.mask_path(s))), src)
    # same inputs -> same id, existing version reused
    v2 = create_dataset_version(report, SplitConfig(seed=1), out, "demo")
    assert v2.dataset_id == v1.dataset_id
    assert v2.manifest["created_at"] == v1.manifest["created_at"]
    # different seed -> different version
    v3 = create_dataset_version(report, SplitConfig(seed=2), out, "demo")
    assert v3.dataset_id != v1.dataset_id
    assert sorted(p.name for p in out.iterdir()) == sorted([v1.dataset_id, v3.dataset_id])
    # tampering is detected
    v1.mask_path(s).write_bytes(b"corrupt")
    assert v1.verify()
    assert DatasetVersion.open(v1.root).dataset_id == v1.dataset_id


def test_invalid_report_cannot_be_published(synthetic_folder, tmp_path):
    (synthetic_folder / "S01_0001_mask.png").unlink()
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    with pytest.raises(ValueError, match="validation errors"):
        create_dataset_version(report, SplitConfig(), tmp_path / "d", "demo")
    assert not (tmp_path / "d").exists() or not any((tmp_path / "d").iterdir())


def test_remove_stale_staging_ignores_real_versions(tmp_path):
    import os
    import time

    (tmp_path / "keep").mkdir()
    (tmp_path / ".staging-x").mkdir()  # unowned and fresh: could be in use by an older build - kept
    assert remove_stale_staging(tmp_path) == []
    old = time.time() - 2 * 86400
    os.utime(tmp_path / ".staging-x", (old, old))
    removed = remove_stale_staging(tmp_path)
    assert [p.name for p in removed] == [".staging-x"]
    assert (tmp_path / "keep").exists()


def test_replace_retries_windows_sharing_violations(tmp_path, monkeypatch):
    from microseg_workbench.core import atomic

    calls = {"n": 0}
    real = atomic.os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] Access is denied")
        real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    monkeypatch.setattr(atomic.os, "name", "nt")
    atomic.write_bytes_atomic(tmp_path / "status.json", b"{}")
    assert (tmp_path / "status.json").read_bytes() == b"{}" and calls["n"] == 3
