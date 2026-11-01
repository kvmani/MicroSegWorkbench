"""Regression tests for the independent review RV-20261101-001..008 (docs/REVIEWS.md).

Adapted from the reviewer's defect-confirming probes (docs/REVIEW_REPRODUCTIONS_2026-11-01.md) with the
expectation inverted: each test asserts the safe behaviour.
"""

import errno
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from microseg_workbench.collab.share import SharedStore, ShareError
from microseg_workbench.core.atomic import write_json_atomic
from microseg_workbench.core.dataset import DatasetVersion, SourceChangedError, create_dataset_version, validate_folder
from microseg_workbench.core.imageio import encode_png
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
from microseg_workbench.core.workspace import Workspace
from microseg_workbench.integrations.onlineannotator import OAImportError, import_bundle
from oa_fixture import make_bundle


def dataset(tmp_path):
    folder = tmp_path / "source"
    write_synthetic_dataset(folder, SyntheticSpec(specimens=3, images_per_specimen=1, size=32))
    return folder, validate_folder(folder, DEFAULT_GROUP_REGEX)


# --- RV-20261101-001: manifest edits are detected; runs are bound to their dataset manifest ----------------
@pytest.mark.parametrize("edit", ["split", "class", "hash", "group", "file"])
def test_001_manifest_edits_fail_verification(tmp_path, edit):
    _, report = dataset(tmp_path)
    v = create_dataset_version(report, SplitConfig(), tmp_path / "versions", "d")
    assert v.verify() == []
    m = json.loads((v.root / "manifest.json").read_text())
    if edit == "split":
        for s in m["samples"]:
            s["split"] = "train"
    elif edit == "class":
        m["class_map"]["classes"][1]["name"] = "renamed"
    elif edit == "hash":
        m["samples"][0]["image_sha256"] = "0" * 64
    elif edit == "group":
        m["samples"][0]["group"] = "other"
    else:
        m["samples"][0]["image_file"] = "../../outside.png"
    write_json_atomic(v.root / "manifest.json", m)
    assert DatasetVersion.open(v.root).verify()


@pytest.mark.training
def test_001_reevaluation_and_resume_refuse_a_changed_dataset(tmp_path):
    pytest.importorskip("torch")
    from microseg_workbench.training.config import TrainConfig
    from microseg_workbench.training.trainer import evaluate_run, run_training

    ws = Workspace(tmp_path / "ws").ensure()
    _, report = dataset(tmp_path)
    v = create_dataset_version(report, SplitConfig(), ws.datasets_dir, "d")
    run = run_training(TrainConfig(dataset_id=v.dataset_id, epochs=1, batch_size=1, base_channels=4, depth=1,
                                   patches_per_image=1, fixed_samples=0, num_threads=1, device="cpu"), ws)
    assert evaluate_run(run, ws)["split"] == "test"  # unchanged dataset stays evaluable
    trained = {s["stem"] for s in v.samples("train")}
    for s in v.manifest["samples"]:  # the reviewer's swap of train and test assignments
        s["split"] = "test" if s["stem"] in trained else ("train" if s["split"] == "test" else s["split"])
    write_json_atomic(v.root / "manifest.json", v.manifest)
    with pytest.raises(ValueError, match="integrity|changed"):
        evaluate_run(run, ws)


# --- RV-20261101-002: sources changed after checking are refused ------------------------------------------
@pytest.mark.parametrize("change", ["smaller", "same_size_edit", "image"])
def test_002_changed_sources_are_refused(tmp_path, change):
    folder, report = dataset(tmp_path)
    pair = report.pairs[0]
    if change == "smaller":
        pair.mask_path.write_bytes(encode_png(np.zeros((16, 16), np.uint8)))
    elif change == "same_size_edit":
        mask = np.asarray(Image.open(pair.mask_path)).copy()
        mask[:4, :4] = 1
        pair.mask_path.write_bytes(encode_png(mask))
    else:
        img = np.asarray(Image.open(pair.image_path)).copy()
        img[0, 0] ^= 1
        Image.fromarray(img).save(pair.image_path)
    with pytest.raises(SourceChangedError, match="Check folder"):
        create_dataset_version(report, SplitConfig(), tmp_path / "versions", "d")
    assert not any(p.name.startswith("d-") for p in (tmp_path / "versions").iterdir())
    fresh = validate_folder(folder, DEFAULT_GROUP_REGEX)  # re-checking picks up the edit
    if change != "smaller":
        assert create_dataset_version(fresh, SplitConfig(), tmp_path / "versions", "d").verify() == []


# --- RV-20261101-003: approval provenance is part of the dataset identity ---------------------------------
def test_003_approval_changes_give_a_new_version_in_both_directions(tmp_path):
    folder, report = dataset(tmp_path)
    stem = report.pairs[0].stem
    samples = {p.stem: {"version": 1, "version_status": "approved", "approved_by": "rev"} for p in report.pairs}
    prov = {"schema": "microseg-workbench.provenance/1", "samples": samples, "unapproved": []}
    write_json_atomic(folder / "provenance.json", prov)
    approved = create_dataset_version(validate_folder(folder, DEFAULT_GROUP_REGEX), SplitConfig(),
                                      tmp_path / "versions", "d")
    prov["unapproved"] = [stem]
    prov["samples"][stem].update(version_status="submitted", approved_by=None, version=2)
    write_json_atomic(folder / "provenance.json", prov)
    submitted = create_dataset_version(validate_folder(folder, DEFAULT_GROUP_REGEX), SplitConfig(),
                                       tmp_path / "versions", "d")
    assert submitted.dataset_id != approved.dataset_id
    assert submitted.manifest["contains_unapproved_annotations"] is True
    assert approved.manifest["contains_unapproved_annotations"] is False
    assert submitted.verify() == [] and approved.verify() == []
    # identical repeated preparation still reuses the same version
    again = create_dataset_version(validate_folder(folder, DEFAULT_GROUP_REGEX), SplitConfig(),
                                   tmp_path / "versions", "d")
    assert again.dataset_id == submitted.dataset_id


def test_003_ids_without_provenance_are_unchanged(tmp_path):
    _, report = dataset(tmp_path)
    v = create_dataset_version(report, SplitConfig(), tmp_path / "versions", "d")
    assert v.manifest["identity_includes_annotation"] is False and v.verify() == []


# --- RV-20261101-004: unsafe names and paths are rejected before anything is written ----------------------
@pytest.mark.parametrize("stem", ["../escaped", "a/b", "a\\b", "C:x", ".hidden", "", "x\x00"])
def test_004_import_rejects_unsafe_stems(tmp_path, stem):
    src = make_bundle(tmp_path / "original.zip", specimens=1, per=1)
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist()}
    m = json.loads(files["manifest.json"])
    m["images"][0]["stem"] = stem
    files["manifest.json"] = json.dumps(m).encode()
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    with pytest.raises(OAImportError, match="cannot be used as a file name"):
        import_bundle(tmp_path / "bad.zip", tmp_path / "imported")
    assert not (tmp_path / "escaped.png").exists() and not (tmp_path / "imported").exists()


@pytest.mark.parametrize("rel", ["../escaped", "/abs/x", "a/../../b", "C:/x", "a\\..\\b"])
def test_004_share_record_paths_cannot_escape(tmp_path, rel):
    store = SharedStore.create(tmp_path / "share")
    ws = Workspace(tmp_path / "ws").ensure()
    src = tmp_path / "payload"
    src.write_bytes(b"overwritten")
    digest = store.blobs.put_file(src)
    outside = ws.datasets_dir / "escaped"
    outside.write_bytes(b"original")
    with pytest.raises(ShareError, match="unsafe file path"):
        store._materialize({rel: digest}, ws, ws.datasets_dir, "safe", None)
    assert outside.read_bytes() == b"original" and not (ws.datasets_dir / "safe").exists()


# --- RV-20261101-005: no silent overwrite on file systems without hard links ------------------------------
def test_005_no_hardlinks_fails_closed_for_records(tmp_path, monkeypatch):
    import microseg_workbench.collab.fsops as f

    if f.os.name == "nt":
        pytest.skip("Windows uses os.rename, which never replaces")
    barrier = threading.Barrier(2)
    rename = f.os.rename

    def no_link(*a, **k):
        raise OSError(errno.EOPNOTSUPP, "not supported")

    def race_rename(src, dst):
        barrier.wait(timeout=5)
        return rename(src, dst)

    monkeypatch.setattr(f.os, "link", no_link)
    monkeypatch.setattr(f.os, "rename", race_rename)

    def attempt(data):
        try:
            return f.create_exclusive(tmp_path / "record", data)
        except f.UnsafeFilesystemError:
            return "refused"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, [b"A", b"B"]))
    assert results == ["refused", "refused"] and not (tmp_path / "record").exists()
    with pytest.raises(f.UnsafeFilesystemError):
        f.create_exclusive(tmp_path / "other", b"x")


def test_005_blobs_still_work_without_hardlinks(tmp_path, monkeypatch):
    import microseg_workbench.collab.fsops as f
    from microseg_workbench.collab.blobstore import BlobStore

    monkeypatch.setattr(f.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, "no")))
    src = tmp_path / "p.bin"
    src.write_bytes(b"content")
    store = BlobStore(tmp_path / "blobs")
    assert store.verify(store.put_file(src))


# --- RV-20261101-006: damaged stored copies are not reused when publishing --------------------------------
def test_006_publication_refuses_damaged_existing_blob(tmp_path):
    _, report = dataset(tmp_path)
    v = create_dataset_version(report, SplitConfig(), tmp_path / "versions", "d")
    store = SharedStore.create(tmp_path / "share")
    digest = store.blobs.put_file(v.image_path(v.samples()[0]))
    store.blobs.path(digest).write_bytes(b"corrupt")
    with pytest.raises(ShareError, match="damaged"):
        store.publish_dataset(v)
    assert store.list_datasets() == []
    assert store.blobs.path(digest).read_bytes() == b"corrupt"  # reported, never silently modified


def test_006_damaged_local_cache_copy_is_refetched(tmp_path, synthetic_folder):
    from microseg_workbench.collab.share import local_blob_cache

    store = SharedStore.create(tmp_path / "share")
    alice, bob = Workspace(tmp_path / "a").ensure(), Workspace(tmp_path / "b").ensure()
    v = create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX), SplitConfig(),
                               alice.datasets_dir, "d")
    rec = store.publish_dataset(v, user="alice")
    cache = local_blob_cache(bob)
    digest = rec["files"]["manifest.json"]
    cache.copy_from(store.blobs, digest)
    cache.path(digest).write_bytes(b"bit rot")
    assert store.pull_dataset(v.dataset_id, bob).verify() == []


# --- RV-20261101-007: same-name inputs are refused before any prediction ----------------------------------
@pytest.mark.training
@pytest.mark.parametrize("names", [("same.png", "same.tif"), ("Same.png", "same.bmp")])
def test_007_prediction_refuses_name_clashes(tmp_path, names):
    pytest.importorskip("torch")
    from microseg_workbench.training.engine import save_checkpoint
    from microseg_workbench.training.model import ModelConfig, UNet
    from microseg_workbench.training.predict import PredictionError, predict_folder
    from microseg_workbench.training.runs import RunDir

    _, r = dataset(tmp_path)
    run = RunDir(tmp_path / "run").create()
    save_checkpoint(run.checkpoints / "best.pt", UNet(ModelConfig(1, 3, 4, 1)),
                    {"class_map": r.class_map.to_dict(), "epoch": 1})
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for n in names:
        Image.fromarray(np.zeros((32, 32), np.uint8)).save(inputs / n)
    with pytest.raises(PredictionError, match="share a name"):
        predict_folder(run, inputs, tmp_path / "out", device="cpu")
    assert not (tmp_path / "out").exists()


# --- RV-20261101-008: cleanup never deletes another process's active work ---------------------------------
def test_008_concurrent_preparations_do_not_delete_each_other(tmp_path, monkeypatch):
    import microseg_workbench.core.dataset as d

    _, report = dataset(tmp_path)
    started, release = threading.Event(), threading.Event()
    copy = d.shutil.copyfile

    def paused_copy(src, dst):
        if ".staging-first-" in str(dst):
            started.set()
            assert release.wait(10)
        return copy(src, dst)

    monkeypatch.setattr(d.shutil, "copyfile", paused_copy)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(create_dataset_version, report, SplitConfig(), tmp_path / "versions", "first")
        assert started.wait(5)
        second = create_dataset_version(report, SplitConfig(), tmp_path / "versions", "second")
        release.set()
        first = future.result()
    assert first.verify() == [] and second.verify() == []
    assert not list((tmp_path / "versions").glob(".staging-*"))
    assert not (first.root / ".msw-owner.json").exists()
