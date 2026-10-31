"""Phase 3: shared-drive store, resumable verified transfers, concurrency, publication rules and reviews."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from microseg_workbench.collab.blobstore import BlobIntegrityError, BlobStore, sha256_of
from microseg_workbench.collab.fsops import create_exclusive
from microseg_workbench.collab.share import ShareError, SharedStore, local_blob_cache
from microseg_workbench.core.dataset import create_dataset_version, validate_folder
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
from microseg_workbench.core.workspace import Workspace

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_blob_dedupe_and_verify(tmp_path):
    store = BlobStore(tmp_path / "s")
    a = tmp_path / "a.bin"
    a.write_bytes(os.urandom(10_000))
    b = tmp_path / "b.bin"
    b.write_bytes(a.read_bytes())
    d1, d2 = store.put_file(a), store.put_file(b)
    assert d1 == d2 == sha256_of(a)
    assert len(list((tmp_path / "s" / "sha256").rglob("*"))) == 3  # 2 dir levels + 1 blob
    out = tmp_path / "out.bin"
    store.get_file(d1, out)
    assert out.read_bytes() == a.read_bytes()
    store.path(d1).write_bytes(b"corrupted")
    assert not store.verify(d1)
    with pytest.raises(BlobIntegrityError):
        store.get_file(d1, tmp_path / "x.bin")


def test_interrupted_copy_resumes_and_verifies(tmp_path):
    src_store, dst_store = BlobStore(tmp_path / "share"), BlobStore(tmp_path / "cache")
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(9 * 1024 * 1024 + 123))
    digest = src_store.put_file(big)
    with pytest.raises(OSError, match="simulated"):
        dst_store.copy_from(src_store, digest, fail_after=5 * 1024 * 1024)
    part = next((tmp_path / "cache" / "partial").glob("*.part"))
    assert part.stat().st_size == 5 * 1024 * 1024 and not dst_store.has(digest)
    seen = []
    dst_store.copy_from(src_store, digest, progress=lambda done, total: seen.append(done))
    assert dst_store.verify(digest)
    assert seen[0] > 5 * 1024 * 1024  # continued, did not restart from zero
    assert not list((tmp_path / "cache" / "partial").glob("*.part"))


def test_source_changing_during_copy_is_detected(tmp_path):
    store = BlobStore(tmp_path / "s")
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc" * 1000)
    wrong = "0" * 64
    with pytest.raises(BlobIntegrityError, match="does not match"):
        store.put_file(f, expected=wrong)
    assert not store.has(wrong)


def test_create_exclusive_has_one_winner_under_contention(tmp_path):
    target = tmp_path / "rec.json"
    results = []
    barrier = threading.Barrier(8)

    def writer(i):
        barrier.wait()
        try:
            results.append((i, create_exclusive(target, f"writer {i}".encode())))
        except Exception as exc:  # any error in a writer thread fails the test
            results.append((i, exc))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not [r for _, r in results if isinstance(r, Exception)], results
    winners = [i for i, ok in results if ok is True]
    assert len(winners) == 1
    assert target.read_text() == f"writer {winners[0]}"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_concurrent_processes_store_same_blob(tmp_path):
    payload = tmp_path / "p.bin"
    payload.write_bytes(os.urandom(3 * 1024 * 1024))
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from pathlib import Path; from microseg_workbench.collab.blobstore import BlobStore;"
            "print(BlobStore(Path(sys.argv[2])).put_file(Path(sys.argv[3])))")
    procs = [subprocess.Popen([sys.executable, "-c", code, SRC, str(tmp_path / "s"), str(payload)],
                              stdout=subprocess.PIPE, text=True) for _ in range(4)]
    digests = {p.communicate(timeout=60)[0].strip() for p in procs}
    assert all(p.returncode == 0 for p in procs)
    assert digests == {sha256_of(payload)}
    assert BlobStore(tmp_path / "s").verify(digests.pop())


@pytest.fixture
def two_workspaces(synthetic_folder, tmp_path):
    share = SharedStore.create(tmp_path / "share")
    alice = Workspace(tmp_path / "alice").ensure()
    bob = Workspace(tmp_path / "bob").ensure()
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    version = create_dataset_version(report, SplitConfig(seed=4), alice.datasets_dir, "shared")
    return share, alice, bob, version


def test_share_must_be_initialised(tmp_path):
    with pytest.raises(ShareError, match="not a MicroSegWorkbench shared folder"):
        SharedStore.open(tmp_path)


def test_publish_and_pull_dataset_roundtrip(two_workspaces):
    share, alice, bob, version = two_workspaces
    rec = share.publish_dataset(version, user="alice")
    assert rec["published_by"] == "alice" and len(rec["files"]) == 1 + 2 * len(version.samples())
    again = share.publish_dataset(version, user="carol")  # idempotent: identical content
    assert again["published_by"] == "alice"
    assert [d["dataset_id"] for d in share.list_datasets()] == [version.dataset_id]
    pulled = share.pull_dataset(version.dataset_id, bob)
    assert pulled.verify() == [] and pulled.manifest == version.manifest
    assert share.pull_dataset(version.dataset_id, bob).root == pulled.root  # reuses intact local copy
    assert share.check() == []


def test_interrupted_dataset_download_recovers(two_workspaces, monkeypatch):
    share, alice, bob, version = two_workspaces
    share.publish_dataset(version, user="alice")
    calls = {"n": 0}
    original = BlobStore.copy_from

    def flaky(self, other, digest, progress=None, fail_after=None):
        calls["n"] += 1
        if calls["n"] == 5:
            raise OSError("network share disconnected")
        return original(self, other, digest, progress, fail_after)

    monkeypatch.setattr(BlobStore, "copy_from", flaky)
    with pytest.raises(OSError, match="disconnected"):
        share.pull_dataset(version.dataset_id, bob)
    assert not (bob.datasets_dir / version.dataset_id).exists()  # nothing half-published
    cached = sum(1 for _ in (bob.root / "cache" / "blobs" / "sha256").rglob("*") if _.is_file())
    assert cached == 4  # completed files stay in the local cache
    pulled = share.pull_dataset(version.dataset_id, bob)
    assert pulled.verify() == []


def test_corrupt_share_blob_is_not_accepted(two_workspaces):
    share, alice, bob, version = two_workspaces
    rec = share.publish_dataset(version, user="alice")
    digest = rec["files"]["manifest.json"]
    share.blobs.path(digest).write_bytes(b"tampered")
    assert share.check()
    with pytest.raises(BlobIntegrityError):
        share.pull_dataset(version.dataset_id, bob)
    assert not local_blob_cache(bob).has(digest)


@pytest.mark.training
def test_run_publication_rules_pull_and_reviews(two_workspaces):
    pytest.importorskip("torch")
    from microseg_workbench.training.compare import compare_runs
    from microseg_workbench.training.config import TrainConfig
    from microseg_workbench.training.runs import RunDir
    from microseg_workbench.training.trainer import run_training

    share, alice, bob, version = two_workspaces
    cfg = TrainConfig(dataset_id=version.dataset_id, epochs=2, batch_size=4, base_channels=4, depth=2,
                      patches_per_image=1, device="cpu", fixed_samples=1)
    run = run_training(cfg, alice, run_id="alice-run")
    with pytest.raises(ShareError, match="Publish dataset version"):
        share.publish_run(run, user="alice")
    share.publish_dataset(version, user="alice")
    rec = share.publish_run(run, user="alice")
    assert "checkpoints/best.pt" in rec["files"] and "checkpoints/last.pt" not in rec["files"]
    assert rec["summary"]["mean_iou"] is not None

    unfinished = RunDir(alice.runs_dir / "partial").create()
    unfinished.write_status(state="running")
    (unfinished.run_json).write_text(json.dumps({"dataset": {"dataset_id": version.dataset_id}}))
    with pytest.raises(ShareError, match="Only completed runs"):
        share.publish_run(unfinished, user="alice")

    pulled = share.pull_run("alice-run", bob)
    assert pulled.state() == "completed" and pulled.info()["imported_from_share"]["published_by"] == "alice"
    assert (bob.datasets_dir / version.dataset_id).exists()
    assert compare_runs([pulled])["runs"][0]["test_miou"] == rec["summary"]["mean_iou"]
    with pytest.raises(ShareError, match="already published"):
        share.publish_run(pulled, user="bob")

    assert share.review_status("alice-run") == "unreviewed"
    with pytest.raises(ShareError, match="someone other"):
        share.add_review("alice-run", "approved", "", reviewer="alice")
    with pytest.raises(ShareError, match="explain"):
        share.add_review("alice-run", "rejected", "  ", reviewer="bob")
    share.add_review("alice-run", "needs_changes", "check pore class", reviewer="bob")
    share.add_review("alice-run", "approved", "looks right", reviewer="carol")
    assert share.review_status("alice-run") == "approved"
    assert [r["run_id"] for r in share.approved_runs()] == ["alice-run"]
    assert [r["reviewer"] for r in share.reviews("alice-run")] == ["bob", "carol"]


def test_gc_unreferenced_blobs_dry_run_by_default(two_workspaces, tmp_path, capsys):
    import time

    from microseg_workbench.cli import main

    share, alice, bob, version = two_workspaces
    rec = share.publish_dataset(version, user="alice")
    orphan_old, orphan_new = tmp_path / "o1.bin", tmp_path / "o2.bin"
    orphan_old.write_bytes(b"left over from an interrupted publication")
    orphan_new.write_bytes(b"maybe still being published")
    d_old, d_new = share.blobs.put_file(orphan_old), share.blobs.put_file(orphan_new)
    past = time.time() - 30 * 86400
    os.utime(share.blobs.path(d_old), (past, past))
    for digest in list(rec["files"].values())[:3]:
        os.utime(share.blobs.path(digest), (past, past))  # old but referenced
    assert [d for d, _ in share.unreferenced_blobs(7 * 86400)] == [d_old]
    ws = ["--workspace", str(tmp_path / "ws")]
    assert main([*ws, "share", "--share", str(share.root), "gc", "--unreferenced"]) == 0
    assert "Would delete" in capsys.readouterr().out and share.blobs.has(d_old)
    assert main([*ws, "share", "--share", str(share.root), "gc", "--unreferenced", "--apply"]) == 0
    assert not share.blobs.has(d_old) and share.blobs.has(d_new)
    assert share.check() == []
