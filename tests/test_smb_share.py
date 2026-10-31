"""Shared-folder behaviour on a real network share (SMB/UNC).

Runs only when MSW_TEST_SHARE points at a writable network folder, e.g. ``\\\\localhost\\mswshare`` (the Windows CI
job creates one) or an office test share. Each test uses its own new subfolder.
"""

import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from microseg_workbench.collab.blobstore import BlobStore, sha256_of
from microseg_workbench.collab.fsops import create_exclusive
from microseg_workbench.collab.share import SharedStore
from microseg_workbench.core.dataset import create_dataset_version, validate_folder
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
from microseg_workbench.core.workspace import Workspace

SHARE = os.environ.get("MSW_TEST_SHARE")
pytestmark = pytest.mark.skipif(not SHARE, reason="set MSW_TEST_SHARE to a network folder to run SMB tests")
SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture
def share_root():
    return Path(SHARE) / f"msw-test-{uuid.uuid4().hex[:8]}"


def test_exclusive_create_contention_on_share(share_root):
    share_root.mkdir(parents=True)
    target = share_root / "record.json"
    results = []
    barrier = threading.Barrier(6)

    def writer(i):
        barrier.wait()
        try:
            results.append(create_exclusive(target, f"writer {i}".encode()))
        except Exception as exc:  # any error in a writer thread fails the test
            results.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not [r for r in results if isinstance(r, Exception)], results
    assert results.count(True) == 1 and len(results) == 6
    assert not [p for p in share_root.iterdir() if p.name.endswith(".tmp")]


def test_processes_store_same_blob_on_share(share_root, tmp_path):
    payload = tmp_path / "p.bin"
    payload.write_bytes(os.urandom(2 * 1024 * 1024))
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from pathlib import Path; from microseg_workbench.collab.blobstore import BlobStore;"
            "print(BlobStore(Path(sys.argv[2])).put_file(Path(sys.argv[3])))")
    procs = [subprocess.Popen([sys.executable, "-c", code, SRC, str(share_root / "blobs"), str(payload)],
                              stdout=subprocess.PIPE, text=True) for _ in range(3)]
    outs = {p.communicate(timeout=120)[0].strip() for p in procs}
    assert all(p.returncode == 0 for p in procs) and outs == {sha256_of(payload)}
    assert BlobStore(share_root / "blobs").verify(outs.pop())


def test_publish_pull_review_through_unc_path(share_root, synthetic_folder, tmp_path):
    store = SharedStore.create(share_root)
    alice, bob = Workspace(tmp_path / "a").ensure(), Workspace(tmp_path / "b").ensure()
    version = create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX), SplitConfig(),
                                     alice.datasets_dir, "smb")
    store.publish_dataset(version, user="alice")
    assert store.publish_dataset(version, user="alice")["published_by"] == "alice"  # idempotent
    pulled = SharedStore.open(share_root).pull_dataset(version.dataset_id, bob)
    assert pulled.verify() == [] and store.check() == []
