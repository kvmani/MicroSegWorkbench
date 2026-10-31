"""Optional metadata service: auth, immutability, concurrency, review rules, sync from the share."""

import threading

import pytest

from microseg_workbench.collab.client import ServiceClient, ServiceError
from microseg_workbench.collab.server import MetadataDB, make_server


@pytest.fixture
def service(tmp_path):
    server = make_server(tmp_path / "meta.sqlite", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    db = MetadataDB(tmp_path / "meta.sqlite")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    tokens = {name: db.add_user(name) for name in ("alice", "bob")}
    yield url, tokens
    server.shutdown()
    server.server_close()


def run_record(run_id="r1", by="alice", miou=0.9):
    return {"schema": "microseg-workbench.shared-run/1", "run_id": run_id, "dataset_id": "d1", "files": {},
            "published_by": by, "published_at": "2026-10-31T00:00:00+00:00", "summary": {"mean_iou": miou}}


def review(decision, reviewer, at, comment="ok"):
    return {"schema": "microseg-workbench.review/1", "run_id": "r1", "decision": decision, "reviewer": reviewer,
            "reviewed_at": at, "comment": comment}


def test_auth_required_and_health_open(service):
    url, tokens = service
    assert ServiceClient(url, "").health()["status"] == "ok"
    with pytest.raises(ServiceError, match="401"):
        ServiceClient(url, "wrong").runs()
    assert ServiceClient(url, tokens["alice"]).whoami() == "alice"


def test_records_are_immutable_and_reviews_enforced(service):
    url, tokens = service
    alice, bob = ServiceClient(url, tokens["alice"]), ServiceClient(url, tokens["bob"])
    assert alice.register("runs", run_record())["status"] == "created"
    assert alice.register("runs", run_record())["status"] == "exists"
    with pytest.raises(ServiceError, match="409"):
        alice.register("runs", run_record(miou=0.99))
    with pytest.raises(ServiceError, match="400"):
        alice.register("runs", {"schema": "something-else"})
    with pytest.raises(ServiceError, match="403"):
        alice.register("reviews", review("approved", "alice", "2026-10-31T01:00:00+00:00"))
    bob.register("reviews", review("needs_changes", "bob", "2026-10-31T02:00:00+00:00", "check pores"))
    bob.register("reviews", review("approved", "bob", "2026-10-31T03:00:00+00:00"))
    runs = alice.runs()
    assert runs[0]["review_status"] == "approved" and runs[0]["identity_verified"]
    assert [r["record"]["run_id"] for r in alice.approved()] == ["r1"]
    with pytest.raises(ServiceError, match="409"):
        bob.register("reviews", {**review("approved", "bob", "x"), "run_id": "unknown"})


def test_concurrent_registrations(service):
    url, tokens = service
    errors = []

    def worker(i):
        try:
            ServiceClient(url, tokens["alice"]).register("runs", run_record(run_id=f"r{i % 5}"))
        except ServiceError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sorted(r["record"]["run_id"] for r in ServiceClient(url, tokens["bob"]).runs()) == [f"r{i}" for i in range(5)]


def test_database_on_network_share_is_refused(tmp_path):
    with pytest.raises(ValueError, match="local disk"):
        MetadataDB(r"\\fileserver\share\meta.sqlite")


def test_unreachable_service_is_a_clear_error():
    with pytest.raises(ServiceError, match="cannot reach"):
        ServiceClient("http://127.0.0.1:9", "t", timeout=2, retries=1).health()


def test_sync_from_share(service, tmp_path):
    from microseg_workbench.collab.fsops import create_exclusive_json
    from microseg_workbench.collab.share import SharedStore

    url, tokens = service
    store = SharedStore.create(tmp_path / "share")
    create_exclusive_json(store.root / "runs" / "r1.json", run_record())
    store.add_review("r1", "approved", "fine", reviewer="bob")
    client = ServiceClient(url, tokens["bob"])
    assert client.sync_from_share(store) == 2
    assert client.sync_from_share(store) == 2  # idempotent
    runs = client.runs()
    assert runs[0]["review_status"] == "approved"
    assert runs[0]["identity_verified"] is False  # alice's record was submitted by bob
