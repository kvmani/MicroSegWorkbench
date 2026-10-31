"""Two colleagues collaborating through the CLI, with the metadata service."""

import threading

import pytest

from microseg_workbench.cli import main
from microseg_workbench.collab.server import MetadataDB, make_server

pytest.importorskip("torch")


def test_two_users_share_review_and_index(tmp_path, capsys, monkeypatch):
    server = make_server(tmp_path / "meta.sqlite", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    db = MetadataDB(tmp_path / "meta.sqlite")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    tok_a, tok_b = db.add_user("alice"), db.add_user("bob")
    share, wa, wb = str(tmp_path / "share"), str(tmp_path / "wa"), str(tmp_path / "wb")
    try:
        assert main(["share", "init", share]) == 0
        assert main(["--workspace", wa, "share", "config", "--path", share, "--user", "alice", "--service", url,
                     "--token", tok_a]) == 0
        assert main(["--workspace", wb, "share", "config", "--path", share, "--user", "bob", "--service", url,
                     "--token", tok_b]) == 0
        data = str(tmp_path / "data")
        assert main(["--workspace", wa, "demo-data", "--out", data, "--specimens", "3", "--per-specimen", "2",
                     "--size", "64"]) == 0
        assert main(["--workspace", wa, "prepare", data, "--name", "d"]) == 0
        dataset_id = next(l.split(": ")[1] for l in capsys.readouterr().out.splitlines()
                          if l.startswith("Dataset version"))
        cfg = tmp_path / "c.json"
        cfg.write_text('{"epochs": 1, "base_channels": 4, "patches_per_image": 1, "device": "cpu"}')
        assert main(["--workspace", wa, "train", dataset_id, "--config", str(cfg), "--run-id", "ra", "--quiet"]) == 0
        assert main(["--workspace", wa, "share", "publish-run", "ra"]) == 1  # dataset not yet published
        assert main(["--workspace", wa, "share", "--quiet", "publish-dataset", dataset_id]) == 0
        assert main(["--workspace", wa, "share", "--quiet", "publish-run", "ra"]) == 0
        assert main(["--workspace", wa, "share", "review", "ra", "--decision", "approved"]) == 1  # self-approval
        assert main(["--workspace", wb, "share", "--quiet", "pull-run", "ra"]) == 0
        assert main(["--workspace", wb, "share", "review", "ra", "--decision", "approved", "--comment", "ok"]) == 0
        capsys.readouterr()
        assert main(["--workspace", wb, "share", "status"]) == 0
        out = capsys.readouterr().out
        assert "[APPROVED]" in out and "local copy" in out
        assert main(["--workspace", wb, "share", "check"]) == 0
        assert main(["--workspace", wb, "share", "sync"]) == 0
        from microseg_workbench.collab.client import ServiceClient

        runs = ServiceClient(url, tok_b).runs()
        assert runs[0]["review_status"] == "approved"
        assert main(["--workspace", wb, "predict", "ra", data, "--out", str(tmp_path / "p"), "--device", "cpu"]) == 0
    finally:
        server.shutdown()
        server.server_close()
