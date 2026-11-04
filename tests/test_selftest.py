"""Roll-out self-test: real code paths on synthetic data, temporary folders cleaned up."""

import pytest

pytest.importorskip("torch")

from microseg_workbench.cli import main  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.selftest import FAIL, PASS, SKIP, run_selftest  # noqa: E402

pytestmark = pytest.mark.training


def test_selftest_passes_and_cleans_up(tmp_path):
    ws = Workspace(tmp_path / "ws")
    share = tmp_path / "share"
    share.mkdir()
    (share / "real-record.json").write_text("{}")
    result = run_selftest(ws, share=share, device="cpu")
    status = {c.name: c.status for c in result.checks}
    assert result.ok, result.text()
    for name in ("Data validation and dataset version", "Training and evaluation", "Segmenting new images",
                 "Shared folder", "Reproducibility (CPU)"):
        assert status[name] == PASS, result.text()
    assert "READY" in result.text() and "Synthetic data only" in result.text()
    assert sorted(p.name for p in share.iterdir()) == ["real-record.json"]  # temporary store removed
    assert not list(ws.root.glob(".selftest-*"))
    assert result.report_path.read_text().startswith("MicroSegWorkbench")
    assert not list(ws.runs_dir.iterdir()) and not list(ws.datasets_dir.iterdir())  # real workspace untouched


def test_unreachable_share_fails_and_missing_share_is_skipped(tmp_path):
    bad = run_selftest(Workspace(tmp_path / "a"), share=tmp_path / "no-such-drive", device="cpu")
    assert not bad.ok and {c.name: c.status for c in bad.checks}["Shared folder"] == FAIL
    assert "NOT READY" in bad.text()
    none = run_selftest(Workspace(tmp_path / "b"), share=None, device="cpu")
    assert none.ok and {c.name: c.status for c in none.checks}["Shared folder"] == SKIP


def test_cli_selftest(tmp_path, capsys):
    assert main(["--workspace", str(tmp_path / "ws"), "selftest", "--no-share", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "RESULT: READY" in out and "Report saved" in out
