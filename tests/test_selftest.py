"""Roll-out self-test: real code paths on synthetic data, temporary folders cleaned up."""

import pytest

pytest.importorskip("torch")

from microseg_workbench.cli import main  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.selftest import FAIL, PASS, SKIP, run_selftest  # noqa: E402

pytestmark = pytest.mark.training


def _tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_selftest_passes_and_cleans_up(tmp_path):
    from microseg_workbench.collab.share import SharedStore

    ws = Workspace(tmp_path / "ws")
    share = tmp_path / "share"
    SharedStore.create(share)
    before = _tree(share)
    result = run_selftest(ws, share=share, device="cpu", share_write=True)
    status = {c.name: c.status for c in result.checks}
    assert result.ok, result.text()
    for name in ("Data validation and dataset version", "Training and evaluation", "Segmenting new images",
                 "Shared folder (read)", "Shared folder (write test)", "Reproducibility (CPU)"):
        assert status[name] == PASS, result.text()
    assert "READY" in result.text() and "Synthetic data only" in result.text()
    assert _tree(share) == before  # temporary store removed, real share untouched
    assert not list(ws.root.glob(".selftest-*"))
    assert result.report_path.read_text().startswith("MicroSegWorkbench")
    assert not list(ws.runs_dir.iterdir()) and not list(ws.datasets_dir.iterdir())  # real workspace untouched


def test_unreachable_share_fails_and_missing_share_is_skipped(tmp_path):
    bad = run_selftest(Workspace(tmp_path / "a"), share=tmp_path / "no-such-drive", device="cpu")
    assert not bad.ok and {c.name: c.status for c in bad.checks}["Shared folder (read)"] == FAIL
    assert "NOT READY" in bad.text()
    none = run_selftest(Workspace(tmp_path / "b"), share=None, device="cpu")
    assert none.ok and {c.name: c.status for c in none.checks}["Shared folder (read)"] == SKIP


def test_researcher_selftest_only_reads_the_share(tmp_path):
    """Researchers may only create files inside the record folders: the default check must not write."""
    from microseg_workbench.collab.share import SharedStore

    share = tmp_path / "share"
    SharedStore.create(share)
    before = _tree(share)
    result = run_selftest(Workspace(tmp_path / "ws"), share=share, device="cpu")
    status = {c.name: c.status for c in result.checks}
    assert status["Shared folder (read)"] == PASS and status["Shared folder (write test)"] == SKIP
    assert _tree(share) == before
    not_a_share = tmp_path / "plain"
    not_a_share.mkdir()
    bad = run_selftest(Workspace(tmp_path / "ws2"), share=not_a_share, device="cpu")
    assert "not a MicroSegWorkbench shared folder" in bad.text()


def test_cli_selftest(tmp_path, capsys):
    assert main(["--workspace", str(tmp_path / "ws"), "selftest", "--no-share", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "RESULT: READY" in out and "Report saved" in out
