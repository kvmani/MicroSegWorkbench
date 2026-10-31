"""Run directories: the on-disk record of one training run.

Layout of ``<workspace>/runs/<run_id>/``::

    run.json          resolved config, dataset id, environment provenance
    status.json       live state (running/completed/failed/cancelled), atomically replaced
    events.jsonl      append-only step/epoch/evaluation events (live dashboards tail this)
    checkpoints/      last.pt and best.pt (atomically replaced)
    samples/          fixed-sample truth and per-epoch prediction overlays
    evaluation/       test-set report and overlays (review_status: unreviewed)
    CANCEL            created by the UI/CLI to request a graceful stop
"""

from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.atomic import read_json, write_json_atomic

TERMINAL_STATES = ("completed", "failed", "cancelled")
ACTIVE_STATES = ("starting", "running", "evaluating")


def pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        if os.name == "nt":
            return True  # cannot tell without psutil; assume alive rather than mislabel
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def status_json(self) -> Path:
        return self.root / "status.json"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def cancel_flag(self) -> Path:
        return self.root / "CANCEL"

    def create(self) -> "RunDir":
        self.root.mkdir(parents=True, exist_ok=False)
        for d in (self.checkpoints, self.samples, self.evaluation):
            d.mkdir()
        return self

    def append_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True) + "\n"
        with open(self.events, "a", encoding="utf-8") as fh:
            fh.write(line)  # readers ignore an unterminated last line, so no fsync per event

    def read_events(self) -> list[dict[str, Any]]:
        if not self.events.exists():
            return []
        out = []
        with open(self.events, encoding="utf-8") as fh:
            for line in fh:
                if line.endswith("\n"):  # a partially written last line is skipped, not misparsed
                    out.append(json.loads(line))
        return out

    def write_status(self, **status: Any) -> None:
        status["updated_at"] = utc_now()
        status.setdefault("pid", os.getpid())
        status.setdefault("host", platform.node())
        write_json_atomic(self.status_json, status)

    def state(self) -> str:
        """Status state, reporting ``interrupted`` when the training process on this computer has died."""
        st = self.status()
        state = st.get("state", "unknown")
        if state in ACTIVE_STATES and st.get("host") == platform.node() and st.get("pid"):
            if not pid_alive(int(st["pid"])):
                return "interrupted"
        return state

    def resumable(self) -> bool:
        return self.state() in ("cancelled", "failed", "interrupted") and (self.checkpoints / "last.pt").exists()

    def epoch_history(self) -> list[dict[str, Any]]:
        """One epoch event per epoch (the latest, so epochs repeated after a resume are not double counted)."""
        by_epoch: dict[int, dict[str, Any]] = {}
        for e in self.read_events():
            if e["type"] == "epoch":
                by_epoch[e["epoch"]] = e
        return [by_epoch[k] for k in sorted(by_epoch)]

    def status(self) -> dict[str, Any]:
        try:
            return read_json(self.status_json)
        except (FileNotFoundError, ValueError):
            return {"state": "unknown"}

    def info(self) -> dict[str, Any]:
        try:
            return read_json(self.run_json)
        except (FileNotFoundError, ValueError):
            return {}

    def request_cancel(self) -> None:
        self.cancel_flag.write_text("cancel requested\n", encoding="utf-8")

    def cancel_requested(self) -> bool:
        return self.cancel_flag.exists()


def list_runs(runs_dir: Path) -> list[RunDir]:
    if not runs_dir.exists():
        return []
    runs = [RunDir(p) for p in runs_dir.iterdir() if p.is_dir() and (p / "run.json").exists()]
    return sorted(runs, key=lambda r: r.run_id, reverse=True)


class EventTail:
    """Incrementally read complete new lines of ``events.jsonl`` (for live dashboards)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def read_new(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        end = data.rfind(b"\n") + 1  # leave an unfinished last line for the next poll
        self.offset += end
        return [json.loads(line) for line in data[:end].decode("utf-8").splitlines() if line.strip()]
