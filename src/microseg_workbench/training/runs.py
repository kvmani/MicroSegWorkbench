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
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.atomic import read_json, write_json_atomic

TERMINAL_STATES = ("completed", "failed", "cancelled")


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
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

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
        write_json_atomic(self.status_json, status)

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
