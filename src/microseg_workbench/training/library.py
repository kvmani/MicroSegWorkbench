"""All trained models available to this user: in the workspace and on the shared folder (torch-free)."""

from __future__ import annotations

import json
from typing import Any

from ..core.workspace import Workspace
from .runs import list_runs


def list_models(workspace: Workspace, store=None) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for run in list_runs(workspace.runs_dir):
        if not (run.checkpoints / "best.pt").exists() or run.state() != "completed":
            continue
        info = run.info()
        ev = run.evaluation / "evaluation.json"
        metrics = (json.loads(ev.read_text()).get("metrics") or {}) if ev.exists() else {}
        rows[run.run_id] = {"run_id": run.run_id, "local": True, "shared": False,
                            "network": info.get("model", {}).get("architecture_title", "U-Net (convolutional)"),
                            "dataset_id": info.get("dataset", {}).get("dataset_id"),
                            "test_miou_fg": metrics.get("mean_iou_foreground"), "review": "not published",
                            "published_by": None,
                            "downloaded": bool(info.get("imported_from_share"))}
    if store is not None:
        for rec in store.list_runs():
            row = rows.setdefault(rec["run_id"], {"run_id": rec["run_id"], "local": False, "downloaded": False})
            card = rec.get("model_card") or {}
            row.update(shared=True, review=rec.get("review", "unreviewed"), published_by=rec.get("published_by"),
                       dataset_id=rec.get("dataset_id"),
                       network=row.get("network") or (card.get("network") or {}).get("title")
                       or (rec.get("model") or {}).get("architecture_title", "U-Net (convolutional)"),
                       test_miou_fg=row.get("test_miou_fg") if row.get("local")
                       else (rec.get("summary") or {}).get("mean_iou_foreground"))
    order = {"approved": 0, "needs_changes": 1, "unreviewed": 2, "rejected": 3, "not published": 4}
    return sorted(rows.values(), key=lambda r: (order.get(r.get("review"), 5), r["run_id"]))
