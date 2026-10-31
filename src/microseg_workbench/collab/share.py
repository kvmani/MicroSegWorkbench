"""The shared office drive: authoritative, immutable publication of dataset versions, runs and reviews.

Layout of the share root::

    msw-share.json                        marker (schema, created_at)
    blobs/sha256/aa/bb/<hash>             content-addressed files (see blobstore.py)
    datasets/<dataset_id>.json            published dataset version record (written once)
    runs/<run_id>.json                    published run record (written once)
    reviews/<run_id>/<time>-<id>.json     append-only review decisions, one file each

Rules:
* Records are created exclusively and never modified; a second publication of the same id is a no-op if the
  content is identical and an error otherwise.
* A record is written only after every file it references is fully uploaded and verified, so readers never see a
  record pointing at missing or partial data.
* No database file lives on the share; concurrent writers on different computers cannot corrupt each other.
* Only completed runs whose dataset version is already published can be published (provenance stays complete).
"""

from __future__ import annotations

import getpass
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .. import TOOL_ID, __version__
from ..core.atomic import canonical_json, publish_dir, remove_stale_staging, staging_dir, write_json_atomic
from ..core.dataset import MANIFEST_FILENAME, DatasetVersion
from ..core.workspace import Workspace
from ..training.runs import RunDir
from .blobstore import BlobStore, sha256_of
from .fsops import create_exclusive_json, read_json

SHARE_SCHEMA = "microseg-workbench.share/1"
DATASET_RECORD = "microseg-workbench.shared-dataset/1"
RUN_RECORD = "microseg-workbench.shared-run/1"
REVIEW_RECORD = "microseg-workbench.review/1"
DECISIONS = ("approved", "rejected", "needs_changes")
Progress = Callable[[str], None]


class ShareError(RuntimeError):
    """Problem with the shared drive; message is safe to show to users."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_user() -> str:
    import os

    return os.environ.get("MSW_USER") or getpass.getuser()


def local_blob_cache(workspace: Workspace) -> BlobStore:
    """Local cache of downloaded blobs: keeps partial progress of large downloads across interruptions."""
    return BlobStore(workspace.root / "cache" / "blobs")


@dataclass(frozen=True)
class SharedStore:
    root: Path

    # ---- setup ---------------------------------------------------------------------------------------------
    @classmethod
    def create(cls, root: Path) -> "SharedStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        create_exclusive_json(root / "msw-share.json", {"schema": SHARE_SCHEMA, "created_at": utc_now(),
                                                         "created_with": f"{TOOL_ID} {__version__}"})
        return cls.open(root)

    @classmethod
    def open(cls, root: Path) -> "SharedStore":
        root = Path(root)
        marker = root / "msw-share.json"
        if not marker.exists():
            raise ShareError(f"{root} is not a MicroSegWorkbench shared folder (no msw-share.json). "
                             "Ask the administrator for the correct path, or initialise it with 'msw share init'.")
        if read_json(marker).get("schema") != SHARE_SCHEMA:
            raise ShareError(f"{root} uses an unsupported share format.")
        return cls(root)

    @property
    def blobs(self) -> BlobStore:
        return BlobStore(self.root / "blobs")

    def _dataset_record_path(self, dataset_id: str) -> Path:
        return self.root / "datasets" / f"{_safe(dataset_id)}.json"

    def _run_record_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"{_safe(run_id)}.json"

    # ---- datasets ------------------------------------------------------------------------------------------
    def publish_dataset(self, version: DatasetVersion, user: str | None = None,
                        progress: Progress | None = None) -> dict[str, Any]:
        problems = version.verify()
        if problems:
            raise ShareError(f"Local dataset {version.dataset_id} failed its integrity check: {problems[:3]}")
        files: dict[str, str] = {}
        rels = [MANIFEST_FILENAME] + [p for s in version.samples() for p in (s["image_file"], s["mask_file"])]
        total = 0
        for i, rel in enumerate(rels, 1):
            if progress:
                progress(f"Uploading {i}/{len(rels)}: {rel}")
            path = version.root / rel
            files[rel] = self.blobs.put_file(path)
            total += path.stat().st_size
        record = {
            "schema": DATASET_RECORD,
            "dataset_id": version.dataset_id,
            "name": version.manifest["name"],
            "files": files,
            "bytes": total,
            "class_names": version.class_map.names,
            "split_counts": version.manifest["split_counts"],
            "published_by": user or current_user(),
            "published_at": utc_now(),
            "tool": f"{TOOL_ID} {__version__}",
        }
        return self._create_record(self._dataset_record_path(version.dataset_id), record, ("files",))

    def list_datasets(self) -> list[dict[str, Any]]:
        return _read_records(self.root / "datasets", DATASET_RECORD)

    def dataset_record(self, dataset_id: str) -> dict[str, Any]:
        path = self._dataset_record_path(dataset_id)
        if not path.exists():
            raise ShareError(f"Dataset {dataset_id} is not published on the share.")
        return read_json(path)

    def pull_dataset(self, dataset_id: str, workspace: Workspace, progress: Progress | None = None) -> DatasetVersion:
        """Download (resumably, verified) into the workspace; reuses an intact local copy."""
        record = self.dataset_record(dataset_id)
        workspace.ensure()
        final = workspace.datasets_dir / dataset_id
        if final.exists():
            existing = DatasetVersion.open(final)
            if not existing.verify():
                return existing
            raise ShareError(f"Local copy of {dataset_id} is damaged; delete {final} and download again.")
        self._materialize(record["files"], workspace, workspace.datasets_dir, dataset_id, progress)
        version = DatasetVersion.open(final)
        problems = version.verify()
        if problems:
            raise ShareError(f"Downloaded dataset failed verification: {problems[:3]}")
        return version

    # ---- runs ----------------------------------------------------------------------------------------------
    def publish_run(self, run: RunDir, checkpoints: tuple[str, ...] = ("best",), user: str | None = None,
                    progress: Progress | None = None) -> dict[str, Any]:
        if run.state() != "completed":
            raise ShareError(f"Only completed runs can be published; {run.run_id} is {run.state()}.")
        info = run.info()
        if info.get("imported_from_share"):
            raise ShareError("This run was downloaded from the share; it is already published.")
        dataset_id = info["dataset"]["dataset_id"]
        if not self._dataset_record_path(dataset_id).exists():
            raise ShareError(f"Publish dataset version {dataset_id} first, so others can reproduce this run.")
        rels = ["run.json", "status.json", "events.jsonl"]
        rels += sorted(p.relative_to(run.root).as_posix() for p in run.evaluation.rglob("*") if p.is_file())
        rels += sorted(p.relative_to(run.root).as_posix() for p in run.samples.rglob("*") if p.is_file())
        rels += [f"checkpoints/{c}.pt" for c in checkpoints]
        files: dict[str, str] = {}
        for i, rel in enumerate(rels, 1):
            path = run.root / rel
            if not path.exists():
                raise ShareError(f"Run file {rel} is missing.")
            if progress:
                progress(f"Uploading {i}/{len(rels)}: {rel}")
            files[rel] = self.blobs.put_file(path)
        evaluation = json.loads((run.evaluation / "evaluation.json").read_text())
        metrics = evaluation.get("metrics") or {}
        record = {
            "schema": RUN_RECORD,
            "run_id": run.run_id,
            "dataset_id": dataset_id,
            "files": files,
            "config": info.get("config"),
            "model": info.get("model"),
            "device": info.get("device"),
            "summary": {"split": evaluation.get("split"), "mean_iou": metrics.get("mean_iou"),
                        "mean_iou_foreground": metrics.get("mean_iou_foreground"),
                        "pixel_accuracy": metrics.get("pixel_accuracy"),
                        "per_class_iou": {c["name"]: c["iou"] for c in metrics.get("per_class", [])}},
            "published_by": user or current_user(),
            "published_at": utc_now(),
            "tool": f"{TOOL_ID} {__version__}",
        }
        return self._create_record(self._run_record_path(run.run_id), record, ("files", "dataset_id"))

    def list_runs(self) -> list[dict[str, Any]]:
        runs = _read_records(self.root / "runs", RUN_RECORD)
        for r in runs:
            r["review"] = self.review_status(r["run_id"])
        return runs

    def run_record(self, run_id: str) -> dict[str, Any]:
        path = self._run_record_path(run_id)
        if not path.exists():
            raise ShareError(f"Run {run_id} is not published on the share.")
        return read_json(path)

    def pull_run(self, run_id: str, workspace: Workspace, with_dataset: bool = True,
                 progress: Progress | None = None) -> RunDir:
        record = self.run_record(run_id)
        workspace.ensure()
        if with_dataset:
            self.pull_dataset(record["dataset_id"], workspace, progress)
        final = workspace.runs_dir / run_id
        if final.exists():
            return RunDir(final)
        self._materialize(record["files"], workspace, workspace.runs_dir, run_id, progress,
                          extra_dirs=("checkpoints", "samples", "evaluation"))
        run = RunDir(final)
        info = run.info()
        info["imported_from_share"] = {"share": str(self.root), "published_by": record["published_by"],
                                       "published_at": record["published_at"], "downloaded_at": utc_now()}
        write_json_atomic(run.run_json, info)
        return run

    # ---- reviews -------------------------------------------------------------------------------------------
    def add_review(self, run_id: str, decision: str, comment: str, reviewer: str | None = None,
                   allow_self_approval: bool = False) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise ShareError(f"Decision must be one of {DECISIONS}.")
        record = self.run_record(run_id)
        reviewer = reviewer or current_user()
        if decision == "approved" and reviewer == record["published_by"] and not allow_self_approval:
            raise ShareError("A model must be approved by someone other than the person who published it.")
        if decision != "approved" and not comment.strip():
            raise ShareError("Please explain a rejection or requested change in the comment.")
        review = {"schema": REVIEW_RECORD, "run_id": run_id, "decision": decision, "comment": comment,
                  "reviewer": reviewer, "reviewed_at": utc_now(),
                  "run_record_sha256": sha256_of(self._run_record_path(run_id))}
        name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}.json"
        create_exclusive_json(self.root / "reviews" / _safe(run_id) / name, review)
        return review

    def reviews(self, run_id: str) -> list[dict[str, Any]]:
        return sorted(_read_records(self.root / "reviews" / _safe(run_id), REVIEW_RECORD),
                      key=lambda r: r["reviewed_at"])

    def review_status(self, run_id: str) -> str:
        """Latest decision, or ``unreviewed``."""
        reviews = self.reviews(run_id)
        return reviews[-1]["decision"] if reviews else "unreviewed"

    def approved_runs(self) -> list[dict[str, Any]]:
        return [r for r in self.list_runs() if r["review"] == "approved"]

    # ---- maintenance ---------------------------------------------------------------------------------------
    def check(self) -> list[str]:
        """Verify every record's blobs exist and match their hashes (slow on large shares)."""
        problems = []
        for kind, records in (("dataset", self.list_datasets()), ("run", self.list_runs())):
            for r in records:
                for rel, digest in r["files"].items():
                    if not self.blobs.verify(digest):
                        problems.append(f"{kind} {r.get('dataset_id') or r.get('run_id')}: {rel} missing or corrupt")
        return problems

    # ---- helpers -------------------------------------------------------------------------------------------
    def _create_record(self, path: Path, record: dict[str, Any], identity_keys: tuple[str, ...]) -> dict[str, Any]:
        if create_exclusive_json(path, record):
            return record
        existing = read_json(path)
        if all(canonical_json(existing.get(k)) == canonical_json(record.get(k)) for k in identity_keys):
            return existing  # already published with identical content
        raise ShareError(f"{path.name} is already published with different content; published records never change.")

    def _materialize(self, files: dict[str, str], workspace: Workspace, parent: Path, name: str,
                     progress: Progress | None, extra_dirs: tuple[str, ...] = ()) -> None:
        cache = local_blob_cache(workspace)
        remove_stale_staging(parent)
        stage = staging_dir(parent, name)
        try:
            for d in extra_dirs:
                (stage / d).mkdir(exist_ok=True)
            for i, (rel, digest) in enumerate(sorted(files.items()), 1):
                if progress:
                    progress(f"Downloading {i}/{len(files)}: {rel}")
                cache.copy_from(self.blobs, digest)
                cache.get_file(digest, stage / rel)
            publish_dir(stage, parent / name)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise


def _safe(name: str) -> str:
    if not name or any(c in name for c in '\\/:*?"<>|') or name.startswith("."):
        raise ShareError(f"Invalid name {name!r}")
    return name


def _read_records(folder: Path, schema: str) -> list[dict[str, Any]]:
    out = []
    if folder.exists():
        for p in sorted(folder.glob("*.json")):
            try:
                rec = read_json(p)
            except (OSError, ValueError):
                continue
            if rec.get("schema") == schema:
                out.append(rec)
    return out

