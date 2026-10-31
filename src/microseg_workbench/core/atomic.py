"""Atomic file and directory publication helpers.

Readers must never observe partially written files. Everything is written to a
temporary sibling first and moved into place with ``os.replace`` (atomic on the
same filesystem on Windows and POSIX).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(payload: Any) -> str:
    """Stable JSON text used for content hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(path: Path, payload: Any) -> None:
    write_bytes_atomic(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def staging_dir(parent: Path, name: str) -> Path:
    """Create a private staging directory next to the final location."""
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f".staging-{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    return path


def publish_dir(staging: Path, final: Path) -> bool:
    """Move a fully written staging directory to ``final``.

    Returns ``True`` when published, ``False`` if ``final`` already existed (the
    staging copy is then discarded; callers use content-addressed names so an
    existing directory has identical content).
    """
    if final.exists():
        shutil.rmtree(staging, ignore_errors=True)
        return False
    try:
        os.replace(staging, final)
    except OSError:
        if final.exists():  # lost a race against an identical publication
            shutil.rmtree(staging, ignore_errors=True)
            return False
        raise
    return True


def remove_stale_staging(parent: Path) -> list[Path]:
    """Delete leftovers of interrupted publications (recovery after a crash)."""
    removed: list[Path] = []
    if not parent.exists():
        return removed
    for child in parent.iterdir():
        if child.is_dir() and child.name.startswith(".staging-"):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
    return removed
