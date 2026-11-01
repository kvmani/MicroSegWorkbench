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
import time
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


def replace_with_retry(src: Path, dst: Path, attempts: int = 20, delay_s: float = 0.05) -> None:
    """``os.replace`` that tolerates Windows sharing violations.

    On Windows, replacing a file fails with PermissionError while another process (e.g. the UI polling
    ``status.json``) has it open; such reads take milliseconds, so a short retry succeeds.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or attempt == attempts - 1:
                raise
            time.sleep(delay_s)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        replace_with_retry(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(path: Path, payload: Any) -> None:
    write_bytes_atomic(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


OWNER_FILE = ".msw-owner.json"
STALE_UNOWNED_S = 24 * 3600


def staging_dir(parent: Path, name: str) -> Path:
    """Create a private staging directory next to the final location, recording which process owns it."""
    import platform

    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f".staging-{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir()
    (path / OWNER_FILE).write_text(json.dumps({"pid": os.getpid(), "host": platform.node(), "created": time.time()}),
                                   encoding="utf-8")
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
    owner = staging / OWNER_FILE
    if owner.exists():
        owner.unlink()
    try:
        os.replace(staging, final)
    except OSError:
        if final.exists():  # lost a race against an identical publication
            shutil.rmtree(staging, ignore_errors=True)
            return False
        raise
    return True


def _staging_abandoned(path: Path) -> bool:
    """True if the process that owns a staging folder is gone (or it is unowned and older than a day)."""
    import platform

    from .proc import pid_alive

    try:
        owner = json.loads((path / OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        try:
            return time.time() - path.stat().st_mtime > STALE_UNOWNED_S
        except OSError:
            return False
    if owner.get("host") == platform.node():
        return not pid_alive(int(owner.get("pid", 0)))
    return time.time() - float(owner.get("created", 0)) > STALE_UNOWNED_S


def remove_stale_staging(parent: Path) -> list[Path]:
    """Delete leftovers of interrupted preparations/downloads; never touches work of a running process."""
    removed: list[Path] = []
    if not parent.exists():
        return removed
    for child in parent.iterdir():
        if child.is_dir() and child.name.startswith(".staging-") and _staging_abandoned(child):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
    return removed


def safe_relpath(rel: str) -> str:
    """Validate a relative path from a manifest or record before it is used to build a destination.

    Rejects absolute paths, drive letters, backslashes, empty parts and '.'/'..' parts. Returns it unchanged.
    """
    from pathlib import PurePosixPath

    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel or "\x00" in rel:
        raise ValueError(f"Unsafe file path {rel!r}")
    parts = PurePosixPath(rel).parts
    if PurePosixPath(rel).is_absolute() or any(p in ("", ".", "..") for p in parts) or rel != "/".join(parts):
        raise ValueError(f"Unsafe file path {rel!r}")
    return rel
