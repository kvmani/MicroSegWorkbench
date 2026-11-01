"""File-system primitives that are safe on local disks and SMB network shares.

* ``create_exclusive`` publishes a complete file under a name that must not exist yet. It never
  overwrites, so immutable records cannot be clobbered by a concurrent writer.
* Partially written data only ever lives under temporary names.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any


def host_id() -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in socket.gethostname()) or "host"


WINDOWS_RETRIES = 40
WINDOWS_RETRY_DELAY_S = 0.05


def unlink_with_retry(path: Path) -> None:
    """Delete a file; on Windows retry while another process briefly has it open (sharing violation)."""
    for attempt in range(WINDOWS_RETRIES):
        try:
            os.unlink(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if os.name != "nt" or attempt == WINDOWS_RETRIES - 1:
                raise
            time.sleep(WINDOWS_RETRY_DELAY_S)


class UnsafeFilesystemError(OSError):
    """The file system cannot create a file only-if-absent atomically."""


def _rename_no_replace(src: Path, dst: Path, identical_content: bool = False) -> None:
    """Move ``src`` to ``dst`` only if ``dst`` does not exist (FileExistsError otherwise).

    ``identical_content`` (content-addressed blobs) allows a last-resort plain rename where no atomic primitive
    exists, because replacing a blob with the same bytes changes nothing. Records never use it: on such file
    systems publishing fails closed with UnsafeFilesystemError.
    """
    if os.name == "nt":
        # Windows: os.rename fails if dst exists. On SMB shares a concurrent creator can make it fail with
        # "Access is denied" instead of FileExistsError; if dst now exists that is the same outcome.
        for attempt in range(WINDOWS_RETRIES):
            try:
                os.rename(src, dst)
                return
            except FileExistsError:
                raise
            except PermissionError:
                if Path(dst).exists():
                    raise FileExistsError(str(dst)) from None
                if attempt == WINDOWS_RETRIES - 1:
                    raise
                time.sleep(WINDOWS_RETRY_DELAY_S)
        return
    try:
        os.link(src, dst)  # POSIX: atomic, fails if dst exists
    except FileExistsError:
        raise
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EACCES):
            raise
        if dst.exists():
            raise FileExistsError(str(dst)) from exc
        if not identical_content:
            raise UnsafeFilesystemError(
                f"{Path(dst).parent} is on a file system without hard links (e.g. a Linux CIFS mount), so a "
                "published record could be overwritten by a simultaneous publication. Publish from a Windows "
                "workstation, or mount the share with hard-link support.") from exc
        os.rename(src, dst)  # content-addressed: an overwrite would store identical bytes
        return
    os.unlink(src)


def create_exclusive(path: Path, data: bytes) -> bool:
    """Atomically create ``path`` with ``data``. Returns False (and writes nothing) if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            _rename_no_replace(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        if tmp.exists():
            unlink_with_retry(tmp)


def create_exclusive_json(path: Path, payload: Any) -> bool:
    return create_exclusive(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
