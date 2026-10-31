"""Content-addressed, immutable blob storage with resumable, verified copies.

Layout: ``<root>/sha256/<aa>/<bb>/<full hash>``. A blob's name is the SHA-256 of its bytes, so

* identical files are stored once (deduplication),
* a blob is never modified after it appears (writers only ever add),
* any reader can verify what it got.

Copies go to ``<root>/partial/<hash>.<host>.part`` first. If a copy is interrupted, the next attempt on the same
computer continues from the bytes already copied (after re-hashing them), verifies the final hash and only then
moves the file into place. A per-host lock file prevents two processes on one computer from appending to the
same partial file; partial files from other computers never collide.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Callable

from .fsops import _rename_no_replace, host_id, unlink_with_retry

CHUNK = 4 * 1024 * 1024
ProgressFn = Callable[[int, int], None]


class BlobError(RuntimeError):
    pass


class BlobIntegrityError(BlobError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _valid_hash(digest: str) -> bool:
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


class _HostLock:
    def __init__(self, path: Path, wait_s: float) -> None:
        self.path = path
        self.wait_s = wait_s

    def __enter__(self) -> "_HostLock":
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                if self._stale():
                    try:
                        unlink_with_retry(self.path)
                    except PermissionError:
                        pass  # someone else is handling it; try again
                    continue
                if time.monotonic() > deadline:
                    raise BlobError(f"Another process on this computer is copying the same file ({self.path}).")
                time.sleep(0.05)

    def _stale(self) -> bool:
        from ..training.runs import pid_alive

        try:
            pid = int(self.path.read_text() or "0")
        except (OSError, ValueError):
            return False
        return pid > 0 and not pid_alive(pid)

    def __exit__(self, *exc) -> None:
        unlink_with_retry(self.path)


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, digest: str) -> Path:
        if not _valid_hash(digest):
            raise BlobError(f"Invalid blob id {digest!r}")
        return self.root / "sha256" / digest[:2] / digest[2:4] / digest

    def has(self, digest: str) -> bool:
        return self.path(digest).is_file()

    def size(self, digest: str) -> int:
        return self.path(digest).stat().st_size

    def verify(self, digest: str) -> bool:
        return self.has(digest) and sha256_of(self.path(digest)) == digest

    def put_file(self, src: Path, expected: str | None = None, progress: ProgressFn | None = None,
                 fail_after: int | None = None) -> str:
        """Store ``src``; returns its hash. ``expected`` (if known) skips hashing when the blob already exists."""
        digest = expected or sha256_of(src)
        if self.has(digest):
            return digest
        self._copy_in(src, digest, progress, fail_after)
        return digest

    def get_file(self, digest: str, dest: Path) -> None:
        """Copy a blob to ``dest`` (verified) via a temporary file."""
        src = self.path(digest)
        if not src.is_file():
            raise BlobError(f"Blob {digest[:12]} is missing from {self.root}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        h = hashlib.sha256()
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            while chunk := fi.read(CHUNK):
                h.update(chunk)
                fo.write(chunk)
        if h.hexdigest() != digest:
            tmp.unlink()
            raise BlobIntegrityError(f"Blob {digest[:12]} in {self.root} is corrupt (hash mismatch).")
        os.replace(tmp, dest)

    def copy_from(self, other: "BlobStore", digest: str, progress: ProgressFn | None = None,
                  fail_after: int | None = None) -> None:
        """Resumable, verified transfer of one blob from another store (e.g. share -> local cache)."""
        if self.has(digest):
            return
        if not other.has(digest):
            raise BlobError(f"Blob {digest[:12]} is missing from {other.root}")
        self._copy_in(other.path(digest), digest, progress, fail_after)

    def _copy_in(self, src: Path, digest: str, progress: ProgressFn | None, fail_after: int | None) -> None:
        partial_dir = self.root / "partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        part = partial_dir / f"{digest}.{host_id()}.part"
        with _HostLock(partial_dir / f"{digest}.{host_id()}.lock", wait_s=600):
            if self.has(digest):
                return
            total = src.stat().st_size
            h = hashlib.sha256()
            done = 0
            if part.exists():
                if part.stat().st_size > total:
                    part.unlink()
                else:
                    with open(part, "rb") as fh:  # re-hash what an interrupted copy already wrote
                        while chunk := fh.read(CHUNK):
                            h.update(chunk)
                            done += len(chunk)
            with open(src, "rb") as fi, open(part, "ab") as fo:
                fi.seek(done)
                while chunk := fi.read(CHUNK):
                    if fail_after is not None and done + len(chunk) > fail_after:  # test hook: simulate a drop
                        fo.write(chunk[: max(0, fail_after - done)])
                        raise OSError("simulated interruption")
                    fo.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
                fo.flush()
                os.fsync(fo.fileno())
            if h.hexdigest() != digest:
                part.unlink()
                raise BlobIntegrityError(f"Copy of {src} does not match its expected hash {digest[:12]}; "
                                         "the source changed or is corrupt. The partial copy was discarded.")
            final = self.path(digest)
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                _rename_no_replace(part, final)
            except FileExistsError:  # another computer published the same blob first: identical content
                unlink_with_retry(part)

    def stale_partials(self, max_age_s: float) -> list[Path]:
        partial_dir = self.root / "partial"
        if not partial_dir.exists():
            return []
        now = time.time()
        return [p for p in partial_dir.iterdir() if now - p.stat().st_mtime > max_age_s]

    def remove_stale_partials(self, max_age_s: float = 7 * 86400) -> list[Path]:
        removed = []
        for p in self.stale_partials(max_age_s):
            try:
                p.unlink() if p.is_file() else shutil.rmtree(p)
                removed.append(p)
            except OSError:
                pass
        return removed
