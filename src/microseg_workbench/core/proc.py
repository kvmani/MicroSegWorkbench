"""Process liveness checks (used to tell interrupted work from work still in progress)."""

from __future__ import annotations

import os


def pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        if os.name == "nt":
            return True  # cannot tell without psutil; assume alive rather than delete live work
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
