"""Local workspace layout (a Windows local disk folder in the office)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "MSW_WORKSPACE"


def default_workspace_root() -> Path:
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser() if env else Path.home() / "MicroSegWorkbench-workspace"


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def demo_dir(self) -> Path:
        return self.root / "demo-data"

    def ensure(self) -> "Workspace":
        for d in (self.datasets_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def default(cls) -> "Workspace":
        return cls(default_workspace_root())
