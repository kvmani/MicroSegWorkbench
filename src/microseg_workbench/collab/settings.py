"""Per-workspace collaboration settings (``<workspace>/settings.json``, local only, never on the share)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..core.atomic import read_json, write_json_atomic
from ..core.workspace import Workspace


@dataclass
class CollabSettings:
    share_path: str = ""
    user: str = ""  # empty = operating-system user name (or MSW_USER)
    service_url: str = ""  # optional metadata service, e.g. http://vm-host:8765
    service_token: str = ""

    @classmethod
    def load(cls, workspace: Workspace) -> "CollabSettings":
        path = workspace.root / "settings.json"
        if not path.exists():
            return cls()
        data = read_json(path)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, workspace: Workspace) -> None:
        workspace.root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(workspace.root / "settings.json", asdict(self))

    def user_name(self) -> str:
        from .share import current_user

        return self.user or current_user()

    def share_root(self) -> Path | None:
        return Path(self.share_path) if self.share_path else None
