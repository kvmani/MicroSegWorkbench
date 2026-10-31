"""Loading developer plugins (extra augmentation operations).

Set ``MSW_PLUGINS`` to a comma-separated list of importable module names (or paths to ``.py`` files). Each module
registers operations with ``microseg_workbench.core.augment.register_photometric_op`` when imported. Loaded plugins
and their file hashes are recorded in every training run for provenance.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

ENV = "MSW_PLUGINS"
_loaded: dict[str, dict[str, Any]] = {}


def load_plugins() -> dict[str, dict[str, Any]]:
    for item in [x.strip() for x in os.environ.get(ENV, "").split(",") if x.strip()]:
        if item in _loaded:
            continue
        if item.endswith(".py"):
            path = Path(item)
            spec = importlib.util.spec_from_file_location(f"msw_plugin_{path.stem}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load plugin {item}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(item)
        file = getattr(module, "__file__", None)
        digest = hashlib.sha256(Path(file).read_bytes()).hexdigest() if file else None
        _loaded[item] = {"module": module.__name__, "file": file, "sha256": digest}
    return dict(_loaded)
