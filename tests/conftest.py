from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset  # noqa: E402


@pytest.fixture
def synthetic_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "synthetic"
    write_synthetic_dataset(folder, SyntheticSpec(specimens=5, images_per_specimen=3, size=64, seed=3))
    return folder
