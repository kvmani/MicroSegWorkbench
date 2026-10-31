"""Numerical regression guard: the quick preset must learn every synthetic class for several seeds.

Thresholds are deliberately loose (scores are typically 0.95-0.98) so normal floating-point differences between
platforms and library versions do not fail the test, while a broken loss, schedule, augmentation or label
mapping (which collapses a class to IoU 0, giving ~0.49) does.
"""

import pytest

pytest.importorskip("torch")

from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training.config import preset_config  # noqa: E402
from microseg_workbench.training.trainer import run_training  # noqa: E402

pytestmark = pytest.mark.training


@pytest.mark.parametrize("seed", [1, 2])
def test_quick_preset_learns_all_classes(tmp_path, seed):
    write_synthetic_dataset(tmp_path / "data", SyntheticSpec())
    ws = Workspace(tmp_path / "ws").ensure()
    version = create_dataset_version(validate_folder(tmp_path / "data", DEFAULT_GROUP_REGEX), SplitConfig(),
                                     ws.datasets_dir, "demo")
    run = run_training(preset_config("quick_cpu", version.dataset_id, seed=seed, device="cpu"), ws)
    import json

    report = json.loads((run.evaluation / "evaluation.json").read_text())
    per_class = {c["name"]: c["iou"] for c in report["metrics"]["per_class"]}
    assert all(v is not None and v > 0.8 for v in per_class.values()), per_class
