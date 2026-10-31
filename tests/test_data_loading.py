"""Lazy loading, block shuffling, prefetching and the memory guard."""

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from microseg_workbench.core.augment import AugmentConfig  # noqa: E402
from microseg_workbench.core.dataset import create_dataset_version, validate_folder  # noqa: E402
from microseg_workbench.core.preprocess import Preprocess  # noqa: E402
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.training import data as data_mod  # noqa: E402
from microseg_workbench.training.config import TrainConfig  # noqa: E402
from microseg_workbench.training.data import (  # noqa: E402
    SplitSource, _patch_order, count_batches, iter_epoch_batches, load_split, prefetch,
)
from microseg_workbench.training.trainer import run_training  # noqa: E402


@pytest.fixture
def version(synthetic_folder, tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    v = create_dataset_version(validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX), SplitConfig(seed=4),
                               ws.datasets_dir, "t")
    return ws, v


def test_patch_order_covers_every_patch_once_and_blocks_are_local():
    for limit in (0, 3, 100):
        order = _patch_order(10, 4, seed=1, epoch=2, images_in_memory=limit)
        assert sorted(order) == [(i, r) for i in range(10) for r in range(4)]
    blocked = _patch_order(10, 4, seed=1, epoch=2, images_in_memory=3)
    first_block = {i for i, _ in blocked[:12]}
    assert len(first_block) == 3  # one block of 3 images = 12 patches before moving on
    assert _patch_order(10, 4, 1, 2, 0) == _patch_order(10, 4, 1, 2, 0)
    assert _patch_order(10, 4, 1, 3, 0) != _patch_order(10, 4, 1, 2, 0)


def test_split_source_cache_limit_and_equivalence(version):
    _, v = version
    src = SplitSource(v, "train", Preprocess(), max_cached=2)
    loads = []
    original = src._load
    src._load = lambda i: (loads.append(i), original(i))[1]
    for i in range(len(src)):
        src.get(i)
    src.get(len(src) - 1)
    assert len(src._cache) == 2 and loads.count(len(src) - 1) == 1
    full = SplitSource(v, "train", Preprocess())
    a = list(iter_epoch_batches(full, AugmentConfig(patch_size=32), 2, 4, seed=3, epoch=1))
    b = list(iter_epoch_batches([full.get(i) for i in range(len(full))], AugmentConfig(patch_size=32), 2, 4, 3, 1))
    assert len(a) == count_batches(len(full), 2, 4)
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert [s.stem for s in load_split(v, "val")] == SplitSource(v, "val").stems


def test_prefetch_preserves_order_and_propagates_errors():
    assert list(prefetch(iter(range(50)), depth=3)) == list(range(50))

    def broken():
        yield 1
        raise ValueError("bad batch")

    with pytest.raises(ValueError, match="bad batch"):
        list(prefetch(broken()))
    gen = prefetch(iter(range(1000)), depth=2)
    assert next(gen) == 0
    gen.close()  # consumer stops early: producer thread must finish
    assert not any(t.name == "msw-batch-prefetch" and t.is_alive() for t in threading.enumerate())


def test_blocked_training_is_deterministic_and_recorded(version):
    ws, v = version
    cfg = TrainConfig(dataset_id=v.dataset_id, epochs=2, batch_size=4, base_channels=4, depth=2, patches_per_image=2,
                      device="cpu", num_threads=1, images_in_memory=2)
    a, b = run_training(cfg, ws), run_training(cfg, ws)
    steps = lambda r: [e["loss"] for e in r.read_events() if e["type"] == "step"]  # noqa: E731
    assert steps(a) == steps(b)
    assert a.info()["config"]["images_in_memory"] == 2


def test_memory_guard_gives_advice(version, monkeypatch):
    ws, v = version
    monkeypatch.setattr(SplitSource, "estimated_bytes", lambda self: 10 * 2**40)  # pretend 10 TB
    cfg = TrainConfig(dataset_id=v.dataset_id, epochs=1, device="cpu")
    with pytest.raises(MemoryError, match="images in memory"):
        run_training(cfg, ws)
    assert data_mod  # module import kept for monkeypatch clarity
