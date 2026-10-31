import pytest

from microseg_workbench.core.splits import SplitConfig, SplitError, assign_splits, derive_group


def test_derive_group_default_and_fallback():
    cfg = SplitConfig()
    assert derive_group("S01_0003", cfg.group_regex) == "S01"
    assert derive_group("lonely", cfg.group_regex) == "lonely"
    assert derive_group("specA-img7", r"^(.*)-img\d+$") == "specA"


def test_invalid_regex_is_user_error():
    with pytest.raises(SplitError):
        derive_group("x", "(")


def test_groups_never_cross_splits_and_are_deterministic():
    sizes = {f"G{i}": (i % 4) + 1 for i in range(20)}
    cfg = SplitConfig(train=0.6, val=0.2, test=0.2, seed=11)
    a = assign_splits(sizes, cfg)
    b = assign_splits(dict(reversed(list(sizes.items()))), cfg)
    assert a == b  # insertion order irrelevant
    assert set(a) == set(sizes)
    assert set(a.values()) == {"train", "val", "test"}
    total = sum(sizes.values())
    counts = {s: sum(n for g, n in sizes.items() if a[g] == s) for s in ("train", "val", "test")}
    assert abs(counts["train"] / total - 0.6) < 0.15
    assert assign_splits(sizes, SplitConfig(seed=12)) != a


def test_too_few_groups_is_explained():
    with pytest.raises(SplitError, match="at least one whole group"):
        assign_splits({"A": 5, "B": 5}, SplitConfig())
    # zero test fraction needs only two groups
    out = assign_splits({"A": 5, "B": 5}, SplitConfig(train=0.8, val=0.2, test=0))
    assert sorted(out.values()) == ["train", "val"]
