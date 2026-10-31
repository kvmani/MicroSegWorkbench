"""Leakage-safe, deterministic train/validation/test splitting by group.

All images sharing a *group* (specimen, acquisition session, source micrograph)
go to the same split, so near-duplicate views never leak between training and
evaluation. Assignment depends only on the group names, their image counts,
the fractions and the seed - never on file system order.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

SPLITS = ("train", "val", "test")
DEFAULT_GROUP_REGEX = r"^([^_]+)_"


class SplitError(ValueError):
    """Split cannot be made as requested; the message is safe to show to users."""


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2
    seed: int = 42
    group_regex: str = DEFAULT_GROUP_REGEX
    source: str = "group"  # "group": seeded group split; "annotator": splits assigned in the annotation tool

    def __post_init__(self) -> None:
        if self.source not in ("group", "annotator"):
            raise SplitError("Split source must be 'group' or 'annotator'.")

    def fractions(self) -> dict[str, float]:
        total = self.train + self.val + self.test
        if min(self.train, self.val, self.test) < 0 or total <= 0:
            raise SplitError("Split fractions must be non-negative and not all zero.")
        if self.train <= 0:
            raise SplitError("The training fraction must be greater than zero.")
        return {"train": self.train / total, "val": self.val / total, "test": self.test / total}


def derive_group(stem: str, regex: str) -> str:
    """Group key from a file stem: first capture group (or whole match) of ``regex``.

    Stems that do not match form their own group (reported by validation).
    """
    try:
        pattern = re.compile(regex)
    except re.error as exc:
        raise SplitError(f"Invalid grouping pattern {regex!r}: {exc}") from exc
    match = pattern.search(stem)
    if match is None:
        return stem
    value = match.group(1) if match.groups() else match.group(0)
    return value or stem


def assign_splits(group_sizes: dict[str, int], config: SplitConfig) -> dict[str, str]:
    """Map each group to a split.

    Groups are visited in a seed-dependent hash order. Every split with a
    positive fraction first receives one group (test, then val, then train), then
    each remaining group goes to the split whose image count is furthest below
    its target. Raises if there are too few groups to honour non-zero fractions.
    """
    fractions = config.fractions()
    wanted = [s for s in ("test", "val", "train") if fractions[s] > 0]
    if len(group_sizes) < len(wanted):
        raise SplitError(
            f"Only {len(group_sizes)} group(s) found but {len(wanted)} non-empty splits requested. "
            "Each split needs at least one whole group (specimen) to stay leakage-free. "
            "Add data from more specimens, adjust the grouping pattern, or set a fraction to 0."
        )
    order = sorted(group_sizes, key=lambda g: hashlib.sha256(f"{config.seed}:{g}".encode()).hexdigest())
    total = sum(group_sizes.values())
    counts = {s: 0 for s in SPLITS}
    assignment: dict[str, str] = {}
    for split, group in zip(wanted, order):
        assignment[group] = split
        counts[split] += group_sizes[group]
    for group in order[len(wanted):]:
        deficits = {s: fractions[s] * total - counts[s] for s in SPLITS if fractions[s] > 0}
        split = max(deficits, key=lambda s: (deficits[s], -SPLITS.index(s)))
        assignment[group] = split
        counts[split] += group_sizes[group]
    return assignment


def check_fixed_splits(assignment: dict[str, str], groups: dict[str, str]) -> None:
    """Validate externally assigned splits (stem -> split) against leakage by group (stem -> group)."""
    bad = sorted(s for s, split in assignment.items() if split not in SPLITS)
    if bad:
        raise SplitError(f"{len(bad)} image(s) have no train/val/test split assigned (e.g. {bad[:3]}). Assign splits "
                         "in the annotation tool or use the group split instead.")
    if "train" not in assignment.values():
        raise SplitError("No image is assigned to the training split.")
    by_group: dict[str, set[str]] = {}
    for stem, split in assignment.items():
        by_group.setdefault(groups[stem], set()).add(split)
    leaking = sorted(g for g, splits in by_group.items() if len(splits) > 1)
    if leaking:
        raise SplitError(f"The assigned splits put images of the same group in different splits (groups "
                         f"{', '.join(leaking[:5])}). That leaks information into evaluation. Fix the assignment "
                         "or the grouping pattern, or use the group split.")
