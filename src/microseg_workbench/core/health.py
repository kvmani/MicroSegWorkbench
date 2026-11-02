"""Plain-language checks of a dataset version that catch common beginner mistakes before training."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .dataset import DatasetVersion

Finding = tuple[str, str]  # (level: "ok" | "info" | "warn" | "block", message)


def dataset_health(version: DatasetVersion) -> list[Finding]:
    m = version.manifest
    names = {str(c["value"]): c["name"] for c in m["class_map"]["classes"]}
    background = str(m["class_map"]["classes"][0]["value"])
    samples = version.samples()
    findings: list[Finding] = []
    by_split: dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}
    images: dict[str, int] = Counter(s["split"] for s in samples)
    for s in samples:
        by_split[s["split"]].update({k: v for k, v in s["class_pixels"].items()})
    for value, name in names.items():
        if value == background:
            continue
        if not by_split["train"][value]:
            findings.append(("block", f"Class '{name}' does not occur in any training image, so the model cannot "
                                      "learn it. Add training images that contain it or change the split."))
        if images["val"] and not by_split["val"][value]:
            findings.append(("warn", f"Class '{name}' is missing from the validation images, so the best epoch is "
                                     "chosen without looking at it."))
        if images["test"] and not by_split["test"][value]:
            findings.append(("warn", f"Class '{name}' is missing from the test images; its test score will be n/a."))
    if images["train"] < 5:
        findings.append(("warn", f"Only {images['train']} training image(s). Use the Standard or Strong augmentation "
                                 "and 'Repeat with seeds' to see how stable the result is."))
    for split, label in (("val", "validation"), ("test", "test")):
        if 0 < images[split] < 2:
            findings.append(("warn", f"The {label} split has a single image; its scores depend on that one picture."))
        if images[split] == 0:
            findings.append(("info", f"There is no {label} split; "
                                     + ("the best epoch is chosen from training loss." if split == "val"
                                        else "results are reported on the validation split instead.")))
    test_groups = {s["group"] for s in samples if s["split"] == "test"}
    if len(test_groups) == 1:
        findings.append(("info", f"All test images come from one group ({next(iter(test_groups))}); scores show how "
                                 "well the model works on that specimen only."))
    total = Counter()
    for s in samples:
        total.update(s["class_pixels"])
    grand = sum(total.values()) or 1
    for value, name in names.items():
        if value != background and total[value] / grand > 0.5:
            findings.append(("warn", f"Class '{name}' covers {100 * total[value] / grand:.0f}% of all pixels - more "
                                     "than the background. Check that the masks are not inverted (feature and "
                                     "background swapped)."))
    if m.get("contains_unapproved_annotations"):
        findings.append(("warn", "Some masks were not approved in the annotation tool; treat results as preliminary."))
    if not [f for f in findings if f[0] in ("warn", "block")]:
        findings.insert(0, ("ok", "No problems found in the dataset."))
    return findings


def summarize(findings: list[Finding]) -> dict[str, Any]:
    return {"block": [m for lv, m in findings if lv == "block"], "warn": [m for lv, m in findings if lv == "warn"],
            "info": [m for lv, m in findings if lv == "info"]}
