"""Discovering, validating and versioning image/mask pair datasets.

Input: a folder of ``<stem>.<ext>`` micrographs, ``<stem>_mask.png`` masks and a
``classes.json`` class map (this matches OnlineAnnotator's ``hydride_pairs``
export naming). Output: an immutable, content-addressed *dataset version*
directory with copied images, canonical indexed masks, a fixed split and a
manifest recording every hash and decision.
"""

from __future__ import annotations

import platform
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import TOOL_ID, __version__
from .atomic import (
    canonical_json,
    publish_dir,
    read_json,
    remove_stale_staging,
    sha256_bytes,
    sha256_file,
    staging_dir,
    write_bytes_atomic,
    write_json_atomic,
)
from .classes import CLASSES_FILENAME, ClassMap, ClassMapError
from .imageio import (
    IMAGE_EXTENSIONS,
    MASK_SUFFIX,
    ImageReadError,
    decode_mask,
    encode_png,
    load_image,
    load_mask_values,
)
from .splits import SplitConfig, SplitError, assign_splits, check_fixed_splits, derive_group

MANIFEST_SCHEMA = "microseg-workbench.dataset/1"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" blocks dataset creation; "warning" is shown but allowed
    code: str
    message: str
    stem: str = ""


@dataclass
class PairRecord:
    stem: str
    image_path: Path
    mask_path: Path
    group: str = ""
    height: int = 0
    width: int = 0
    channels: int = 0
    dtype: str = ""
    image_sha256: str = ""
    mask_sha256: str = ""
    class_pixels: dict[int, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    ok: bool = False


@dataclass
class ValidationReport:
    folder: Path
    class_map: ClassMap | None
    pairs: list[PairRecord]
    issues: list[Issue]
    group_regex: str
    provenance: dict[str, Any] | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors and self.class_map is not None and any(p.ok for p in self.pairs)

    def summary(self) -> str:
        valid = sum(p.ok for p in self.pairs)
        groups = len({p.group for p in self.pairs if p.ok})
        return (f"{valid}/{len(self.pairs)} valid pairs in {groups} group(s); "
                f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")


def discover_pairs(folder: Path) -> tuple[list[PairRecord], list[Issue]]:
    """Pair ``<stem>.<ext>`` with ``<stem>_mask.png``; report orphans and ambiguity."""
    issues: list[Issue] = []
    if not folder.is_dir():
        return [], [Issue("error", "folder_missing", f"Folder {folder} does not exist.")]
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    masks: dict[str, Path] = {}
    images: dict[str, list[Path]] = {}
    for path in files:
        if path.stem.endswith(MASK_SUFFIX):
            stem = path.stem[: -len(MASK_SUFFIX)]
            if path.suffix.lower() != ".png":
                issues.append(Issue("error", "mask_not_png",
                                    f"Mask {path.name} must be a lossless PNG (JPEG would alter class values).", stem))
                continue
            masks[stem] = path
        else:
            images.setdefault(path.stem, []).append(path)
    pairs: list[PairRecord] = []
    for stem in sorted(set(images) | set(masks)):
        candidates = images.get(stem, [])
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            issues.append(Issue("error", "ambiguous_image", f"Several images share stem {stem!r}: {names}.", stem))
            continue
        if not candidates:
            issues.append(Issue("error", "mask_without_image", f"Mask for {stem!r} has no matching image.", stem))
            continue
        if stem not in masks:
            issues.append(Issue("error", "image_without_mask",
                                f"Image {candidates[0].name} has no mask named {stem}{MASK_SUFFIX}.png.", stem))
            continue
        pairs.append(PairRecord(stem=stem, image_path=candidates[0], mask_path=masks[stem]))
    return pairs, issues


def validate_folder(folder: Path, group_regex: str) -> ValidationReport:
    """Check class map, pairing, shapes, mask values, channel consistency and duplicates."""
    folder = Path(folder)
    issues: list[Issue] = []
    class_map: ClassMap | None
    try:
        class_map = ClassMap.load(folder / CLASSES_FILENAME)
    except ClassMapError as exc:
        class_map = None
        issues.append(Issue("error", "class_map", str(exc)))
    pairs, pair_issues = discover_pairs(folder)
    issues.extend(pair_issues)
    if not pairs and folder.is_dir():
        issues.append(Issue("error", "no_pairs", f"No image/mask pairs found in {folder}."))

    try:
        for pair in pairs:
            pair.group = derive_group(pair.stem, group_regex)
    except SplitError as exc:
        issues.append(Issue("error", "group_regex", str(exc)))

    for pair in pairs:
        try:
            img = load_image(pair.image_path)
            raw = load_mask_values(pair.mask_path)
        except ImageReadError as exc:
            issues.append(Issue("error", "unreadable", str(exc), pair.stem))
            continue
        pair.height, pair.width = img.array.shape[:2]
        pair.channels = img.channels
        pair.dtype = str(img.array.dtype)
        pair.notes = img.notes
        pair.image_sha256 = sha256_file(pair.image_path)
        pair.mask_sha256 = sha256_file(pair.mask_path)
        if raw.shape != img.array.shape[:2]:
            issues.append(Issue("error", "shape_mismatch",
                                f"Image is {pair.height}x{pair.width} but mask is {raw.shape[0]}x{raw.shape[1]} "
                                "(height x width). Masks must align pixel-for-pixel.", pair.stem))
            continue
        if img.array.dtype == np.float32:
            issues.append(Issue("warning", "float_image",
                                "Floating-point image is used without intensity scaling.", pair.stem))
        if class_map is None:
            continue
        try:
            labels = decode_mask(raw, class_map)
        except ImageReadError as exc:
            issues.append(Issue("error", "mask_values", f"{pair.mask_path.name}: {exc}", pair.stem))
            continue
        counts = np.bincount(labels.ravel(), minlength=256)
        pair.class_pixels = {v: int(counts[v]) for v in class_map.values}
        if all(pair.class_pixels[v] == 0 for v in class_map.values[1:]):
            issues.append(Issue("warning", "background_only", "Mask contains background only.", pair.stem))
        pair.ok = True

    loaded = [p for p in pairs if p.ok]
    channel_counts = Counter(p.channels for p in loaded)
    if len(channel_counts) > 1:
        issues.append(Issue("error", "mixed_channels",
                            f"Images mix channel counts {dict(channel_counts)}; use all grayscale or all RGB."))
    dtypes = Counter(p.dtype for p in loaded)
    if len(dtypes) > 1:
        issues.append(Issue("error", "mixed_bit_depth",
                            f"Images mix pixel types {dict(dtypes)}; convert them to one bit depth first."))
    by_hash: dict[str, list[str]] = {}
    for p in loaded:
        by_hash.setdefault(p.image_sha256, []).append(p.stem)
    for stems in by_hash.values():
        if len(stems) > 1:
            issues.append(Issue("error", "duplicate_image",
                                f"Identical image files: {', '.join(stems)}. Duplicates can leak between splits."))
    if class_map is not None and loaded:
        totals = {v: sum(p.class_pixels.get(v, 0) for p in loaded) for v in class_map.values}
        for c in class_map.classes:
            if totals[c.value] == 0:
                issues.append(Issue("warning", "class_absent",
                                    f"Class {c.value} ({c.name}) does not occur in any mask."))
    groups = Counter(p.group for p in loaded)
    if loaded and len(groups) == len(loaded) and len(loaded) > 3:
        issues.append(Issue("warning", "singleton_groups",
                            "Every image is its own group. If several images come from the same specimen, "
                            "adjust the grouping pattern so they stay in the same split."))
    provenance = None
    try:
        from ..integrations.onlineannotator import read_provenance

        provenance = read_provenance(folder)
    except ValueError as exc:
        issues.append(Issue("error", "provenance", f"provenance.json is unreadable: {exc}"))
    if provenance:
        unapproved = set(provenance.get("unapproved", [])) & {p.stem for p in loaded}
        if unapproved:
            issues.append(Issue("warning", "unapproved_annotations",
                                f"{len(unapproved)} mask(s) were not approved in the annotation tool "
                                f"(e.g. {sorted(unapproved)[:3]}). Results will be marked as using unapproved "
                                "ground truth."))
        missing = sorted({p.stem for p in loaded} - set(provenance.get("samples", {})))
        if missing:
            issues.append(Issue("warning", "provenance_missing",
                                f"{len(missing)} image(s) have no provenance record (e.g. {missing[:3]})."))
    return ValidationReport(folder=folder, class_map=class_map, pairs=pairs, issues=issues, group_regex=group_regex,
                            provenance=provenance)


def plan_split(report: ValidationReport, config: SplitConfig) -> dict[str, str]:
    """Stem -> split for all valid pairs."""
    if config.source == "annotator":
        samples = (report.provenance or {}).get("samples", {})
        if not samples:
            raise SplitError("This folder has no split assignments from an annotation tool; use the group split.")
        assignment = {p.stem: (samples.get(p.stem) or {}).get("split") or "unassigned" for p in report.pairs if p.ok}
        check_fixed_splits(assignment, {p.stem: p.group for p in report.pairs if p.ok})
        return assignment
    sizes = Counter(p.group for p in report.pairs if p.ok)
    by_group = assign_splits(dict(sizes), config)
    return {p.stem: by_group[p.group] for p in report.pairs if p.ok}


@dataclass(frozen=True)
class DatasetVersion:
    """A published, immutable dataset version on disk."""

    root: Path
    manifest: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.manifest["dataset_id"])

    @property
    def class_map(self) -> ClassMap:
        payload = dict(self.manifest["class_map"])
        payload["mask_encoding"] = "indexed"  # stored masks are always canonical indexed PNGs
        return ClassMap.from_dict(payload)

    def samples(self, split: str | None = None) -> list[dict[str, Any]]:
        return [s for s in self.manifest["samples"] if split is None or s["split"] == split]

    def image_path(self, sample: dict[str, Any]) -> Path:
        return self.root / sample["image_file"]

    def mask_path(self, sample: dict[str, Any]) -> Path:
        return self.root / sample["mask_file"]

    @classmethod
    def open(cls, root: Path) -> "DatasetVersion":
        manifest = read_json(Path(root) / MANIFEST_FILENAME)
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"{root} is not a MicroSegWorkbench dataset version.")
        return cls(root=Path(root), manifest=manifest)

    def verify(self) -> list[str]:
        """Re-hash every file; returns a list of problems (empty when intact)."""
        problems = []
        for s in self.samples():
            for key, hkey in (("image_file", "image_sha256"), ("mask_file", "stored_mask_sha256")):
                path = self.root / s[key]
                if not path.exists():
                    problems.append(f"missing {s[key]}")
                elif sha256_file(path) != s[hkey]:
                    problems.append(f"hash mismatch {s[key]}")
        return problems


def create_dataset_version(report: ValidationReport, split_config: SplitConfig, datasets_dir: Path,
                           name: str) -> DatasetVersion:
    """Copy validated pairs into a new immutable dataset version (atomic publication).

    The ``dataset_id`` is a hash of the scientific content (file hashes, class map,
    split assignment, preparation options), so preparing identical inputs twice
    yields the same id and reuses the existing version.
    """
    if not report.ok or report.class_map is None:
        raise ValueError("Dataset has validation errors; fix them before creating a version.")
    if not name.strip() or any(ch in name for ch in '\\/:*?"<>|'):
        raise ValueError("Dataset name must be non-empty and contain no path characters.")
    if split_config.group_regex != report.group_regex:
        raise ValueError("Split grouping pattern differs from the one used during validation; validate again.")
    split = plan_split(report, split_config)
    class_map = report.class_map
    remove_stale_staging(datasets_dir)
    stage = staging_dir(datasets_dir, name)
    try:
        (stage / "images").mkdir()
        (stage / "masks").mkdir()
        samples = []
        for pair in sorted((p for p in report.pairs if p.ok), key=lambda p: p.stem):
            ext = pair.image_path.suffix.lower()
            image_rel = f"images/{pair.stem}{ext}"
            mask_rel = f"masks/{pair.stem}{MASK_SUFFIX}.png"
            shutil.copyfile(pair.image_path, stage / image_rel)
            if sha256_file(stage / image_rel) != pair.image_sha256:
                raise OSError(f"Image {pair.image_path} changed while preparing the dataset.")
            labels = decode_mask(load_mask_values(pair.mask_path), class_map)
            mask_bytes = encode_png(labels)
            write_bytes_atomic(stage / mask_rel, mask_bytes)
            samples.append({
                "stem": pair.stem,
                "group": pair.group,
                "split": split[pair.stem],
                "image_file": image_rel,
                "mask_file": mask_rel,
                "height": pair.height,
                "width": pair.width,
                "channels": pair.channels,
                "dtype": pair.dtype,
                "image_sha256": pair.image_sha256,
                "source_mask_sha256": pair.mask_sha256,
                "stored_mask_sha256": sha256_bytes(mask_bytes),
                "class_pixels": {str(k): v for k, v in pair.class_pixels.items()},
                "load_notes": list(pair.notes),
                "source_image": str(pair.image_path),
                "source_mask": str(pair.mask_path),
                "annotation_provenance": ((report.provenance or {}).get("samples", {}).get(pair.stem)),
            })
        split_identity = asdict(split_config)
        if split_config.source == "group":
            split_identity.pop("source")  # keeps ids of group-split versions identical to earlier releases
        content = {
            "class_map": class_map.to_dict(),
            "split_config": split_identity,
            "samples": [{k: s[k] for k in ("stem", "group", "split", "image_sha256", "stored_mask_sha256")}
                        for s in samples],
        }
        dataset_id = f"{name}-{sha256_bytes(canonical_json(content).encode())[:12]}"
        counts = Counter(s["split"] for s in samples)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "dataset_id": dataset_id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_with": {"tool": TOOL_ID, "version": __version__, "python": platform.python_version()},
            "source_folder": str(report.folder),
            "class_map": class_map.to_dict(),
            "stored_mask_encoding": "indexed",
            "split_config": asdict(split_config),
            "split_counts": {k: counts.get(k, 0) for k in ("train", "val", "test")},
            "groups": {g: sorted({s["split"] for s in samples if s["group"] == g}) for g in
                       sorted({s["group"] for s in samples})},
            "warnings": [asdict(i) for i in report.warnings],
            "annotation_source": (report.provenance or {}).get("source"),
            "contains_unapproved_annotations": bool(set((report.provenance or {}).get("unapproved", []))
                                                    & {s["stem"] for s in samples}),
            "samples": samples,
        }
        write_json_atomic(stage / MANIFEST_FILENAME, manifest)
        final = datasets_dir / dataset_id
        publish_dir(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return DatasetVersion.open(final)


def list_dataset_versions(datasets_dir: Path) -> list[DatasetVersion]:
    out = []
    if datasets_dir.exists():
        for child in sorted(datasets_dir.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and (child / MANIFEST_FILENAME).exists():
                try:
                    out.append(DatasetVersion.open(child))
                except (ValueError, OSError):
                    continue
    return out
