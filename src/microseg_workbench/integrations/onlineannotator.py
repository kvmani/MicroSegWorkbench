"""Import OnlineAnnotator ML export bundles (manifest schema ``online-annotator.export/1``).

Contract (from OnlineAnnotator ``services/exports.py``): a ZIP (or its extracted folder) with ``manifest.json``,
images and masks in layout ``hydride_pairs`` (``pairs/<stem>.<ext>`` + ``pairs/<stem>_mask.png``) or
``split_folders`` (``<split>/images/...`` + ``<split>/masks/<stem>_mask.png``). Classes are numbered 1..255 with
0 = background. Mask styles:

* ``indexed`` - pixel value = class index;
* ``binary`` - 255 = ``options.target_class``, 0 = everything else;
* ``red`` - RGB (255, 0, 0) = target class, black = everything else;
* ``colour`` - RGB in each class's colour, black = background.

The import writes a MicroSegWorkbench pair folder whose masks are indexed PNGs holding the *original
OnlineAnnotator class numbers*, a ``classes.json`` and a ``provenance.json`` (who annotated/approved which version,
assigned splits, bundle hash). Every file is checked against the SHA-256 in the manifest and the decoded class
pixel counts are checked against the manifest's counts; any mismatch aborts the import. Nothing is guessed:
unknown colours or values are errors.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .. import TOOL_ID, __version__
from ..core.atomic import publish_dir, staging_dir, write_bytes_atomic, write_json_atomic
from ..core.classes import CLASSES_FILENAME, ClassMap, ClassMapError, SegClass
from ..core.imageio import MASK_SUFFIX, encode_png

OA_SCHEMA = "online-annotator.export/1"
PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_SCHEMA = "microseg-workbench.provenance/1"
SUPPORTED_STYLES = ("indexed", "binary", "red", "colour")


class OAImportError(ValueError):
    """Bundle cannot be imported; message is safe to show to users."""





@dataclass
class ImportReport:
    folder: Path
    images: int
    class_map: ClassMap
    mask_style: str
    unapproved: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class _Source:
    """Uniform read access to a ZIP bundle or an extracted folder."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.zip: zipfile.ZipFile | None = None
        if self.path.is_file():
            try:
                self.zip = zipfile.ZipFile(self.path)
            except zipfile.BadZipFile as exc:
                raise OAImportError(f"{self.path.name} is not a valid ZIP file.") from exc
        elif not self.path.is_dir():
            raise OAImportError(f"{self.path} does not exist.")

    def read(self, name: str) -> bytes:
        if ".." in Path(name).parts or name.startswith(("/", "\\")):
            raise OAImportError(f"Unsafe path in manifest: {name!r}")
        try:
            if self.zip is not None:
                return self.zip.read(name)
            return (self.path / name).read_bytes()
        except (KeyError, FileNotFoundError) as exc:
            raise OAImportError(f"File {name} listed in the manifest is missing from the bundle.") from exc

    def sha256(self) -> str | None:
        if self.zip is None:
            return None
        h = hashlib.sha256()
        with open(self.path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()


def _hex_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def build_class_map(manifest: dict[str, Any]) -> tuple[ClassMap, dict[int, tuple[int, int, int]]]:
    style = manifest["options"]["mask_style"]
    classes = sorted(manifest["classes"], key=lambda c: c["index"])
    if not classes:
        raise OAImportError("The export defines no classes.")
    names = {c["name"] for c in classes}
    bg_name = "background" if "background" not in names else "background (unlabelled)"
    if style in ("binary", "red"):
        target = manifest["options"].get("target_class")
        chosen = [c for c in classes if c["index"] == target]
        if not chosen:
            raise OAImportError(f"Target class {target} of the binary export is not among the classes.")
        classes = chosen
    try:
        class_map = ClassMap(tuple([SegClass(0, bg_name, "#000000")] +
                                   [SegClass(int(c["index"]), str(c["name"]), str(c["color"]).lower())
                                    for c in classes]), mask_encoding="indexed")
    except ClassMapError as exc:
        raise OAImportError(f"Class definitions cannot be used: {exc}") from exc
    colours = {int(c["index"]): _hex_rgb(c["color"]) for c in classes}
    return class_map, colours


def decode_oa_mask(data: bytes, style: str, target: int | None, colours: dict[int, tuple[int, int, int]],
                   allowed: set[int]) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    img.load()
    if style in ("indexed", "binary"):
        if img.mode not in ("L", "P", "1"):
            raise OAImportError(f"expected an 8-bit single-channel mask, got mode {img.mode}")
        arr = np.asarray(img, dtype=np.uint8) * (255 if img.mode == "1" else 1)
        if style == "binary":
            bad = sorted(set(np.unique(arr).tolist()) - {0, 255})
            if bad:
                raise OAImportError(f"binary mask contains values {bad[:5]} other than 0/255")
            return np.where(arr == 255, target, 0).astype(np.uint8)
        bad = sorted(set(np.unique(arr).tolist()) - allowed)
        if bad:
            raise OAImportError(f"mask contains class numbers {bad[:5]} not defined in the export")
        return arr
    rgb = np.asarray(img.convert("RGB") if img.mode in ("RGB", "P") else img, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise OAImportError(f"expected an RGB mask, got mode {img.mode}")
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    known = np.all(rgb == 0, axis=2)
    palette = {target: (255, 0, 0)} if style == "red" else colours
    for index, colour in palette.items():
        sel = np.all(rgb == np.array(colour, np.uint8), axis=2)
        out[sel] = index
        known |= sel
    if not known.all():
        ys, xs = np.nonzero(~known)
        raise OAImportError(f"mask has colour {tuple(rgb[ys[0], xs[0]])} at ({ys[0]}, {xs[0]}) that matches no class")
    return out


_BAD_STEM_CHARS = set('\\/:*?"<>|') | {chr(c) for c in range(32)}


def safe_stem(stem: object) -> str:
    """A stem is used as a file name: refuse separators, traversal and other unsafe characters."""
    if not isinstance(stem, str) or not stem.strip() or stem in (".", "..") or stem.startswith(".") \
            or any(ch in _BAD_STEM_CHARS for ch in stem) or stem != stem.strip():
        raise OAImportError(f"Image name {stem!r} in the export cannot be used as a file name.")
    return stem


def import_bundle(source: Path, dest: Path, progress: Callable[[str], None] | None = None) -> ImportReport:
    """Convert an OnlineAnnotator export into a MicroSegWorkbench pair folder at ``dest`` (must not exist)."""
    dest = Path(dest)
    if dest.exists():
        raise OAImportError(f"Destination {dest} already exists; choose a new folder.")
    src = _Source(Path(source))
    try:
        try:
            manifest = json.loads(src.read("manifest.json"))
        except ValueError as exc:
            raise OAImportError("manifest.json is not valid JSON.") from exc
        if manifest.get("schema") != OA_SCHEMA:
            raise OAImportError(f"Unsupported manifest schema {manifest.get('schema')!r}; expected {OA_SCHEMA!r}.")
        options = manifest.get("options", {})
        style = options.get("mask_style")
        if style not in SUPPORTED_STYLES:
            raise OAImportError(f"Unsupported mask style {style!r}.")
        if manifest.get("background_value", 0) != 0:
            raise OAImportError("Only exports with background value 0 are supported.")
        class_map, colours = build_class_map(manifest)
        target = options.get("target_class")
        allowed = set(class_map.values)
        stage = staging_dir(dest.parent, dest.name)
        records: dict[str, Any] = {}
        warnings: list[str] = []
        try:
            seen: set[str] = set()
            for i, rec in enumerate(manifest.get("images", []), 1):
                stem = safe_stem(rec.get("stem"))
                if stem.lower() in seen:
                    raise OAImportError(f"Image name {stem!r} occurs twice in the export.")
                seen.add(stem.lower())
                if progress:
                    progress(f"Importing {i}/{len(manifest['images'])}: {stem}")
                image_bytes, mask_bytes = src.read(rec["image_file"]), src.read(rec["mask_file"])
                if hashlib.sha256(image_bytes).hexdigest() != rec["image_sha256"]:
                    raise OAImportError(f"{rec['image_file']}: file does not match its SHA-256 in the manifest.")
                if hashlib.sha256(mask_bytes).hexdigest() != rec["mask_png_sha256"]:
                    raise OAImportError(f"{rec['mask_file']}: file does not match its SHA-256 in the manifest.")
                try:
                    labels = decode_oa_mask(mask_bytes, style, target, colours, allowed)
                except OAImportError as exc:
                    raise OAImportError(f"{rec['mask_file']}: {exc}") from exc
                counts = np.bincount(labels.ravel(), minlength=256)
                expected = {int(k): int(v) for k, v in (rec.get("class_pixels") or {}).items()}
                for index, n in expected.items():
                    if style in ("binary", "red") and index != target:
                        continue
                    if index in allowed and index != 0 and counts[index] != n:
                        raise OAImportError(f"{stem}: class {index} has {counts[index]} pixels but the manifest "
                                            f"records {n}; the export is inconsistent.")
                ext = Path(rec["image_file"]).suffix.lower()
                if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                    raise OAImportError(f"{rec['image_file']}: unsupported image type {ext!r}.")
                write_bytes_atomic(stage / f"{stem}{ext}", image_bytes)
                write_bytes_atomic(stage / f"{stem}{MASK_SUFFIX}.png", encode_png(labels))
                records[stem] = {k: rec.get(k) for k in (
                    "original_filename", "split", "version", "version_status", "version_kind", "mask_source",
                    "mask_source_tool", "annotated_by", "annotated_at", "contributors", "approved_by",
                    "approved_at", "image_sha256", "original_sha256", "mask_png_sha256", "label_sha256")}
            if not records:
                raise OAImportError("The export contains no images.")
            unapproved = sorted(s for s, r in records.items() if r.get("version_status") != "approved")
            if unapproved:
                warnings.append(f"{len(unapproved)} mask(s) are submitted but NOT approved in OnlineAnnotator; use "
                                "them for experiments only, not as final ground truth.")
            payload = class_map.to_dict()
            payload["notes"] = [f"Imported from OnlineAnnotator export ({style} masks); mask values are the "
                                "OnlineAnnotator class numbers; 0 = background."]
            write_json_atomic(stage / CLASSES_FILENAME, payload)
            write_json_atomic(stage / PROVENANCE_FILENAME, {
                "schema": PROVENANCE_SCHEMA,
                "source": {"tool": manifest.get("tool"), "schema": OA_SCHEMA, "bundle": str(source),
                           "bundle_sha256": src.sha256(), "project": manifest.get("project"),
                           "exported_at": manifest.get("created_at"), "exported_by": manifest.get("created_by"),
                           "mask_style": style, "layout": options.get("layout"), "options": options},
                "imported_with": f"{TOOL_ID} {__version__}",
                "unapproved": unapproved,
                "samples": records,
            })
            publish_dir(stage, dest)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        splits: dict[str, int] = {}
        for r in records.values():
            splits[r.get("split") or "unassigned"] = splits.get(r.get("split") or "unassigned", 0) + 1
        return ImportReport(folder=dest, images=len(records), class_map=class_map, mask_style=style,
                            unapproved=unapproved, splits=splits, warnings=warnings)
    finally:
        src.close()


def read_provenance(folder: Path) -> dict[str, Any] | None:
    path = Path(folder) / PROVENANCE_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("schema") == PROVENANCE_SCHEMA else None
