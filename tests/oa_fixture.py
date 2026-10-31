"""Build OnlineAnnotator-style export bundles (following OnlineAnnotator services/exports.py) for tests."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from microseg_workbench.core.synthetic import SyntheticSpec, generate_sample

CLASSES = [{"index": 2, "name": "hydride", "color": "#E6194B", "description": ""},
           {"index": 5, "name": "pore", "color": "#4363D8", "description": ""}]


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def make_bundle(path: Path, style: str = "indexed", layout: str = "hydride_pairs", specimens: int = 4,
                per: int = 2, unapproved: tuple[str, ...] = (), splits: dict[str, str] | None = None) -> Path:
    spec = SyntheticSpec(specimens=specimens, images_per_specimen=per, size=48, seed=11)
    records = []
    with zipfile.ZipFile(path, "w") as zf:
        for s in range(specimens):
            for i in range(per):
                image, mask = generate_sample(spec, s, i)
                labels = np.select([mask == 1, mask == 2], [2, 5], 0).astype(np.uint8)  # OA class numbers
                stem = f"S{s + 1}_{i + 1}"
                split = (splits or {}).get(stem, "unassigned")
                if style == "indexed":
                    m = labels
                elif style == "binary":
                    m = np.where(labels == 2, 255, 0).astype(np.uint8)
                elif style == "red":
                    m = np.zeros((*labels.shape, 3), np.uint8)
                    m[labels == 2] = (255, 0, 0)
                else:
                    m = np.zeros((*labels.shape, 3), np.uint8)
                    m[labels == 2] = (0xE6, 0x19, 0x4B)
                    m[labels == 5] = (0x43, 0x63, 0xD8)
                img_b, mask_b = _png(image), _png(m)
                folder = "pairs" if layout == "hydride_pairs" else split
                img_arc = f"{folder}/{stem}.png" if layout == "hydride_pairs" else f"{folder}/images/{stem}.png"
                mask_arc = (f"{folder}/{stem}_mask.png" if layout == "hydride_pairs"
                            else f"{folder}/masks/{stem}_mask.png")
                zf.writestr(img_arc, img_b)
                zf.writestr(mask_arc, mask_b)
                counts = {str(k): int((labels == k).sum()) for k in (2, 5) if (labels == k).any()}
                records.append({
                    "stem": stem, "original_filename": f"{stem}.tif", "image_file": img_arc, "mask_file": mask_arc,
                    "split": split, "width": 48, "height": 48,
                    "image_sha256": hashlib.sha256(img_b).hexdigest(), "original_sha256": "0" * 64,
                    "mask_png_sha256": hashlib.sha256(mask_b).hexdigest(), "label_sha256": "1" * 64,
                    "version": 3, "version_status": "submitted" if stem in unapproved else "approved",
                    "version_kind": "annotation", "mask_source": "manual", "annotated_by": "ann@lab",
                    "annotated_at": "2026-10-01T00:00:00+00:00", "contributors": ["ann@lab"],
                    "approved_by": None if stem in unapproved else "rev@lab",
                    "approved_at": None if stem in unapproved else "2026-10-02T00:00:00+00:00",
                    "class_pixels": counts,
                })
        manifest = {
            "schema": "online-annotator.export/1", "tool": {"id": "online-annotator", "version": "1.4.0"},
            "project": {"id": 1, "name": "Zr hydrides", "description": ""},
            "created_at": "2026-10-03T00:00:00+00:00", "created_by": "rev@lab",
            "options": {"layout": layout, "mask_style": style, "target_class": 2 if style in ("binary", "red") else None,
                        "include": "approved", "split_mode": "assigned"},
            "mask_encoding": style, "classes": CLASSES, "background_value": 0,
            "summary": {"images": len(records)}, "images": records,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return path
