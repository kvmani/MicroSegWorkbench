"""Class maps: the exact mapping between mask pixel values and scientific classes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic import read_json

CLASSES_SCHEMA = "microseg-workbench.classes/1"
CLASSES_FILENAME = "classes.json"
MASK_ENCODINGS = ("indexed", "binary_0_255")
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
DEFAULT_COLORS = ["#000000", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6"]


class ClassMapError(ValueError):
    """Invalid class map; the message is safe to show to users."""


@dataclass(frozen=True)
class SegClass:
    value: int
    name: str
    color: str

    def rgb(self) -> tuple[int, int, int]:
        return (int(self.color[1:3], 16), int(self.color[3:5], 16), int(self.color[5:7], 16))


@dataclass(frozen=True)
class ClassMap:
    """Ordered classes. ``classes[i]`` is model output channel ``i``.

    ``mask_encoding`` describes the *source* masks:

    * ``indexed`` - pixel value equals the class ``value``.
    * ``binary_0_255`` - exactly two classes with values 0 and 1; source pixel 255
      means class 1. Any other source value is an error (never silently thresholded).
    """

    classes: tuple[SegClass, ...]
    mask_encoding: str = "indexed"
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.mask_encoding not in MASK_ENCODINGS:
            raise ClassMapError(f"Unknown mask encoding {self.mask_encoding!r}; use one of {MASK_ENCODINGS}.")
        if len(self.classes) < 2:
            raise ClassMapError("A class map needs at least two classes (background and one feature).")
        values = [c.value for c in self.classes]
        if len(set(values)) != len(values):
            raise ClassMapError(f"Duplicate class values in class map: {values}.")
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ClassMapError(f"Duplicate class names in class map: {names}.")
        for c in self.classes:
            if not 0 <= c.value <= 255:
                raise ClassMapError(f"Class value {c.value} for {c.name!r} must be within 0..255 (8-bit masks).")
            if not c.name.strip():
                raise ClassMapError("Class names must not be empty.")
            if not _COLOR_RE.match(c.color):
                raise ClassMapError(f"Class colour {c.color!r} for {c.name!r} must look like #RRGGBB.")
        if values != sorted(values):
            raise ClassMapError("Classes must be listed in increasing value order.")
        if self.mask_encoding == "binary_0_255" and values != [0, 1]:
            raise ClassMapError("Binary 0/255 masks need exactly two classes with values 0 and 1.")

    @property
    def values(self) -> list[int]:
        return [c.value for c in self.classes]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CLASSES_SCHEMA,
            "mask_encoding": self.mask_encoding,
            "classes": [{"value": c.value, "name": c.name, "color": c.color} for c in self.classes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClassMap":
        schema = payload.get("schema")
        if schema != CLASSES_SCHEMA:
            raise ClassMapError(f"Unsupported class map schema {schema!r}; expected {CLASSES_SCHEMA!r}.")
        raw = payload.get("classes")
        if not isinstance(raw, list):
            raise ClassMapError("Class map must contain a 'classes' list.")
        classes = []
        for i, item in enumerate(raw):
            try:
                value = item["value"]
                name = str(item["name"])
            except (KeyError, TypeError) as exc:
                raise ClassMapError(f"Class entry {i} needs 'value' and 'name'.") from exc
            if not isinstance(value, int) or isinstance(value, bool):
                raise ClassMapError(f"Class value for {name!r} must be an integer, got {value!r}.")
            classes.append(SegClass(value=value, name=name, color=str(item.get("color", DEFAULT_COLORS[i % 8]))))
        return cls(classes=tuple(classes), mask_encoding=str(payload.get("mask_encoding", "indexed")))

    @classmethod
    def load(cls, path: Path) -> "ClassMap":
        try:
            payload = read_json(path)
        except FileNotFoundError as exc:
            raise ClassMapError(
                f"No {CLASSES_FILENAME} found in {path.parent}. Every dataset folder needs one so that "
                "mask pixel values map to named classes explicitly."
            ) from exc
        except ValueError as exc:
            raise ClassMapError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)
