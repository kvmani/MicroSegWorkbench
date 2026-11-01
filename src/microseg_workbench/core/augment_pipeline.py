"""The augmentation pipeline as data: order, presets, single-step previews, recipes and a sanity check.

``PIPELINE`` lists the steps in the exact order :func:`microseg_workbench.core.augment.augment` applies them, so the
visual builder can never show an order that differs from training. Geometry steps move image and mask together;
intensity steps change the image only and run as one group with a shared probability.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .augment import IGNORE_INDEX, PHOTOMETRIC_OPS, AugmentConfig, _rotate, augment, gaussian_blur


@dataclass(frozen=True)
class Step:
    id: str  # trace id used by augment()
    title: str
    kind: str  # "geometry" (image + mask) | "intensity" (image only) | "crop"
    help: str
    prob_field: str | None = None  # own probability (geometry steps)
    strength_field: str | None = None
    strength_range: tuple[float, float, float] = (0.0, 1.0, 0.01)  # min, max, step
    strength_unit: str = ""


PIPELINE: tuple[Step, ...] = (
    Step("rotate", "Small rotation", "geometry",
         "Turns the patch by a random angle up to the limit. Corners that would come from outside the image are "
         "ignored during training, never called background.", "rotate_prob", "rotate_max_deg", (0, 90, 1), "deg"),
    Step("crop", "Cut training patch", "crop",
         "Cuts a square patch at a random position; this is what the model sees in each training step.",
         None, "patch_size", (16, 1024, 8), "px"),
    Step("hflip", "Mirror left-right", "geometry", "Mirrors the patch horizontally.", "hflip"),
    Step("vflip", "Mirror up-down", "geometry", "Mirrors the patch vertically.", "vflip"),
    Step("rot90", "Quarter turns", "geometry", "Rotates by 90, 180 or 270 degrees.", "rot90"),
    Step("contrast", "Contrast", "intensity", "Stretches or flattens grey levels around the mean.",
         None, "contrast", (0, 0.9, 0.01), ""),
    Step("brightness", "Brightness", "intensity", "Makes the whole patch lighter or darker.",
         None, "brightness", (0, 0.5, 0.01), ""),
    Step("gamma", "Gamma", "intensity", "Bends the grey scale: brightens dark areas or darkens bright ones.",
         None, "gamma", (0, 1, 0.05), ""),
    Step("noise_std", "Noise", "intensity", "Adds random grain like a noisy detector.",
         None, "noise_std", (0, 0.3, 0.005), ""),
    Step("blur_max_sigma", "Blur", "intensity", "Softens the image like slightly defocused optics.",
         None, "blur_max_sigma", (0, 5, 0.1), "px"),
)
STEPS = {s.id: s for s in PIPELINE}
INTENSITY_GROUP_FIELD = "photometric_prob"

PRESETS: dict[str, dict[str, Any]] = {
    "none": dict(hflip=0.0, vflip=0.0, rot90=0.0, rotate_prob=0.0, photometric_prob=0.0, brightness=0.0,
                 contrast=0.0, gamma=0.0, noise_std=0.0, blur_max_sigma=0.0),
    "gentle": dict(hflip=0.5, vflip=0.5, rot90=0.0, rotate_prob=0.0, photometric_prob=0.5, brightness=0.04,
                   contrast=0.08, gamma=0.0, noise_std=0.0, blur_max_sigma=0.0),
    "standard": dict(hflip=0.5, vflip=0.5, rot90=0.5, rotate_prob=0.3, rotate_max_deg=15.0, photometric_prob=0.8,
                     brightness=0.08, contrast=0.15, gamma=0.0, noise_std=0.0, blur_max_sigma=0.0),
    "strong": dict(hflip=0.5, vflip=0.5, rot90=0.5, rotate_prob=0.5, rotate_max_deg=30.0, photometric_prob=0.9,
                   brightness=0.12, contrast=0.25, gamma=0.2, noise_std=0.02, blur_max_sigma=1.0),
}
PRESET_HELP = {
    "none": "No augmentation - the model only sees original patches. Useful as a baseline.",
    "gentle": "Mirrors and mild brightness/contrast changes. Safe for almost any data.",
    "standard": "Mirrors, quarter turns, small rotations and moderate brightness/contrast. A good default.",
    "strong": "Adds larger rotations, gamma, noise and blur. For small datasets or very variable imaging.",
}


def apply_preset(cfg: AugmentConfig, name: str) -> AugmentConfig:
    """Preset values; patch size and plugin operations are kept."""
    return replace(cfg, **PRESETS[name])


def matching_preset(cfg: AugmentConfig) -> str | None:
    for name, values in PRESETS.items():
        if all(abs(getattr(cfg, k) - v) < 1e-9 for k, v in values.items() if k != "rotate_max_deg" or cfg.rotate_prob):
            return name
    return None


def step_enabled(cfg: AugmentConfig, step: Step) -> bool:
    if step.kind == "crop":
        return True
    if step.prob_field:
        return getattr(cfg, step.prob_field) > 0 and (step.id != "rotate" or cfg.rotate_max_deg > 0)
    return getattr(cfg, INTENSITY_GROUP_FIELD) > 0 and getattr(cfg, step.strength_field) > 0


def expected_rate(cfg: AugmentConfig, step: Step) -> float:
    """Fraction of training patches on which the step is applied."""
    if not step_enabled(cfg, step):
        return 0.0
    if step.kind == "crop":
        return 1.0
    if step.prob_field:
        return float(getattr(cfg, step.prob_field))
    return float(getattr(cfg, INTENSITY_GROUP_FIELD))


def preview_step(image: np.ndarray, labels: np.ndarray, step_id: str, value: float, sign: int = 1,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One step alone at a given strength (``sign`` -1/+1 picks the weak/strong end), for thumbnails."""
    if step_id == "rotate":
        return _rotate(image, labels, sign * value, 2)  # bilinear image, nearest labels
    if step_id == "hflip":
        return image[:, ::-1], labels[:, ::-1]
    if step_id == "vflip":
        return image[::-1], labels[::-1]
    if step_id == "rot90":
        k = 1 if sign > 0 else 3
        return np.rot90(image, k, axes=(0, 1)), np.rot90(labels, k, axes=(0, 1))
    if step_id == "contrast":
        f = 1.0 + sign * value
        out = (image - float(image.mean())) * f + float(image.mean())
    elif step_id == "brightness":
        out = image + sign * value
    elif step_id == "gamma":
        out = np.clip(image, 0, 1) ** float(np.exp(sign * value))
    elif step_id == "noise_std":
        out = image + np.random.default_rng(seed).normal(0, value if sign > 0 else 0.0, image.shape)
    elif step_id == "blur_max_sigma":
        out = gaussian_blur(image, value) if sign > 0 else image
    elif step_id.startswith("extra:"):
        spec = PHOTOMETRIC_OPS[step_id[6:]]
        out = spec["fn"](image, np.random.default_rng(seed), dict(spec["defaults"])) if sign > 0 else image
    else:
        raise ValueError(f"Unknown step {step_id!r}")
    return np.clip(out, 0, 1).astype(np.float32), labels


def feature_widths(labels: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Typical width in pixels of each foreground class in one label image (2 x area / boundary length)."""
    out = {}
    for c in range(1, len(class_names)):
        m = labels == c
        area = int(m.sum())
        boundary = int((m[1:, :] != m[:-1, :]).sum() + (m[:, 1:] != m[:, :-1]).sum())
        if area >= 4 and boundary:
            out[class_names[c]] = 2.0 * area / boundary
    return out


def thinnest_feature_width(samples: Sequence[tuple[np.ndarray, np.ndarray]], class_names: list[str]
                           ) -> tuple[str, float] | None:
    """Typical width (px) of the thinnest foreground class: 2 * area / boundary length, median over images."""
    widths: dict[int, list[float]] = {}
    for _, labels in samples:
        for c in range(1, len(class_names)):
            m = labels == c
            area = int(m.sum())
            if area < 4:
                continue
            boundary = int((m[1:, :] != m[:-1, :]).sum() + (m[:, 1:] != m[:, :-1]).sum())
            if boundary:
                widths.setdefault(c, []).append(2.0 * area / boundary)
    if not widths:
        return None
    c, w = min(((c, float(np.median(v))) for c, v in widths.items()), key=lambda t: t[1])
    return class_names[c], w


def sanity_check(samples: Sequence[tuple[np.ndarray, np.ndarray]], cfg: AugmentConfig, class_names: list[str],
                 n: int = 200, seed: int = 0) -> dict[str, Any]:
    """Run the real pipeline ``n`` times and report what it does to the data, in plain language.

    ``samples`` are (image in [0, 1], channel labels) pairs at model scale.
    """
    fired: Counter = Counter()
    clipped, ignored, total, valid_total = 0, 0, 0, 0
    intensity_ids = {st.id for st in PIPELINE if st.kind == "intensity"}
    for i in range(n):
        image, labels = samples[i % len(samples)]
        trace: list = []
        out, lab, _ = augment(image, labels, cfg, np.random.default_rng([seed, i]), trace=trace)
        fired.update({t[0] for t in trace[1:]} | {"crop"})
        valid = lab != IGNORE_INDEX
        stages = [t for t in trace if t[0] in intensity_ids]
        if stages:  # values outside [0, 1] before the final clip are the pixels that lose detail
            v = stages[-1][2]
            clipped += int((((v < 0.0) | (v > 1.0)).any(axis=2) & valid).sum())
        ignored += int((~valid).sum())
        total += lab.size
        valid_total += int(valid.sum())
    rates = []
    for step in PIPELINE:
        exp = expected_rate(cfg, step)
        rates.append({"step": step.id, "title": step.title, "expected": exp,
                      "observed": fired.get(step.id, 0) / n if step.kind != "crop" else 1.0})
    for k, op in enumerate(cfg.extra_ops):
        rates.append({"step": f"extra:{op['name']}", "title": op["name"],
                      "expected": float(op.get("probability", 0)), "observed": fired.get(f"extra:{op['name']}", 0) / n})
    findings: list[tuple[str, str]] = []
    extra_clip = clipped / max(1, valid_total)
    if extra_clip > 0.02:
        findings.append(("warn", f"Intensity changes push {100 * extra_clip:.1f}% of pixels to pure black or white, "
                                 "where detail is lost. Reduce Brightness or Contrast."))
    ignored_frac = ignored / max(1, total)
    if ignored_frac > 0.10:
        findings.append(("warn", f"{100 * ignored_frac:.0f}% of patch pixels lie outside the image and are ignored. "
                                 "Use a smaller patch size or larger images."))
    smallest = min(min(img.shape[:2]) for img, _ in samples)
    if cfg.patch_size > smallest:
        findings.append(("warn", f"The patch ({cfg.patch_size} px) is larger than the smallest image ({smallest} px); "
                                 "the missing area is padded and ignored. Choose a patch size of at most "
                                 f"{smallest - smallest % 8} px."))
    thin = thinnest_feature_width(samples, class_names)
    if cfg.blur_max_sigma > 0 and cfg.photometric_prob > 0 and thin:
        name, width = thin
        fwhm = 2.355 * cfg.blur_max_sigma
        if fwhm > 0.5 * width:
            findings.append(("warn", f"The strongest blur (about {fwhm:.1f} px wide) is comparable to the thinnest "
                                     f"'{name}' features (about {width:.1f} px wide); they may be smeared away. "
                                     f"Keep Blur at or below {0.5 * width / 2.355:.1f} px."))
    if not any(step_enabled(cfg, s) for s in PIPELINE if s.kind != "crop") and not cfg.extra_ops:
        findings.append(("info", "No augmentation is switched on: every patch is an unchanged crop. Fine as a "
                                 "baseline; small datasets usually benefit from at least the Gentle preset."))
    for r in rates:
        tol = 4 * (r["expected"] * (1 - r["expected"]) / n) ** 0.5 + 0.03
        if abs(r["observed"] - r["expected"]) > tol:
            findings.append(("warn", f"{r['title']} fired on {100 * r['observed']:.0f}% of patches but "
                                     f"{100 * r['expected']:.0f}% was expected."))
    if not [f for f in findings if f[0] == "warn"]:
        findings.insert(0, ("ok", "No problems found: masks stay aligned, few pixels are clipped or ignored."))
    return {"patches": n, "rates": rates, "clipped_extra_fraction": extra_clip, "ignored_fraction": ignored_frac,
            "thinnest_feature": thin, "findings": findings}


# ---- named recipes (saved per workspace) ----------------------------------------------------------------------
def recipes_dir(workspace_root: Path) -> Path:
    return Path(workspace_root) / "recipes"


def save_recipe(workspace_root: Path, name: str, cfg: AugmentConfig) -> Path:
    from .atomic import write_json_atomic

    clean = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()
    if not clean:
        raise ValueError("Give the recipe a name (letters, digits, spaces, - or _).")
    path = recipes_dir(workspace_root) / f"{clean}.json"
    write_json_atomic(path, {"schema": "microseg-workbench.augment-recipe/1", "name": clean,
                             "augment": cfg.to_dict()})
    return path


def list_recipes(workspace_root: Path) -> list[str]:
    folder = recipes_dir(workspace_root)
    return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []


def load_recipe(workspace_root: Path, name: str) -> AugmentConfig:
    data = json.loads((recipes_dir(workspace_root) / f"{name}.json").read_text(encoding="utf-8"))
    return AugmentConfig.from_dict(data["augment"])
