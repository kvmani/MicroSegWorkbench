"""Self-contained HTML report of one run (no scripts, no external files; printable, works offline)."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from .. import TOOL_ID, __version__
from ..core.atomic import write_bytes_atomic
from .runs import RunDir, utc_now

MAX_OVERLAYS = 8


def _f(v: Any) -> str:
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else html.escape(str(v))


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def _svg_chart(title: str, series: dict[str, list[tuple[float, float]]], y_max: float | None = None) -> str:
    """Line chart as inline SVG (one y-axis; legend; validated categorical colours)."""
    colors = ["#2a78d6", "#eb6834"]
    pts = [p for s in series.values() for p in s]
    if not pts:
        return ""
    w, h, left, top, bottom = 520, 220, 48, 30, 30
    x0, x1 = min(p[0] for p in pts), max(max(p[0] for p in pts), min(p[0] for p in pts) + 1)
    y1 = y_max if y_max is not None else max(p[1] for p in pts) * 1.1 or 1.0

    def xy(x: float, y: float) -> str:
        return f"{left + (x - x0) / (x1 - x0) * (w - left - 12):.1f},{h - bottom - y / y1 * (h - top - bottom):.1f}"

    parts = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="4" y="16" font-size="13" fill="#2b2b2b">{html.escape(title)}</text>']
    for i in range(5):
        y = y1 * i / 4
        py = h - bottom - y / y1 * (h - top - bottom)
        parts.append(f'<line x1="{left}" x2="{w - 12}" y1="{py:.1f}" y2="{py:.1f}" stroke="#e4e4e0"/>'
                     f'<text x="{left - 4}" y="{py + 4:.1f}" font-size="10" text-anchor="end" fill="#6b6b6b">{y:.2f}</text>')
    parts.append(f'<text x="{(w + left) / 2:.0f}" y="{h - 6}" font-size="10" text-anchor="middle" fill="#6b6b6b">'
                 f'epoch ({x0:g}-{x1:g})</text>')
    lx = left + 120  # legend left to right, in series order
    for k, (name, s) in enumerate(series.items()):
        if not s:
            continue
        color = colors[k % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(xy(*p) for p in s)}"/>')
        if len(series) > 1:
            parts.append(f'<rect x="{lx}" y="22" width="12" height="4" fill="{color}"/>'
                         f'<text x="{lx + 16}" y="28" font-size="11" fill="#2b2b2b">{html.escape(name)}</text>')
            lx += 7 * len(name) + 34
    parts.append("</svg>")
    return "".join(parts)


def build_report(run: RunDir, review: str = "unreviewed") -> str:
    info, status = run.info(), run.status()
    cfg = info.get("config", {})
    evaluation = json.loads((run.evaluation / "evaluation.json").read_text()) \
        if (run.evaluation / "evaluation.json").exists() else None
    history = run.epoch_history()
    metrics = (evaluation or {}).get("metrics") or {}
    rows = "".join(
        f"<tr><td>{html.escape(c['name'])}</td><td>{_f(c['iou'])}</td><td>{_f(c['dice'])}</td><td>{_f(c['precision'])}"
        f"</td><td>{_f(c['recall'])}</td><td>{c['true_pixels']}</td><td>{c['predicted_pixels']}</td></tr>"
        for c in metrics.get("per_class", []))
    hist_rows = "".join(f"<tr><td>{e['epoch']}</td><td>{_f(e['train_loss'])}</td><td>{_f(e['val_loss'])}</td>"
                        f"<td>{_f(e['val_mean_iou_foreground'])}</td></tr>" for e in history)
    loss_chart = _svg_chart("Loss", {"training": [(e["epoch"], e["train_loss"]) for e in history if e["train_loss"] is not None],
                                     "validation": [(e["epoch"], e["val_loss"]) for e in history if e["val_loss"] is not None]})
    iou_chart = _svg_chart("Validation mean IoU (feature classes)",
                           {"validation mIoU": [(e["epoch"], e["val_mean_iou_foreground"]) for e in history
                                                if e["val_mean_iou_foreground"] is not None]}, y_max=1.0)
    overlays = ""
    for stem in ((evaluation or {}).get("images") or [])[:MAX_OVERLAYS]:
        path = run.evaluation / f"{stem}.png"
        if path.exists():
            overlays += f"<figure>{_img(path, stem)}<figcaption>{html.escape(stem)}: image | ground truth | " \
                        "prediction</figcaption></figure>"
    settings = {k: cfg.get(k) for k in ("preset", "epochs", "learning_rate", "loss", "seed", "base_channels", "depth",
                                         "patches_per_image", "resize_factor", "normalization", "exclude_border",
                                         "early_stopping_patience", "images_in_memory")}
    settings["batch size (resolved)"] = info.get("resolved", {}).get("batch_size")
    settings["mixed precision"] = info.get("resolved", {}).get("mixed_precision")
    setting_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{_f(v)}</td></tr>" for k, v in settings.items())
    aug_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{_f(v)}</td></tr>" for k, v in cfg.get("augment", {}).items())
    env = info.get("environment", {})
    banner_color = "#1a7f37" if review.startswith("approved") else "#8a5a00"
    split = (evaluation or {}).get("split", "test")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Run report {html.escape(run.run_id)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px auto;max-width:1100px;color:#222;background:#fff;padding:0 16px}}
table{{border-collapse:collapse;margin:8px 0 18px}}th,td{{border:1px solid #ccc;padding:3px 8px;text-align:left;font-size:13px}}
th{{background:#f4f4f2}}img{{max-width:100%;height:auto;image-rendering:pixelated}}figure img{{width:min(100%,720px)}}figure{{margin:0 0 14px}}
figcaption{{font-size:12px;color:#555}}.banner{{padding:8px;border:1px solid {banner_color};color:{banner_color};
font-weight:600}}.grid{{display:flex;flex-wrap:wrap;gap:16px}}</style></head><body>
<h1>Segmentation run {html.escape(run.run_id)}</h1>
<p class="banner">Review status: {html.escape(review)}. Scores are from the {html.escape(split)} split of dataset version
{html.escape(info.get('dataset', {}).get('dataset_id', '?'))}.</p>
<p>State: {html.escape(status.get('state', '?'))} - {html.escape(status.get('message', ''))}<br>
Ground truth: {html.escape((evaluation or {}).get('ground_truth', 'n/a'))}<br>
Checkpoint: best validation epoch {_f((evaluation or {}).get('checkpoint_epoch'))}</p>
<h2>Test results</h2>
<p>Mean IoU {_f(metrics.get('mean_iou'))}, mean IoU of feature classes {_f(metrics.get('mean_iou_foreground'))},
pixel accuracy {_f(metrics.get('pixel_accuracy'))}, evaluated pixels {_f(metrics.get('evaluated_pixels'))}.</p>
<table><tr><th>Class</th><th>IoU</th><th>Dice</th><th>Precision</th><th>Recall</th><th>True pixels</th>
<th>Predicted pixels</th></tr>{rows}</table>
<h2>Training</h2><div class="grid">{loss_chart}{iou_chart}</div>
<table><tr><th>Epoch</th><th>Training loss</th><th>Validation loss</th><th>Validation mIoU (features)</th></tr>{hist_rows}</table>
<h2>Test images</h2>{overlays or '<p>No overlays.</p>'}
<h2>Settings</h2><table>{setting_rows}</table><h3>Augmentation</h3><table>{aug_rows}</table>
<h2>Provenance</h2><table>
<tr><th>Dataset version</th><td>{_f(info.get('dataset', {}).get('dataset_id'))}</td></tr>
<tr><th>Dataset manifest SHA-256</th><td>{_f(info.get('dataset', {}).get('manifest_sha256'))}</td></tr>
<tr><th>Software</th><td>{_f(info.get('tool', {}).get('id'))} {_f(info.get('tool', {}).get('version'))}
(commit {_f(info.get('tool', {}).get('git_commit'))})</td></tr>
<tr><th>Environment</th><td>Python {_f(env.get('python'))}, PyTorch {_f(env.get('torch'))}, {_f(env.get('platform'))}</td></tr>
<tr><th>Device</th><td>{_f(info.get('device', {}).get('kind'))}: {_f(info.get('device', {}).get('name'))}</td></tr>
<tr><th>Model</th><td>U-Net, {_f(info.get('model', {}).get('parameters'))} parameters</td></tr>
</table><p style="font-size:12px;color:#555">Generated {html.escape(utc_now())} by {TOOL_ID} {__version__}.</p>
</body></html>"""


def write_report(run: RunDir, review: str = "unreviewed") -> Path:
    path = run.evaluation / "report.html"
    write_bytes_atomic(path, build_report(run, review).encode("utf-8"))
    return path
