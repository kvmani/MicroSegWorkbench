"""Plain-language diagnosis of a finished run: what went well, what to try next (torch-free)."""

from __future__ import annotations

import json
from typing import Any

from .runs import RunDir

Finding = tuple[str, str]


def diagnose_run(run: RunDir) -> list[Finding]:
    history = run.epoch_history()
    findings: list[Finding] = []
    if not history:
        return [("info", "No finished epochs yet.")]
    status = run.status()
    eval_path = run.evaluation / "evaluation.json"
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    metrics = evaluation.get("metrics") or {}

    val = [(e["epoch"], e["val_loss"]) for e in history if e.get("val_loss") is not None]
    train = [(e["epoch"], e["train_loss"]) for e in history if e.get("train_loss") is not None]
    if len(val) >= 4 and len(train) >= 4:
        best_epoch, best_loss = min(val, key=lambda t: t[1])
        last_val, last_train = val[-1][1], train[-1][1]
        train_at_best = dict(train).get(best_epoch, train[0][1])
        if best_epoch <= val[-1][0] - 3 and last_val > 1.2 * best_loss and last_train < train_at_best:
            findings.append(("warn", f"Overfitting: validation loss was lowest at epoch {best_epoch} and rose "
                                     f"afterwards while training loss kept falling. The best checkpoint (used for "
                                     "the results) is from before; next time try early stopping, stronger "
                                     "augmentation or more training images."))
    scores = [(e["epoch"], e.get("val_mean_iou_foreground")) for e in history
              if e.get("val_mean_iou_foreground") is not None]
    if len(scores) >= 4 and not status.get("stopped_early"):
        last3 = [s for _, s in scores[-3:]]
        best = max(s for _, s in scores)
        if last3[-1] >= best - 1e-9 and last3[-1] - last3[0] > 0.02:
            findings.append(("info", "Still improving at the last epoch - training for more epochs will probably "
                                     "give better results."))
    if status.get("stopped_early"):
        findings.append(("info", f"Stopped early after epoch {status['stopped_early']} because validation stopped "
                                 "improving."))
    for c in metrics.get("per_class", [])[1:]:
        if c["true_pixels"] and (c["iou"] or 0) == 0:
            findings.append(("warn", f"The model never finds class '{c['name']}' on the test images. Use the "
                                     "Cross-entropy + Dice loss, train longer, or add images containing it."))
        elif 0 < c["true_pixels"] < 500:
            findings.append(("info", f"Class '{c['name']}' has only {c['true_pixels']} test pixels; its score is "
                                     "uncertain."))
    best_val = max((s for _, s in scores), default=None)
    test = metrics.get("mean_iou_foreground")
    if best_val is not None and test is not None and best_val - test > 0.15:
        findings.append(("warn", f"The test score ({test:.2f}) is much lower than the best validation score "
                                 f"({best_val:.2f}). Validation images may not represent the test specimens; check "
                                 "the groups and consider more varied training data."))
    if evaluation.get("ground_truth", "").startswith("contains"):
        findings.append(("warn", "The ground truth includes masks that were not approved in the annotation tool."))
    if test is not None and not [f for f in findings if f[0] == "warn"]:
        findings.insert(0, ("ok", f"Training looks healthy (test mean IoU of feature classes {test:.2f})."))
    return findings


def diagnostics_dict(run: RunDir) -> dict[str, Any]:
    return {"findings": [{"level": lv, "message": m} for lv, m in diagnose_run(run)]}
