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
    split = evaluation.get("split")
    split_label = {"test": "test", "val": "validation"}.get(split, "evaluation")
    if split != "test":
        findings.append(("warn", "No independent test result is available. " +
                         ("These validation images also guide checkpoint selection; their scores are not an "
                          "independent measure of generalization. Reserve separate specimen groups for testing."
                          if split == "val" else
                          "The evaluation split is unknown; evaluate on a held-out test split before judging generalization.")))
    elif not metrics:
        findings.append(("warn", "No test metrics are available; evaluate the best checkpoint before judging quality."))

    val = [(e["epoch"], e["val_loss"]) for e in history if e.get("val_loss") is not None]
    train = [(e["epoch"], e["train_loss"]) for e in history if e.get("train_loss") is not None]
    if len(val) >= 4 and len(train) >= 4:
        best_epoch, best_loss = min(val, key=lambda t: t[1])
        last_val, last_train = val[-1][1], train[-1][1]
        train_at_best = dict(train).get(best_epoch, train[0][1])
        if best_epoch <= val[-1][0] - 3 and last_val > 1.2 * best_loss and last_train < train_at_best:
            findings.append(("warn", f"Overfitting: validation loss was lowest at epoch {best_epoch} and rose "
                                     "afterwards while training loss kept falling. Checkpoint selection uses "
                                     "validation foreground IoU, so the selected epoch may differ from the lowest-loss "
                                     "epoch. Next time try early stopping, stronger "
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
            findings.append(("warn", f"The model never finds class '{c['name']}' on the {split_label} images. Use the "
                                     "Cross-entropy + Dice loss, train longer, or add images containing it."))
        elif 0 < c["true_pixels"] < 500:
            findings.append(("info", f"Class '{c['name']}' has only {c['true_pixels']} {split_label} pixels; its score is "
                                     "uncertain."))
    findings += _accuracy_findings(evaluation.get("accuracy"), metrics, split_label)
    best_val = max((s for _, s in scores), default=None)
    test = metrics.get("mean_iou_foreground")
    if split == "test" and best_val is not None and test is not None and best_val - test > 0.15:
        findings.append(("warn", f"The test score ({test:.2f}) is much lower than the best validation score "
                                 f"({best_val:.2f}). Validation images may not represent the test specimens; check "
                                 "the groups and consider more varied training data."))
    if evaluation.get("ground_truth", "").startswith("contains"):
        findings.append(("warn", "The ground truth includes masks that were not approved in the annotation tool."))
    if test is not None and not [f for f in findings if f[0] == "warn"]:
        findings.insert(0, ("ok", f"Training looks healthy (test mean IoU of feature classes {test:.2f})."))
    return findings


AREA_RELATIVE_LIMIT = 0.2  # predicted area fraction more than 20 % off the true one ...
AREA_ABSOLUTE_LIMIT = 0.005  # ... and more than 0.5 percentage points
WEAK_IMAGE_GAP = 0.2  # an image this far below the average feature IoU is pointed out


def _accuracy_findings(acc: dict[str, Any] | None, metrics: dict[str, Any], split_label: str = "test") -> list[Finding]:
    """Biased area fractions, misplaced boundaries and unusually weak images, in plain words."""
    if not acc:
        return []
    out: list[Finding] = []
    iou = {c["name"]: c["iou"] for c in metrics.get("per_class", [])}
    for c in acc["per_class"][1:]:
        true, err = c["area_fraction_true"], c["area_fraction_error"]
        if true and err is not None and abs(err) > AREA_ABSOLUTE_LIMIT and abs(err) > AREA_RELATIVE_LIMIT * true:
            word = "over" if err > 0 else "under"
            out.append(("warn", f"The model {word}estimates the area fraction of '{c['name']}': "
                                f"{100 * c['area_fraction_pred']:.2f} % predicted vs {100 * true:.2f} % true on the "
                                f"{split_label} images. Area fractions measured with this model would be biased."))
        f1 = c["boundary_f1"]
        if f1 is not None and f1 < 0.5 and (iou.get(c["name"]) or 0) >= 0.6:
            out.append(("info", f"Boundaries of '{c['name']}' are often more than {acc['boundary_tolerance_px']} px "
                                f"off (boundary F1 {f1:.2f}) although the overlap is good: feature widths and shapes "
                                "are less reliable than areas."))
    scored = [i for i in acc.get("per_image", []) if i["mean_iou_foreground"] is not None]
    if len(scored) >= 3:
        mean = sum(i["mean_iou_foreground"] for i in scored) / len(scored)
        worst = scored[0]
        if worst["mean_iou_foreground"] < mean - WEAK_IMAGE_GAP:
            out.append(("info", f"{split_label.title()} image {worst['stem']} scores much lower ({worst['mean_iou_foreground']:.2f}) "
                                f"than the average ({mean:.2f}); look at its overlay - it may be unusual or its "
                                "mask may need checking."))
    return out


def diagnostics_dict(run: RunDir) -> dict[str, Any]:
    return {"findings": [{"level": lv, "message": m} for lv, m in diagnose_run(run)]}
