"""Step status for the guided workflow: done / needs attention / to do / locked / optional, with a reason."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.dataset import list_dataset_versions
from ..training.runs import list_runs

DONE, ATTENTION, TODO, LOCKED, OPTIONAL, RUNNING = "done", "attention", "todo", "locked", "optional", "running"
ICONS = {DONE: "✓", ATTENTION: "⚠", TODO: "•", LOCKED: "·", OPTIONAL: "◦", RUNNING: "▶"}
COLORS = {DONE: "#1a7f37", ATTENTION: "#9a5b00", TODO: "#1f2d3d", LOCKED: "#9a9a96", OPTIONAL: "#1f2d3d",
          RUNNING: "#1f5fae"}

GUIDE = [
    ("Data", "Choose a folder of images and masks (or create demo data), press Check folder, fix any red problems, "
             "then Create dataset version."),
    ("Preprocess", "A: leave out image borders or resize only if needed (often nothing to do). B: pick a "
                   "starting point (Standard suits most data), look at the training patches, then press Check these "
                   "settings."),
    ("Training", "Keep the Quick CPU preset for a first try, check the list under 'Ready to train?', then Start."),
    ("Results", "Read the test scores per class and look at the overlays; export a report to share."),
    ("Compare", "Tick two or more runs to see which settings made the difference."),
    ("Share", "Publish datasets and finished runs to the shared folder and review colleagues' models."),
    ("Help", "The full guide to every step."),
]


@dataclass(frozen=True)
class StepStatus:
    state: str
    reason: str


def step_statuses(window) -> list[StepStatus]:
    ctx = window.ctx
    datasets = list_dataset_versions(ctx.workspace.datasets_dir)
    runs = list_runs(ctx.workspace.runs_dir)
    states = [r.state() for r in runs]
    completed = states.count("completed")
    out: list[StepStatus] = []

    report = window.data_page.report
    if datasets:
        out.append(StepStatus(DONE, f"{len(datasets)} dataset version(s) ready."))
    elif report is not None and not report.ok:
        out.append(StepStatus(ATTENTION, "The folder has problems - fix the red items and check again."))
    else:
        out.append(StepStatus(TODO, "Start here: check a folder and create a dataset version."))

    aug = window.augment_page
    if not datasets:
        out.append(StepStatus(LOCKED, "Needs a dataset version from step 1."))
    elif aug.last_check is None:
        out.append(StepStatus(TODO, "Recommended: press Check these settings."))
    elif aug.check_is_stale():
        out.append(StepStatus(TODO, "Settings changed since the last check - check again."))
    elif any(level == "warn" for level, _ in aug.last_check["findings"]):
        out.append(StepStatus(ATTENTION, "The settings check found something to review."))
    else:
        out.append(StepStatus(DONE, "Settings checked - no problems found."))

    train = window.train_page
    if not datasets:
        out.append(StepStatus(LOCKED, "Needs a dataset version from step 1."))
    elif train.is_running():
        out.append(StepStatus(RUNNING, "Training is running."))
    elif states and states[0] == "failed":
        out.append(StepStatus(ATTENTION, "The latest run failed - see the message in step 3 or 4."))
    elif completed:
        out.append(StepStatus(DONE, f"{completed} finished run(s)."))
    else:
        out.append(StepStatus(TODO, "Ready to train."))

    resumable = sum(1 for r in runs if r.resumable())
    if not runs:
        out.append(StepStatus(LOCKED, "Train a model first."))
    elif resumable:
        out.append(StepStatus(ATTENTION, f"{resumable} stopped run(s) can be resumed."))
    else:
        out.append(StepStatus(DONE if completed else TODO, f"{completed} run(s) with test results."))

    out.append(StepStatus(OPTIONAL, "Compare finished runs.") if completed >= 2 else
               StepStatus(LOCKED, "Needs at least two finished runs."))
    shared = bool(window.share_page.store)
    out.append(StepStatus(DONE if shared else OPTIONAL,
                          "Connected to the shared folder." if shared else "Optional: work with colleagues."))
    out.append(StepStatus(OPTIONAL, "The full guide."))
    return out
