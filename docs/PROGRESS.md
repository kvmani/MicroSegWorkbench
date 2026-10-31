# Working ledger — agent handoff

## Current status
**Phase 1 — local vertical slice: implemented and tested on Linux/CPU with synthetic data (2026-10-31).**
Desktop UI (PySide6) and CLI cover: check a folder -> create an immutable dataset version with a leakage-safe
split -> preview augmentation -> train a small U-Net for real with live charts, resource figures and per-epoch
predictions on fixed validation images -> unreviewed test-set evaluation with per-class metrics and overlays.
Phase 1 is not closed until it is run on a Windows workstation and independently reviewed (see Next steps).

## Current objective
Close Phase 1 (Windows check, independent review), then Phase 2 (GPU detection/presets, VRAM monitoring,
checkpoint resume, run comparison).

## Decisions
- New independent repository; reference rather than modify HydrideSegmentation, OnlineAnnotator, ml_server and ml_server_deploy.
- Single universal AGENTS.md, no CLAUDE.md.
- Routine verified work committed and pushed directly to main.
- Review-only agent uses REVIEW.md and tracks findings in docs/REVIEWS.md.
- Bulk files on office network share; lightweight VM only for metadata coordination; local Windows machine runs training.
- Cloud tests must use real tiny CPU training and generated synthetic data, not rely on private office inputs.
- Phase 1 technical decisions AD-1..AD-5 are in docs/ARCHITECTURE.md: PyTorch + PySide6 (same as
  HydrideSegmentation), torch-free `core/`, training as a child process with a self-describing run directory,
  content-addressed dataset versions with atomic publication, explicit class maps / ignore label / no silent remapping.
- Input format = OnlineAnnotator `hydride_pairs` naming (`<stem>.png` + `<stem>_mask.png`) plus our `classes.json`.
  Reading OnlineAnnotator's `manifest.json` directly is deferred to Phase 4.
- `quick_cpu` preset (15 epochs, 8 base channels, depth 2) chosen so the synthetic demo visibly learns in ~11 s.

## Done
- Phase 0 essentials: package skeleton (`pyproject.toml`, `src/microseg_workbench`), pytest harness, deterministic
  synthetic micrographs (5 specimens x 4 images, classes matrix/hydride/pore), reference formats inspected
  (OnlineAnnotator exports, HydrideSegmentation splitting/training).
- Core: class maps, image/mask IO (8/16-bit, grayscale/RGB, rejects alpha/colour masks), validation report,
  group split, dataset versions + integrity verification, augmentation, rendering, metrics, workspace.
- Training: device choice with visible CPU fallback, U-Net, deterministic trainer, run records, cancellation,
  failure recording, evaluation (`msw evaluate`).
- CLI `msw` and GUI `msw-gui` (steps Data, Augmentation, Training, Results, Help).

## Verification
- 2026-10-31, Linux sandbox, Python 3.13.18, torch 2.15.0 (CUDA build, no GPU present), numpy 2.5.3, Pillow 12.3,
  PySide6 6.12 offscreen: `python -m pytest -q` -> **42 passed** in ~15 s.
  - Numerical: hand-computed confusion matrix/IoU/Dice/precision/recall; absent classes -> None; ignore pixels excluded.
  - Data integrity: pairing/shape/value/duplicate/colour-mask/missing-class-map errors; binary 0/255 decoding;
    stored masks equal source labels; content-addressed reuse; tamper detection; stale staging cleanup.
  - Leakage: groups never cross splits; order-independent; seed-dependent; too-few-groups error.
  - Augmentation: image/label alignment under flips/rot90/free rotation (nearest check), no new class values,
    padding = ignore, reproducible photometric ops.
  - Training: real learning (loss falls >40 % in 10 epochs; test feature mIoU > 0.3), identical losses for
    identical seeds, cancellation, failure status, corrupt dataset refused, checkpoint reload, odd-size inference.
  - Workflows: CLI end-to-end; offscreen GUI end-to-end including a real training subprocess.
- `msw quickstart`: ~11 s on 4 CPU cores; synthetic test split mIoU ~0.96-0.98 (synthetic, not a scientific result).
- UI screenshots of all pages inspected manually (offscreen rendering).

## Known gaps (honest coverage)
- **Not run on Windows**, nor on an NVIDIA GPU. GPU code paths (device choice, VRAM figures) are untested.
- Guided resizing and alternative normalization (per-image standardization) are not implemented; images are
  used at native resolution with dtype-range scaling. Whole dataset is loaded into memory for training.
- No checkpoint resume, run comparison, approval workflow, or prediction on new unlabelled images yet.
- No share/VM integration (Phase 3). Packaged (frozen) Windows build would need a different training launcher.
- CPU/RAM figures need `psutil`; without it they are reported as not measured.

## Blockers
None for synthetic development. Needs from the office: a Windows workstation run of `pip install -e .[all]`,
`python -m pytest -q` and `msw-gui`; GPU model/VRAM; exact share paths and permissions (Phase 3).

## Next steps
1. Independent review of Phase 1 per REVIEW.md (scientific rules in ARCHITECTURE AD-4 are the main target).
2. Windows smoke test (paths, QProcess launch, fonts/DPI, offline wheel install list).
3. Phase 2: CUDA/VRAM detection and safe presets, live GPU monitoring, resume from `last.pt`, run comparison view,
   out-of-memory handling with a clear message.
