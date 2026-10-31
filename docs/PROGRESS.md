# Working ledger — agent handoff

## Current status
**Phase 2 — local GPU research experience: implemented and tested on Linux/CPU (2026-10-31).**
Phase 1 (vertical slice) and Phase 2 (resume, GPU safety, telemetry, comparison, prediction, preprocessing) work
end to end in the UI and CLI on synthetic data. Neither phase is closed: both still need a Windows run, a real
NVIDIA GPU run and an independent review. Next: Phase 3 (collaboration: shared-drive publication, local cache,
optional VM metadata service).

## Current objective
Phase 3: immutable publication of dataset versions and runs to a (mock) shared drive with manifests, hash-based
deduplicated transfer, local cache with verification, interruption recovery, concurrency safety, and an optional
lightweight metadata API (never a multi-writer SQLite file on the share).

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
- `quick_cpu` preset: 15 epochs, 16 base channels, depth 2, CE+Dice loss (~11 s on 4 cores). Changed from 8
  channels in Phase 2 after a 4-seed sweep: the 8-channel model failed to learn the rare pore class for some seeds
  (this also affected Phase 1, whose "~0.97" figure was seed-dependent).
- Phase 2 decisions AD-6..AD-8 (resume semantics, GPU safety/telemetry, preprocessing at original-resolution
  scoring) are in docs/ARCHITECTURE.md.

## Done
- Phase 0 essentials: package skeleton (`pyproject.toml`, `src/microseg_workbench`), pytest harness, deterministic
  synthetic micrographs (5 specimens x 4 images, classes matrix/hydride/pore), reference formats inspected
  (OnlineAnnotator exports, HydrideSegmentation splitting/training).
- Core: class maps, image/mask IO (8/16-bit, grayscale/RGB, rejects alpha/colour masks), validation report,
  group split, dataset versions + integrity verification, augmentation, rendering, metrics, workspace.
- Training: device choice with visible CPU fallback, U-Net, deterministic trainer, run records, cancellation,
  failure recording, evaluation (`msw evaluate`).
- CLI `msw` and GUI `msw-gui` (steps Data, Preprocess, Training, Results, Compare, Help).
- Phase 2: exact resume + interrupted detection; OOM advice; auto batch size; mixed precision; CUDA diagnosis;
  resource sampler (psutil/NVML/nvidia-smi, simulated mode flagged); run comparison with test-set guard; prediction
  on new folders with tiled inference; resize factor and per-image standardization; CE+Dice loss; warm-up+cosine
  schedule; `large_gpu` preset; UI resources tab, resume button, compare page, segment-new-images button.

## Verification
- 2026-10-31, Linux sandbox, Python 3.13.18, torch 2.15.0 (CUDA build, no GPU present), numpy 2.5.3, Pillow 12.3,
  PySide6 6.12 offscreen: `python -m pytest -q` -> **65 passed** in ~1.5 min.
  - Phase 1 coverage as before (numerical metrics, data integrity, leakage, augmentation alignment, training,
    CLI and UI workflows).
  - Phase 2: resumed run bit-identical to uninterrupted run (losses and final weights); resume refused when completed
    or no checkpoint; dead pid -> interrupted; injected OOM -> actionable failure; batch-size probe halving;
    nvidia-smi parsing incl. [N/A]; simulated GPU flagged in events and UI; LR schedule valid for 1..1000 steps;
    comparison differences/warnings/CSV; prediction writes original class values, rejects channel mismatch,
    refuses existing output, leaves no staging dirs; tiled vs whole-image agreement > 90 %; resized +
    standardized training scored at original resolution; UI: cancel -> resume from Results -> completed with
    de-duplicated charts, compare page, segment new images, preprocessing controls reach the training config.
  - Numerical regression: quick preset learns all synthetic classes (IoU > 0.8) for seeds 1 and 2.
- Seed sweep (4 seeds, synthetic): quick preset test feature mIoU 0.956-0.982.
- UI screenshots of training resources tab (simulated GPU) and compare page inspected manually.

## Known gaps (honest coverage)
- **Not run on Windows**, nor on an NVIDIA GPU: CUDA paths (auto batch probe, mixed precision, NVML/nvidia-smi
  readings, OOM on real hardware) are only exercised with fakes/simulation.
- Whole training split is loaded into memory; very large datasets need a streaming loader (Phase 3/4 concern).
- Augmentation runs in the training process main thread (no worker processes); may bottleneck a fast GPU.
- No approval workflow or publication yet (Phase 3); all results are marked unreviewed.
- Packaged (frozen) Windows build would need a different training launcher (UI uses `sys.executable -m`).
- CPU/RAM figures need `psutil`; without it they are reported as not measured.
- Mixed precision on CPU is not offered (bfloat16 CPU autocast not evaluated).

## Blockers
None for synthetic development. Needs from the office: a Windows workstation run of `pip install -e .[all]`,
`python -m pytest -q` and `msw-gui`; GPU model/VRAM; exact share paths and permissions (Phase 3).

## Next steps
1. Phase 3 (in progress next): shared-drive store with content-addressed blobs + manifests, atomic publish of
   dataset versions and runs, local cache with hash verification, interruption recovery, concurrency tests,
   review/approval records, optional metadata API.
2. Office checks: Windows smoke test (`pip install -e .[all]`, `python -m pytest -q`, `msw-gui`), GPU run with
   `standard` preset, `msw devices` output.
3. Independent review of Phases 1-2 per REVIEW.md.
