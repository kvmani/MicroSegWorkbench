# Changelog

## Unreleased
- Phase 2 local GPU research experience:
  - Exact resume of cancelled/failed/interrupted runs (`msw resume`, Results page), interrupted-process detection.
  - CUDA diagnosis, automatic batch size, mixed precision, actionable out-of-memory errors, `large_gpu` preset.
  - Background CPU/RAM/GPU telemetry (NVML or nvidia-smi) with live charts; simulated GPU mode clearly flagged.
  - Run comparison (`msw compare`, Compare page) with warnings when test sets differ; CSV export.
  - Segment new images (`msw predict`, Results page) with tiled inference for large micrographs.
  - Optional resize factor and per-image standardization; metrics always at original resolution.
  - Cross-entropy + Dice loss (default); warm-up + cosine LR schedule (fixes crash on very short runs).
  - Numerical regression test: quick preset must learn every synthetic class for two seeds.
- Phase 1 local vertical slice (0.1.0):
  - Explicit class maps, image/mask pair validation, leakage-safe group splits and immutable content-addressed dataset versions with atomic publication.
  - Synchronized augmentation with an ignore label for out-of-image pixels, and augmentation previews.
  - Small U-Net trained for real on CPU (GPU when available) with live events, atomic checkpoints, cancellation, fixed-sample prediction evolution and unreviewed test-set evaluation.
  - `msw` CLI (including `quickstart`) and `msw-gui` PySide6 desktop app with guided steps and in-app help.
  - Deterministic synthetic micrographs and 42 tests (numerical metrics, leakage, alignment, integrity, determinism, cancellation, CLI and offscreen UI workflows).
- Initialize project mission, office-specific constraints, development rules, phased roadmap, continuous progress ledger and independent review procedure.
