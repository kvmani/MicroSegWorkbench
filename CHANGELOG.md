# Changelog

## Unreleased
- Phase 1 local vertical slice (0.1.0):
  - Explicit class maps, image/mask pair validation, leakage-safe group splits and immutable content-addressed dataset versions with atomic publication.
  - Synchronized augmentation with an ignore label for out-of-image pixels, and augmentation previews.
  - Small U-Net trained for real on CPU (GPU when available) with live events, atomic checkpoints, cancellation, fixed-sample prediction evolution and unreviewed test-set evaluation.
  - `msw` CLI (including `quickstart`) and `msw-gui` PySide6 desktop app with guided steps and in-app help.
  - Deterministic synthetic micrographs and 42 tests (numerical metrics, leakage, alignment, integrity, determinism, cancellation, CLI and offscreen UI workflows).
- Initialize project mission, office-specific constraints, development rules, phased roadmap, continuous progress ledger and independent review procedure.
