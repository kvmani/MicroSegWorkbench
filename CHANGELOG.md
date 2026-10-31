# Changelog

## Unreleased (0.4.0)
- Display-only previews and overlays are capped (768 px per-epoch previews, 1024 px evaluation/prediction panels) so
  large micrographs do not fill the disk; prediction masks stay full resolution.
- Windows: training child process started without a console window (untested on Windows).
- Seed variability: `--repeat-seeds N`, Training page queue, Compare page mean +/- sd when runs differ only by seed.
- Exclude-border preprocessing (e.g. SEM data bars): never trained on or scored; masks cover the analysed region.
- Early stopping (`--early-stopping N`, Training page).
- Data page shows class balance and warns about rare classes (< 2 % of pixels).
- Results page shows the shared review decision (approved / changes requested / rejected) when connected.
- `msw share gc --unreferenced [--apply]`: list or delete unreferenced shared files older than a grace period.
- Tests can no longer touch the real default workspace.
- Large datasets: on-demand loading with an explicit `images_in_memory` block mode, up-front memory check with
  advice, streaming validation/test evaluation, background batch prefetching (results unchanged by default).
- Phase 4 integration and deployment:
  - `msw import-oa` / Data page import of OnlineAnnotator exports (all layouts and mask styles) with hash and
    pixel-count verification, exact class numbers and annotation provenance; optional annotator-assigned splits
    with leakage check; unapproved masks flagged through to evaluation.
  - Optional HydrideSegmentation adapter: feature statistics and truth-vs-prediction distribution distances
    (`msw hydride-stats`, Results page tab).
  - Deployment kit: offline wheelhouse instructions, Windows per-user installer and launcher, systemd unit,
    ml_server_deploy component and portal tile proposals, runbook (docs/DEPLOYMENT.md).
  - Metadata service: default port 5080 (8765 collides with pytex), read-only registry page (`--public-read`).
  - Frozen-executable support for the training child process.
- Phase 3 collaboration:
  - Shared office folder with content-addressed, deduplicated, hash-verified storage and immutable records.
  - Resumable uploads/downloads with a local cache; atomic materialisation; corrupt blobs rejected.
  - Publish dataset versions and completed runs (provenance rules enforced); download colleagues' work.
  - Review/approval records (no self-approval); approved-models view.
  - Optional metadata service (`msw-service`) with token auth and SQLite on the VM's local disk; `msw share sync`.
  - `msw share ...` commands and the Share page in the desktop app.
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
  - Deterministic synthetic micrographs and tests (numerical metrics, leakage, alignment, integrity, determinism, cancellation, CLI and offscreen UI workflows).
- Initialize project mission, office-specific constraints, development rules, phased roadmap, continuous progress ledger and independent review procedure.
