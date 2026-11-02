# Changelog

## Unreleased (0.4.0)
- Dataset check (missing classes per split, inverted-looking masks, tiny splits, single test specimen, unapproved
  annotations) and run check (overfitting, under-training, never-found classes, validation-test gap, thin test
  evidence) in the app, reports and CLI (`msw check-run`).
- Pluggable networks: architecture registry with U-Net and SegFormer-lite (transformer, plain PyTorch); plugins can add
  more; choose in step 3 or `--architecture`; `msw architectures`; architecture recorded in runs and comparisons.
- Workflow map across the top (eight stages coloured by status, click to go there) and a "This run" summary card with a
  rough time estimate from timing one real training step on this computer.
- Visual data step: picture gallery with problem pairs framed red (click a problem to jump to it), mask opacity slider,
  class-balance bars, split shown as specimen tiles in Training/Validation/Test columns.
- Visual preprocessing (step 2, tab A): drag the excluded border on the image; resize slider with before/after views
  at true pixel size and a warning when features would become thinner than ~1.5 px.
- Guided workflow: step status icons with reasons, footer with "what to do here" and Back/Next, Beginner/Advanced
  mode (Beginner default, remembered per workspace), "Ready to train?" checklist that blocks only impossible starts.
- Visual augmentation pipeline builder (step 2): blocks in execution order with on/off, probability dots, strength
  sliders and weakest/strongest thumbnails on the user's image; presets (None/Gentle/Standard/Strong); saved recipes;
  live gallery of real training patches with mask overlay; step-by-step strip; selected-block view; settings check
  (clipping, ignored pixels, blur vs feature width, fire rates).
- Review fixes (RV-20261101-001..008): manifest edits detected and runs bound to their dataset manifest; sources
  re-checked at dataset creation; approval provenance in dataset ids; unsafe import/share paths rejected; fail-closed
  publishing on file systems without hard links; damaged stored blobs refused at publication; prediction refuses
  same-name inputs; cleanup never deletes another process's staging folder.
- Self-contained HTML run report (`msw report RUN`, Results page *Export report*): review status, per-class test
  scores, inline SVG curves, test overlays, settings and provenance; no scripts or external resources.
- Faster free rotation: only a window of side patch x sqrt(2) around the patch is rotated (was the whole image;
  ~3x faster per patch on 1024 px images, no rotation corners inside patches). Changes the exact augmented
  samples of configurations with rotate_prob > 0 compared with earlier builds.
- Packaged Windows executable: `deploy/windows/launcher.py` + manual CI job that builds it with PyInstaller and
  checks the frozen exe trains and evaluates a model.
- Windows SMB: fixed concurrent record creation ("Access is denied" for losers) and lock-file deletion races found
  by new CI tests over a real `\\localhost` SMB share; CI also checks the offline Windows installer.
- CI: full test suite on Windows and Linux (GitHub Actions); first run green on both.
- Windows: atomic file replacement retries briefly on sharing violations (UI reading status.json while training
  replaces it).
- Augmentation: random Gaussian blur; plugin extension point for image-only operations (`MSW_PLUGINS`), recorded
  with file hashes in each run; example plugin with unsharp mask and vignette.
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
