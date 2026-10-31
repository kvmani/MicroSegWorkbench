# Working ledger — agent handoff

## Current status
**All four roadmap phases are implemented (v0.4.0, 2026-10-31) and tested in a Linux sandbox on CPU with synthetic
data, a local mock share, an in-process metadata service and the real HydrideSegmentation analysis code.**
No phase is formally closed: the remaining work is verification in the office (Windows workstation, NVIDIA GPU,
SMB share, Ubuntu VM), an independent review (REVIEW.md), and a few deliberately deferred items (see Known gaps).

## Current objective
Hand over for office rehearsal and independent review. Implementation agents should next address review findings
and the highest-value remaining gaps (multi-process augmentation, CUDA package, service authentication).

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
- Phase 4 decisions AD-12..AD-14 (OnlineAnnotator import contract, HydrideSegmentation reuse via optional adapter,
  deployment layout; reference repositories unmodified) are in docs/ARCHITECTURE.md.
- Metadata service port 5080: ml_server_deploy already runs pytex on 8765.
- Phase 3 decisions AD-9..AD-11 (immutable content-addressed share without database or locks, local blob cache,
  append-only reviews with no self-approval, optional stdlib metadata service) are in docs/ARCHITECTURE.md.
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
- Phase 3: blob store (dedupe, resumable verified copies, host locks), shared store (exclusive immutable records,
  publish/pull datasets and runs, reviews, check), local cache, `msw share ...`, metadata service + client,
  Share page.
- Post-roadmap hardening: bounded-memory data loading (AD-15), early stopping and excluded borders (AD-16),
  shared review status on Results, share GC, class-balance summary, hermetic tests, seed-variability runs and
  summary (motivated by the observed seed sensitivity of rare classes), capped display previews, blur and
  augmentation plugins (AD-17).
- Phase 4: OnlineAnnotator import (CLI + Data page, annotator splits), HydrideSegmentation adapter (CLI + Results
  tab), deployment kit (`deploy/`, docs/DEPLOYMENT.md), registry page, frozen-build launcher, version 0.4.0.

## Verification
- 2026-10-31, Linux sandbox, Python 3.13.18, torch 2.15.0 (CUDA build, no GPU present), numpy 2.5.3, Pillow 12.3,
  PySide6 6.12 offscreen: `python -m pytest -q` -> **116 passed, 2 skipped** in ~1.5 min (the 2 skips need HydrideSegmentation);
  with `MSW_HYDRIDESEG_PATH=<HydrideSegmentation checkout @ main, v2.0.0>` and scipy 1.18.1 / scikit-image 0.26.0 /
  matplotlib 3.11.2 installed -> **118 passed**.
  - Manual smoke test: 12 synthetic 700x700 images, `images_in_memory: 3`, 128 px patches - tiled evaluation
    scored all 1,470,000 test pixels at full resolution; peak RSS ~0.8 GB; 11 s on 4 cores.
  - Post-roadmap: early stopping (forced plateau -> stops after patience, evaluates epoch-1 checkpoint); excluded
    border never scored (evaluated pixel count, cropped masks, recorded analysed region, survives resume);
    class-balance summary and rare-class warning; Results banner shows shared review decision; share GC.
  - Data loading: patch order covers every patch once, blocks are local, cached/uncached sources give identical
    batches, prefetch keeps order/propagates errors/stops cleanly, blocked training deterministic, memory guard.
  - Phase 1 coverage as before (numerical metrics, data integrity, leakage, augmentation alignment, training,
    CLI and UI workflows).
  - Phase 2: resumed run bit-identical to uninterrupted run (losses and final weights); resume refused when completed
    or no checkpoint; dead pid -> interrupted; injected OOM -> actionable failure; batch-size probe halving;
    nvidia-smi parsing incl. [N/A]; simulated GPU flagged in events and UI; LR schedule valid for 1..1000 steps;
    comparison differences/warnings/CSV; prediction writes original class values, rejects channel mismatch,
    refuses existing output, leaves no staging dirs; tiled vs whole-image agreement > 90 %; resized +
    standardized training scored at original resolution; UI: cancel -> resume from Results -> completed with
    de-duplicated charts, compare page, segment new images, preprocessing controls reach the training config.
  - Phase 3: 8-thread contention on exclusive create (one winner, no temp leftovers); 4 processes storing one blob;
    9 MB transfer interrupted at 5 MB resumes from 5 MB and verifies; source/hash mismatch rejected; corrupt shared
    blob rejected and not cached; interrupted dataset download leaves nothing half-published and keeps cached
    files; publication rules (dataset first, completed runs only, imported runs not republished, identical
    republish no-op); reviews (no self-approval, comment required, latest decision wins); service auth, 409 on
    conflicting content, self-approval 403, 25 concurrent registrations, UNC database path refused, idempotent
    sync; two-user CLI workflow with service; two-user Share page workflow.
  - Phase 4: OnlineAnnotator bundles in all 4 mask styles and both layouts (fixture built to the OnlineAnnotator
    export code), ZIP and extracted folder, tampered file / inconsistent pixel counts / unknown colour / wrong schema
    abort without leftovers, provenance and unapproved flags reach the dataset manifest, annotator splits incl.
    leakage and unassigned refusals, CLI and Data page import; HydrideSegmentation contract (bar orientations
    0/90 deg, zero distance for identical masks, end-to-end run and predictions analysis, Results tab); registry page
    private by default and HTML-escaped; frozen launch command; service imports without third-party packages.
  - Numerical regression: quick preset learns all synthetic classes (IoU > 0.8) for seeds 1 and 2.
- **CI on GitHub Actions (2026-10-31, run 41244484430, commit ecd3a59): Windows (windows-latest, Python 3.12.10, CPU
  PyTorch, Qt offscreen) 115 passed, 2 skipped; Linux (ubuntu-latest) passed.** This covers the Windows file-system
  semantics (exclusive create via os.rename, directory publication), QProcess training launch, multi-process blob
  writes and the metadata service on Windows. Run 41245163997 (commit fd01a8e) also passed the **Windows offline
  installer job**: wheelhouse built as in docs/DEPLOYMENT.md, `install.ps1` with `pip --no-index`, quickstart from the
  installed venv, desktop shortcut present. From run 41245688448 the Windows job also creates a real SMB share
  (`\\localhost\mswshare`) and runs the shared-folder tests through the UNC path; that run **found two SMB-only
  bugs** (concurrent rename losers got "Access is denied"; lock-file deletion hit sharing violations), fixed in
  1d62ec0 and green in run 41246145574. Run 41247114634 (manual) also built a **PyInstaller Windows package** and the
  frozen exe trained and evaluated a model via `--cli quickstart`. Still not covered: NVIDIA GPU, the office file server (permissions,
  latency, multiple machines), an interactive desktop session.
- Seed sweep (4 seeds, synthetic): quick preset test feature mIoU 0.956-0.982.
- UI screenshots of training resources tab (simulated GPU) and compare page inspected manually.

## Known gaps (honest coverage)
- **Office verification outstanding**: the test suite and the offline installer pass on a Windows CI runner, but
  there has been no interactive desktop check (DPI, fonts, real clicks) and the `.bat` launcher was not executed; no NVIDIA GPU run; no SMB share; no VM deployment; OnlineAnnotator import
  tested only against a fixture built from its export code, not a real export ZIP.
- Packaged Windows executable: built and smoke-tested in CI (CPU PyTorch, unsigned); GUI of the frozen build not
  clicked through interactively; no CUDA package built.
- Independent scientific review (Phase 4 deliverable) not done - must be a separate reviewer.
- **Not run on Windows**, nor on an NVIDIA GPU: CUDA paths (auto batch probe, mixed precision, NVML/nvidia-smi
  readings, OOM on real hardware) are only exercised with fakes/simulation.
- Data loading (AD-15): bounded-memory block mode and one background prefetch thread; no multi-process workers,
  so CPU augmentation may still limit a fast GPU on large patches.
- Share tested on local disks and on a Windows SMB loopback share in CI; not yet on the office file server
  (permissions, latency, several machines at once, Linux CIFS mounts without hard links). Unreferenced-blob GC is manual (`msw share gc --unreferenced --apply`).
- Identity on the share is self-declared (user name); only the optional service authenticates (tokens; no
  ml_server/LDAP integration yet). Service has no TLS (intended for the intranet VM behind existing services).
- CPU/RAM figures need `psutil`; without it they are reported as not measured.
- Mixed precision on CPU is not offered (bfloat16 CPU autocast not evaluated).

## Blockers
None for synthetic development. Needs from the office: a Windows workstation run of `pip install -e .[all]`,
`python -m pytest -q` and `msw-gui`; GPU model/VRAM; exact share paths and permissions (Phase 3).

## Next steps
1. Independent review of v0.4.0 per REVIEW.md (scientific rules: ARCHITECTURE AD-4, AD-8, AD-12; concurrency and
   publication: AD-9..AD-11).
2. Office rehearsal following docs/DEPLOYMENT.md: wheelhouse build, Windows install + `python -m pytest -q`, GPU
   `standard` preset, SMB share with two workstations, VM service, a real OnlineAnnotator export; record results here.
3. Then, by value: multi-process augmentation workers for GPU throughput, CUDA package build, ml_server
   authentication for the service.
