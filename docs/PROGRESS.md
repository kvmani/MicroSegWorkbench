# Working ledger — agent handoff

## Current status
**Phase 3 — collaboration: implemented and tested with a local mock share and an in-process service (2026-10-31).**
Phases 1-3 work end to end in the UI and CLI on synthetic data. None is formally closed: each still needs the
office checks (Windows, NVIDIA GPU, real SMB share, Ubuntu VM) and an independent review. Next: Phase 4
(OnlineAnnotator import, HydrideSegmentation reuse, deployment/packaging documentation).

## Current objective
Phase 4: import OnlineAnnotator export bundles directly (manifest, classes, provenance), assess and reuse stable
HydrideSegmentation algorithms behind explicit contracts, offline Windows packaging and VM deployment docs.

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

## Verification
- 2026-10-31, Linux sandbox, Python 3.13.18, torch 2.15.0 (CUDA build, no GPU present), numpy 2.5.3, Pillow 12.3,
  PySide6 6.12 offscreen: `python -m pytest -q` -> **83 passed** in ~1.6 min.
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
  - Numerical regression: quick preset learns all synthetic classes (IoU > 0.8) for seeds 1 and 2.
- Seed sweep (4 seeds, synthetic): quick preset test feature mIoU 0.956-0.982.
- UI screenshots of training resources tab (simulated GPU) and compare page inspected manually.

## Known gaps (honest coverage)
- **Not run on Windows**, nor on an NVIDIA GPU: CUDA paths (auto batch probe, mixed precision, NVML/nvidia-smi
  readings, OOM on real hardware) are only exercised with fakes/simulation.
- Whole training split is loaded into memory; very large datasets need a streaming loader (Phase 3/4 concern).
- Augmentation runs in the training process main thread (no worker processes); may bottleneck a fast GPU.
- Share tested only on a local file system; SMB behaviour (no hard links -> check-then-rename fallback, latency,
  permissions, Windows paths) is untested. No garbage collection of unreferenced blobs.
- Identity on the share is self-declared (user name); only the optional service authenticates (tokens; no
  ml_server/LDAP integration yet). Service has no TLS (intended for the intranet VM behind existing services).
- Local runs show "unreviewed" in the Results page even when approved on the share (approval shown on Share page).
- Packaged (frozen) Windows build would need a different training launcher (UI uses `sys.executable -m`).
- CPU/RAM figures need `psutil`; without it they are reported as not measured.
- Mixed precision on CPU is not offered (bfloat16 CPU autocast not evaluated).

## Blockers
None for synthetic development. Needs from the office: a Windows workstation run of `pip install -e .[all]`,
`python -m pytest -q` and `msw-gui`; GPU model/VRAM; exact share paths and permissions (Phase 3).

## Next steps
1. Phase 4: OnlineAnnotator bundle import; HydrideSegmentation reuse assessment; deployment and packaging docs
   (offline wheelhouse, Windows launcher, VM service unit).
2. Office checks: Windows (`pip install -e .[all]`, `python -m pytest -q`, `msw-gui`), GPU `standard` preset, SMB
   share (`msw share init` on a test folder, two workstations publish/pull), VM service.
3. Independent review of Phases 1-3 per REVIEW.md.
