# Architecture decisions

Status: Phases 1-2 implemented (decisions AD-1..AD-8 below); collaboration architecture still open.

The system has three logical responsibilities:
1. **Desktop data/compute client (Windows):** Guided UI, verified dataset cache, real local model training and telemetry, intermediate inference, reporting.
2. **Shared network storage:** Authoritative bulk data, immutable dataset snapshots, source annotations, published selected checkpoints and result artifacts.
3. **Optional VM coordination service:** Small database and API for experiment metadata, permissions, consistent registration and orchestration; no bulk data stored on VM.

Separate bulk-data traffic from metadata API traffic. Prefer content addressing, manifest-based syncing, staging and atomic publication. Avoid direct multi-client DB writes to a network-share SQLite file.

Do not choose a complex distributed framework before a tested vertical slice justifies it. Record significant design decisions with rationale, tradeoffs and migration implications here.

## Decisions (Phase 1, 2026-10-31)

### AD-1 Stack: Python + numpy/Pillow core, PyTorch training, PySide6 desktop UI
- **Why:** HydrideSegmentation already uses PyTorch and PySide6, so algorithms can be reused later; all are
  available as offline wheels for Windows; Qt gives a native, responsive desktop app without a browser or CDN.
- **Tradeoff:** PySide6 and CUDA torch wheels are large (hundreds of MB to GB) for the offline mirror.
- `core/` never imports torch or Qt, so validation, splitting and metrics are testable and reusable anywhere
  (e.g. a future VM service).

### AD-2 Training runs in a child process; the run directory is the interface
- The UI launches `python -m microseg_workbench train ... --quiet` and tails `events.jsonl`/`status.json`.
  The CLI, UI and tests share one code path; a training crash cannot freeze or kill the UI; cancellation is a
  `CANCEL` flag file (works on Windows without signals).
- Run directory: `run.json` (resolved config, dataset id and manifest hash, versions, device, git commit),
  `status.json` (atomically replaced), append-only `events.jsonl` (partial last line ignored by readers),
  `checkpoints/{best,last}.pt` (atomic replace, loaded with `weights_only=True`), `samples/` (fixed-sample
  predictions per epoch), `evaluation/` (test report with `review_status: unreviewed`).
- **Migration:** Phase 3 publication copies a finished run directory to the share atomically; the format is
  already self-describing. Packaged (frozen) builds will need a different child-process launcher.

### AD-3 Immutable, content-addressed dataset versions
- `datasets/<name>-<hash12>/` holds copied images (byte-identical, SHA-256 checked), canonical indexed mask
  PNGs, and `manifest.json` (class map, split config, per-sample split/group/hashes, source paths, warnings).
- The id hashes the scientific content (file hashes, class map, split assignment), so identical inputs reuse the
  same version and any change creates a new one. The split - including the test set - is fixed per version.
- Built in a `.staging-*` directory and published with one `os.replace`; leftovers of interrupted builds are
  removed on the next build. Training re-verifies all hashes before starting.

### AD-4 Scientific data rules
- Class maps are explicit (`classes.json`); masks containing undeclared values, colour masks, non-PNG masks,
  size mismatches, mixed bit depth/channels and duplicate images are errors. No thresholding or remapping
  except the declared `binary_0_255` encoding (255 -> class 1).
- Model channel i = i-th class in the map; checkpoints store the class map and predicted masks are written in the
  original class values.
- Split by group (regex on file stem, default: text before first `_`); every group goes to exactly one split;
  assignment depends only on group names/sizes, fractions and seed.
- Augmentation moves image and labels together; labels are only permuted or nearest-sampled; pixels from outside
  the image (rotation corners, padding) get label 255 = ignore, excluded from loss and metrics.
- Metrics come from one confusion matrix over full images; undefined ratios are `None`, never 0 or 1.
- Model selection uses validation only; the test split is evaluated once at the end with the best checkpoint.
- Normalization: divide by the dtype range (uint8 /255, uint16 /65535). Recorded in config and checkpoint.

### AD-5 Determinism
Seeds drive weight init, batch order and every augmentation (per-sample numpy generators keyed by
seed/epoch/index). `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN benchmark off.
Verified: identical loss sequences for identical seeds on CPU (same machine and thread count). GPU
bit-reproducibility is not claimed (mixed precision and some CUDA kernels are non-deterministic).

## Decisions (Phase 2, 2026-10-31)

### AD-6 Resume = restore everything that influences the future
`last.pt` holds model, AdamW, LR-schedule and AMP-scaler state plus best-score bookkeeping and the global step.
Because batches and augmentation are keyed by (seed, epoch, sample), resuming after epoch k reproduces the
uninterrupted run exactly on CPU (tested bit-for-bit). Resolved settings (batch size, mixed precision) are read
from `run.json` so a resume cannot silently change them. Epochs repeated after a crash are de-duplicated by
readers (`RunDir.epoch_history`, charts replace points). `status.json` records pid and host; an active state whose
process is gone on this host is reported as `interrupted`.

### AD-7 GPU safety and telemetry
- Out-of-memory (CUDA or host) in a step becomes `TrainingOutOfMemory` with concrete advice; the run is `failed`
  with `error_type: out_of_memory` and remains resumable.
- `batch_size: 0` probes 64, 32, ... with a real forward/backward pass on zeros (weights and BatchNorm statistics
  restored afterwards); CPU uses 8. Mixed precision (fp16 autocast + GradScaler) is automatic on CUDA only.
- Telemetry: a daemon thread samples psutil and NVML/`nvidia-smi` every 2 s so a slow query never stalls training.
  `MSW_SIMULATED_GPU_TELEMETRY=1` adds fake GPU figures flagged `simulated: true` for UI tests; the UI labels them.
- LR schedule: linear warm-up (10 %) + cosine decay to 5 %; replaces OneCycleLR, which raised ZeroDivisionError
  for very short runs.
- Loss default `cross_entropy_dice` (CE + soft Dice over classes present in the batch). A seed sweep showed that
  with CE alone the smallest model sometimes never predicted the rare synthetic pore class.

### AD-8 Preprocessing and resolution
`Preprocess(resize_factor, normalization)` is stored in every checkpoint and applied identically in training,
evaluation and prediction. Training uses resized images/masks (area/bilinear for images, nearest for masks);
evaluation and prediction upsample class probabilities (bilinear) to the original size and score against the
original masks. Per-image standardization uses whole-image statistics (computed before patching) so patches and
full-image inference see the same scaling. Inputs larger than 512 px are predicted in overlapping 512 px tiles with
averaged probabilities (bounded GPU memory).
