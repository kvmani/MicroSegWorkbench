# Architecture decisions

Status: Phases 1-4 implemented (decisions AD-1..AD-14 below).

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
  the image (rotation corners, padding) get label 255 = ignore, excluded from loss and metrics. With cropping,
  free rotation rotates only a window of side >= patch x sqrt(2) and keeps its centre, so rotation corners never
  fall inside a patch taken from a large enough image.
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

## Decisions (Phase 3, 2026-10-31)

### AD-9 Shared folder = content-addressed blobs + immutable records, no database, no locks
```
<share>/msw-share.json
<share>/blobs/sha256/aa/bb/<sha256>        immutable file contents (deduplicated)
<share>/blobs/partial/<sha256>.<host>.part in-progress uploads (resumable per computer)
<share>/datasets/<dataset_id>.json         dataset record: relative path -> blob hash, summary, publisher
<share>/runs/<run_id>.json                 run record: files (best checkpoint, logs, samples, evaluation), summary
<share>/reviews/<run_id>/<time>-<id>.json  one file per review decision (append-only)
```
- Records are created with an exclusive, atomic "create if absent" (`os.link` on POSIX, `os.rename` on Windows,
  check-then-rename fallback on file systems without hard links). They are never overwritten; republishing identical
  content is a no-op, different content under the same id is an error.
- A record is written only after every referenced blob is uploaded and its SHA-256 verified, so a reader never sees a
  record pointing to partial data. Two computers uploading the same blob race benignly (identical content).
- Concurrency without locks: dataset ids are content hashes, run ids are unique, reviews are separate files.
  A per-host lock file only prevents two processes on one computer appending to the same partial upload.
- Windows/SMB (found by CI on a real `\\localhost` share): a losing concurrent `os.rename` may fail with
  "Access is denied" rather than FileExistsError - treated as "already exists" when the target exists, otherwise
  retried; deleting lock/partial files retries sharing violations (WinError 32). Atomic `os.replace` writes retry
  sharing violations as well.
- Unreferenced blobs (from interrupted publications) are removed only by an administrator command with a grace
  period (`msw share gc --unreferenced --apply`).

### AD-10 Downloads go through a local blob cache
`<workspace>/cache/blobs` uses the same store, so an interrupted multi-GB download keeps its completed (and partially
completed) files; materialisation into `datasets/` or `runs/` happens in a staging directory published atomically.
Every byte is hash-verified on the way in; a corrupt shared blob is rejected and never enters the cache.

### AD-11 Review/approval and the optional metadata service
- Approval is an append-only review record (approved / needs_changes / rejected with comment). The latest decision
  is the status. Self-approval is refused (share and service); rejections need a comment.
- Identity on the share is the configured/OS user name (trust within the office). The optional service adds token
  authentication and records `submitted_by`; entries whose claimed author differs are shown as unverified.
- Service: Python standard library (`http.server`, `sqlite3`), SQLite in WAL mode on the VM's local disk with a single
  write lock (refuses UNC paths), immutable registrations (409 on conflicting content), audit table, connection
  backlog 128. Clients retry connection failures (all endpoints idempotent). The share remains authoritative;
  `msw share sync` rebuilds the index. Integration with ml_server authentication is a Phase 4 item.

## Decisions (Phase 4, 2026-10-31)

### AD-12 OnlineAnnotator import contract
Input: `online-annotator.export/1` bundles (ZIP or extracted), both layouts, all four mask styles. Output: an ordinary
pair folder whose masks hold the OnlineAnnotator class numbers (1..255, 0 = background; binary/red styles map the
target to its own class number), plus `classes.json` and `provenance.json`. Each image and mask is checked against
the manifest SHA-256 and decoded class pixel counts against the manifest counts; unknown colours/values abort the
import (staging folder discarded). Provenance flows into the dataset manifest per sample; unapproved masks set
`contains_unapproved_annotations`, which evaluation reports repeat. Assigned splits can be used
(`SplitConfig.source = "annotator"`), validated for completeness and group leakage. `source` enters the dataset id
only when not the default, so earlier group-split ids are unchanged.

### AD-13 HydrideSegmentation reuse through an optional adapter
HydrideSegmentation's `compute_hydride_statistics`, `statistics_to_json` and `scientific_distance_metrics` are called,
not copied (`integrations/hydrideseg.py`), found via installation or `MSW_HYDRIDESEG_PATH` (appended to `sys.path`
so it cannot shadow workbench modules). Reports name the HydrideSegmentation version. Training, metrics and data
handling do not depend on it; its heavier dependencies (scipy, scikit-image, matplotlib) stay optional. Contract
tests check orientations of known bars (0 deg / 90 deg) and zero distances for identical masks. Its U-Net training was not
adopted: the workbench's trainer is small, tested and integrated with resume/telemetry; revisit if a stronger
architecture is needed (e.g. its segmentation-models-pytorch backbones).

### AD-14 Deployment
Workstations: per-user venv from an offline wheelhouse (`deploy/windows/install.ps1`), desktop shortcut, workspace
on local disk. A frozen build re-launches itself with `--cli` for training. VM: the metadata service imports only
the standard library and the workbench package, port 5080 (8765 is pytex), registrable as an `ml_server_deploy`
component (manifest entry proposed in docs/DEPLOYMENT.md, not applied there); optional `--public-read` HTML registry
for an `ml_server` portal tile. Neither reference repository was modified.

## Decisions (post-roadmap hardening, 2026-10-31)

### AD-15 Bounded memory for large datasets
`SplitSource` decodes samples on demand. Training keeps all training images in memory by default (unchanged batch
order and results) after an up-front estimate from the manifest; if that would exceed 60 % of free RAM the run is
refused with a suggested `images_in_memory`. With a limit N, images are shuffled and processed in blocks of N,
patches shuffled within a block (each image decoded once per epoch). The order depends only on seed, epoch and N -
never on the machine - and N is part of the recorded config, so resume and re-runs are exact. Validation keeps only
the fixed-sample predictions; test evaluation streams one image at a time and writes overlays as it goes. Batches
are prepared one to two steps ahead in a background thread (`prefetch`); errors propagate to the trainer.

### AD-16 Excluded borders and early stopping
`Preprocess.exclude_border` (top, bottom, left, right; original pixels) is applied before resizing in training,
evaluation and prediction. Excluded pixels are never trained on or scored. Evaluation and prediction masks cover the
analysed region only, with `analysed_region` recorded, because filling the border with background would bias
downstream area fractions. `early_stopping_patience` stops after N epochs without validation improvement (needs a
validation split); the best checkpoint is evaluated as usual and `status.json` records `stopped_early`.

### AD-17 Custom augmentation operations (extension point)
Developers add image-only operations with `register_photometric_op(name, fn, defaults, description)` in a plugin
module listed in `MSW_PLUGINS` (module names or `.py` paths; loaded by the CLI, the UI and the training child).
Operations never receive the mask (no misalignment possible), must draw randomness from the provided generator,
must keep the image shape (checked), and are listed in `AugmentConfig.extra_ops` with a probability. Runs record the
operations used and every loaded plugin's file SHA-256. Extra random draws happen only when extra operations are
configured, so existing configurations reproduce exactly. Geometric (mask-moving) plugin operations are deliberately
not supported yet: they would need the alignment property tests the built-in transforms have.
Example: `examples/plugins/example_ops.py`.

### AD-18 Review hardening (RV-20261101-001..008)
- Dataset identity: `content_identity` (shared by creation and `verify`) also covers annotation version/approval when
  provenance exists; `verify` recomputes it, so manifest edits are detected. Runs record the manifest SHA-256 and
  `check_dataset_unchanged` enforces it for resume, re-evaluation and publication.
- Dataset creation decodes exactly the validated source bytes (hash-checked); changes since validation abort.
- Paths from external manifests/records pass `safe_relpath` (or `safe_stem`) before any write.
- Records on file systems without hard links fail closed (`UnsafeFilesystemError`); publishing is supported from
  Windows workstations. Blob reuse during publication re-verifies; damaged shared blobs are reported, never modified.
- Staging folders carry an owner file; cleanup removes only abandoned ones.

## Decisions (beginner workflow UI, 2026-11-01)

### AD-19 Visual augmentation pipeline builder
`core/augment_pipeline.PIPELINE` describes the steps in the exact order `augment()` applies them (geometry, then the
intensity group, then plugins); the builder draws it, so the display cannot diverge from training. Previews come from
the real `augment()` via an optional `trace` that records each applied stage without touching the random stream
(tested byte-identical). Single-step thumbnails use `preview_step`. The order is fixed (owner decision): geometry
before intensity keeps masks aligned. `sanity_check` runs the real pipeline on 200 patches and reports clipping
(values outside [0, 1] before the final clip, valid pixels only), ignored pixels, patch larger than images, blur
FWHM vs. the thinnest class width (2 x area / boundary length) and fire rates vs. settings. Named recipes are JSON files
in `<workspace>/recipes`. Exact-value boxes remain the source of truth behind the sliders (Advanced mode shows them).
