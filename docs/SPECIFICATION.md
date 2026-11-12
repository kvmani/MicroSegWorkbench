# Product specification — gross overview

## Goal
Python-based collaborative platform for end-to-end scientific image segmentation research, operated by colleagues without coding experience. Hydrides first; general multiclass microstructure features later.

## Workflows
- Import and visually review images and pixel-exact annotations (especially approved `OnlineAnnotator` exports); validate shape, class maps, pairing, duplicates, provenance and versioned datasets.
- Guided resizing, normalization, cropping, preprocessing and configurable geometric/photometric augmentation with synchronized image/mask previews and support for future custom operations.
- Reproducible train/validation/test split by acquisition/specimen/source grouping to avoid leakage; immutable test sets.
- Local Windows GPU model training using sensible presets plus optional hyperparameter controls; safe device discovery and CPU execution where possible.
- Responsive dashboards for epoch/step status, loss and segmentation metrics, CPU/GPU/VRAM utilization, ETA, checkpoints and evolving inference on fixed validation examples.
- Scientific comparison and review of runs/models, per-class measures and image overlays; human approval for deployment-ready model candidates.
- Collaborative discovery of centrally registered dataset versions, experiments and published models; users train independently using locally cached data.

## Deployment
- Air-gapped office; internal Python package mirrors or offline wheels; no external runtime services.
- Shared network drive is authoritative for **all large research data and published artifacts**; never use Ubuntu VM disk for large images/checkpoints.
- Optional dedicated lightweight Python API on existing Ubuntu VM for identity, coordination, experiment metadata and consistent database transactions; separate from existing services.
- Windows training reads cached verified datasets and uploads compact metrics/selected checkpoints when publishing; resilient interrupted transfers, hash checks, deduplication, atomic commits, conflict handling, and controlled permissions.
- Central transactional metadata storage must be server-managed rather than a shared network SQLite database.

## External references
- https://github.com/Pushpalathadevi/HydrideSegmentation
- https://github.com/kvmani/OnlineAnnotator
- https://github.com/kvmani/ml_server
- https://github.com/kvmani/ml_server_deploy

Integrate through stable explicit contracts; none are required for standalone synthetic-data smoke tests.

## Development
Deliver a real, minimal CPU-capable synthetic-data workflow before expanding. Architecture, libraries and detailed storage formats should be chosen by implementing agents based on evidence; keep components testable, documented and replaceable.

## Pilot data preparation contract (2026-11-12)
- OnlineAnnotator imports retain distinct records with colliding display stems, allocate Windows-safe names
  deterministically on every platform, reserve original names before suffixes, and keep exact member paths
  and original/output stems in provenance. Duplicate exact ZIP member references and hash failures abort.
- Channel policy is explicit: native by default; optional grayscale applies before mixed-channel validation
  and identically to preview/preparation. RGB uses Pillow L / ITU-R 601-2, lossless PNG; native single-channel
  files retain bytes/bit depth. Source files, alignment and mask labels are preserved. Mixed bit depths are errors.
  Dataset identity and model checkpoints carry the policy into inference; absent policy means native for legacy data.
- Group splitting remains the default. Explicit seeded Random by image allows specimen overlap with a visible
  warning and expected counts before creation. Full prepared IDs can be copied and are preselected for training.
