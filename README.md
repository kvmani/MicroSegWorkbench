# MicroSegWorkbench

Collaborative, reproducible machine-learning workbench for scientific microstructure segmentation.

**Status (v0.4.0):** Phases 1-4 implemented and tested on synthetic data on Linux/CPU:
data checking -> immutable dataset version with leakage-safe split -> preprocessing and augmentation preview ->
real model training with live progress, resource monitoring, resume and GPU safety -> test-set evaluation, run
comparison and segmentation of new images -> publication to a shared folder, download by colleagues, independent
review/approval, an optional metadata service with a read-only registry page, OnlineAnnotator export import and
optional HydrideSegmentation feature statistics. Desktop UI and CLI. Office deployment: [DEPLOYMENT](docs/DEPLOYMENT.md).
Not yet verified on Windows or on a real GPU. See [PROGRESS](docs/PROGRESS.md) for evidence and known gaps.

Start with [the specification](docs/SPECIFICATION.md). Coding agents follow [AGENTS.md](AGENTS.md); independent reviewers follow [REVIEW.md](REVIEW.md).

## Mission
Enable researchers without programming experience to review annotated image datasets, configure preprocessing and augmentation, train segmentation models on local Windows GPUs, inspect live predictions, and compare reproducible experiments.

Hydride segmentation is the initial use case; workflows must extend to phase, grain, precipitate, crack, pore and other feature segmentation.

## Quick start
Requires Python 3.10+. In the office, install from the internal mirror or an offline wheel folder
(`pip install --no-index --find-links <wheels> -e .[all]`).

```bash
pip install -e .[all]        # numpy, pillow, torch, psutil, PySide6, pytest
msw-gui                      # desktop app; press "Create demo data" in step 1
msw quickstart               # or: synthetic data -> dataset -> quick CPU training -> evaluation
python -m pytest -q          # tests (real tiny CPU training; no GPU or network needed)
                             # CI runs the same suite on Windows and Linux (.github/workflows/tests.yml)
```

Workspace folder (datasets and runs) defaults to `~/MicroSegWorkbench-workspace`; override with
`--workspace` or the `MSW_WORKSPACE` environment variable.

### Command line
| Command | Purpose |
|---|---|
| `msw demo-data [--out DIR]` | Write SYNTHETIC micrographs, masks and `classes.json` |
| `msw validate FOLDER [--group-regex R]` | Check pairing, sizes, class values, duplicates, groups |
| `msw prepare FOLDER --name N [--train/--val/--test/--seed]` | Create an immutable dataset version |
| `msw datasets` / `msw runs` | List dataset versions and runs |
| `msw preview-aug DATASET --out p.png [--augment cfg.json]` | Augmentation preview image |
| `msw train DATASET [--preset quick_cpu\|standard\|large_gpu] [--config cfg.json]` | Train, then evaluate on the test split |
| `msw train DATASET --repeat-seeds N` | N runs differing only by seed; prints mean +/- sd of test scores |
| `msw resume RUN` | Continue a cancelled, failed or interrupted run exactly where it stopped |
| `msw evaluate RUN` | Re-evaluate a run's best checkpoint |
| `msw report RUN` | Self-contained HTML report (scores, curves, overlays, settings, provenance) |
| `msw compare RUN RUN... [--csv f]` | Compare runs (warns if test sets differ) |
| `msw predict RUN FOLDER --out NEW` | Segment new images (masks, overlays, predictions.json) |
| `msw devices` | CPU/GPU, CUDA diagnosis, live GPU readings |
| `msw import-oa EXPORT.zip [--out DIR]` | Import an OnlineAnnotator export (verified, provenance kept) |
| `msw prepare DIR --name N --split-source annotator` | Use the splits assigned in OnlineAnnotator (leakage-checked) |
| `msw hydride-stats RUN [--um-per-px X]` | HydrideSegmentation feature statistics, truth vs prediction (optional) |

### Collaboration (shared office folder)
| Command | Purpose |
|---|---|
| `msw share init PATH` | Prepare an empty shared folder (administrator, once) |
| `msw share config --path PATH [--user NAME] [--service URL --token T]` | Connect this workspace |
| `msw share status` | Shared datasets and runs with review status |
| `msw share publish-dataset ID` / `publish-run RUN` | Publish (verified, immutable, deduplicated) |
| `msw share pull-dataset ID` / `pull-run RUN` | Download (resumable, verified) |
| `msw share review RUN --decision approved\|needs_changes\|rejected --comment "..."` | Record a review |
| `msw share check` / `gc` / `sync` | Verify all files / remove stale partial uploads / re-index the service |

Optional metadata service on the Ubuntu VM (standard library only, SQLite on the VM's local disk):
`msw-service add-user alice --db /var/lib/msw/meta.sqlite` (prints a token), then
`msw-service serve --db /var/lib/msw/meta.sqlite --host 0.0.0.0 --port 5080 [--public-read]`.

### Input folder format
`<stem>.png|tif|jpg|bmp` micrographs, `<stem>_mask.png` masks (same size, lossless) and a `classes.json`
mapping every mask value to a named class (`indexed`) or declaring `binary_0_255` masks. This matches
OnlineAnnotator's `hydride_pairs` export naming; `msw import-oa` converts complete OnlineAnnotator exports.
Details: in-app Help and [ARCHITECTURE](docs/ARCHITECTURE.md).

## Office deployment
- **Windows workstations:** Friendly Python desktop UI, local data cache and GPU/CPU execution.
- **Shared office network drive:** Authoritative bulk datasets, approved annotations, published checkpoints and final artifacts.
- **Optional Ubuntu VM:** Lightweight API, central metadata database and coordination only; no bulk dataset storage.
- **Air-gapped:** Internal Python mirrors/offline packages, no cloud or CDN dependencies.

## Related repositories (reference/integration, not runtime requirements)
- [HydrideSegmentation](https://github.com/Pushpalathadevi/HydrideSegmentation) — existing training, inference, preprocessing and metrics.
- [OnlineAnnotator](https://github.com/kvmani/OnlineAnnotator) — approved pixel-exact image/mask datasets and provenance.
- [ml_server](https://github.com/kvmani/ml_server) — existing Ubuntu intranet scientific service platform.
- [ml_server_deploy](https://github.com/kvmani/ml_server_deploy) — existing deployment integration patterns.

The new repository is independent. Do not modify these reference repositories without a separate explicit request.

## Development
See [ROADMAP](docs/ROADMAP.md), [working ledger](docs/PROGRESS.md), [architecture decisions](docs/ARCHITECTURE.md) and [review findings](docs/REVIEWS.md).

Code layout: `src/microseg_workbench/core` (torch-free data contracts, validation, splits, augmentation,
metrics, synthetic data), `training` (PyTorch model, trainer, run records), `gui` (PySide6 app), `cli.py`.
