# MicroSegWorkbench

Collaborative, reproducible machine-learning workbench for scientific microstructure segmentation.

**Status:** Phase 1 (local vertical slice) implemented and tested on synthetic data on Linux/CPU:
data checking -> immutable dataset version with leakage-safe split -> augmentation preview -> real model
training with live progress -> unreviewed test-set evaluation, in a desktop UI and a CLI.
Not yet verified on Windows or on a GPU. See [PROGRESS](docs/PROGRESS.md) for evidence and known gaps.

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
```

Workspace folder (datasets and runs) defaults to `~/MicroSegWorkbench-workspace`; override with
`--workspace` or the `MSW_WORKSPACE` environment variable.

### Command line
| Command | Purpose |
|---|---|
| `msw demo-data [--out DIR]` | Write SYNTHETIC micrographs, masks and `classes.json` |
| `msw validate FOLDER [--group-regex R]` | Check pairing, sizes, class values, duplicates, groups |
| `msw prepare FOLDER --name N [--train/--val/--test/--seed]` | Create an immutable dataset version |
| `msw datasets` / `msw runs` / `msw devices` | List dataset versions, runs, compute devices |
| `msw preview-aug DATASET --out p.png [--augment cfg.json]` | Augmentation preview image |
| `msw train DATASET [--preset quick_cpu\|standard] [--config cfg.json]` | Train, then evaluate on the test split |
| `msw evaluate RUN` | Re-evaluate a run's best checkpoint |

### Input folder format
`<stem>.png|tif|jpg|bmp` micrographs, `<stem>_mask.png` masks (same size, lossless) and a `classes.json`
mapping every mask value to a named class (`indexed`) or declaring `binary_0_255` masks. This matches
OnlineAnnotator's `hydride_pairs` export naming. Details: in-app Help and [ARCHITECTURE](docs/ARCHITECTURE.md).

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
