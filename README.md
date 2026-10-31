# MicroSegWorkbench

Collaborative, reproducible machine-learning workbench for scientific microstructure segmentation.

**Status:** Project initialized; implementation not started.

Start with [the specification](docs/SPECIFICATION.md). Coding agents follow [AGENTS.md](AGENTS.md); independent reviewers follow [REVIEW.md](REVIEW.md).

## Mission
Enable researchers without programming experience to review annotated image datasets, configure preprocessing and augmentation, train segmentation models on local Windows GPUs, inspect live predictions, and compare reproducible experiments.

Hydride segmentation is the initial use case; workflows must extend to phase, grain, precipitate, crack, pore and other feature segmentation.

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
See [ROADMAP](docs/ROADMAP.md), [working ledger](docs/PROGRESS.md), and [review findings](docs/REVIEWS.md). Initial task: build a genuinely working local CPU-first synthetic-data end-to-end vertical slice; add network coordination later.

No executable application is included in the initialization commit.
