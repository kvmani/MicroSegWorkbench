# Working ledger — agent handoff

## Current status
**Phase 1 — local vertical slice, in progress.** Torch-free core (class maps, pair validation, leakage-safe group split, immutable dataset versions, synchronized augmentation, metrics), deterministic synthetic data, real CPU U-Net training with run records, evaluation and CLI are implemented and tested. Desktop UI not yet started.

## Current objective
Build a smallest real end-to-end CPU synthetic dataset -> preparation -> segmentation training -> evaluation path with a usable graphical entry point, while respecting office deployment separation.

## Decisions
- New independent repository; reference rather than modify HydrideSegmentation, OnlineAnnotator, ml_server and ml_server_deploy.
- Single universal AGENTS.md, no CLAUDE.md.
- Routine verified work committed and pushed directly to main.
- Review-only agent uses REVIEW.md and tracks findings in docs/REVIEWS.md.
- Bulk files on office network share; lightweight VM only for metadata coordination; local Windows machine runs training.
- Cloud tests must use real tiny CPU training and generated synthetic data, not rely on private office inputs.

## Verification
- 2026-10-31: `python -m pytest -q` -> 39 passed (Python 3.13, torch 2.15 CPU, numpy 2.5, Linux sandbox, no GPU).
- `msw quickstart` (synthetic, quick_cpu preset, 15 epochs) ~11 s on 4 CPU cores; test-split foreground mIoU ~0.97 on synthetic data (not a scientific result).

## Blockers
None for synthetic first slice. Exact office drive paths, permissions, GPU variants and service endpoint remain configurable deployment details.

## Next steps
1. Inspect only relevant reference formats/training APIs.
2. Add tiny deterministic image/mask generator, core validation contracts and CPU smoke test.
3. Create guided working UI and expand iteratively; update this ledger with test evidence and commit references.
