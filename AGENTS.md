# AGENTS.md — MicroSegWorkbench

## Mission and context
Build an independent, scientifically rigorous, easy-to-use collaborative microstructure segmentation workbench. Read `docs/SPECIFICATION.md` and `docs/PROGRESS.md` first; explore other documentation and code only as needed for the assigned task.

**Reference repositories:** `Pushpalathadevi/HydrideSegmentation` (existing segmentation and training), `kvmani/OnlineAnnotator` (approved annotation exports), `kvmani/ml_server` (intranet services), and `kvmani/ml_server_deploy` (deployment patterns). Inspect relevant interfaces before designing integrations. Do not change these repositories as part of this project without explicit approval.

## Rules that matter
1. **Office environment:** Entire production workflow must run air-gapped, using internal Python mirrors or offline distributions. Windows workstations run local training and automatically detect NVIDIA GPU/VRAM; offer safe, visible CPU fallback where feasible. Optional lightweight Python coordination service runs alongside existing Ubuntu VM services.
2. **Data placement:** Multi-GB authoritative images, annotations, versioned datasets, published models and results reside on the shared office network drive, **not on the VM**. VM holds only small transactional metadata and coordination state. Local Windows disks hold temporary caches and in-progress runs; use hash-based synchronization, atomic publication and recovery. Never use a multi-writer SQLite database directly on an SMB/network share.
3. **Scientific integrity:** Preserve exact annotation class labels and image/mask alignment; use leakage-safe data splits, deterministic seeds, explicit versions and complete provenance. Never silently change scientific behavior, publish partially written data, or present unreviewed results as approved.
4. **Usability:** Assume nonprogrammer researchers. Provide guided, discoverable workflows, graphic augmentation previews, informative errors, ongoing training progress, GPU/memory monitoring, live metrics and intermediate fixed-sample predictions.
5. **Autonomous testability:** Include tiny deterministic synthetic micrographs/masks and local mock share/service adapters so Codex/Claude/Gemini can exercise real data preparation, training, evaluation and synchronization without office access, internet, or GPU. Mark simulated telemetry; do not replace actual training with mocked success.
6. **Engineering:** Favor readable, modular, typed Python and simple interfaces. Reuse proven algorithms without duplicating or tightly coupling whole applications. Maintain tests for numerical correctness, failure/recovery, data integrity, concurrency and representative user workflows. Run relevant tests for each change, broader regression tests per phase. Never claim tests passed if unrun.
7. **Documentation:** Keep `README.md` useful; update applicable specs, design notes and in-app help with behavior changes. Maintain `CHANGELOG.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md` and `docs/REVIEWS.md` as living records. Do not commit private images, weights, caches, secrets or generated outputs.

## Git and handoff
- Work **directly on `main`**; no routine feature branches or PRs. Push every coherent **verified** increment. Never force-push or rewrite shared history.
- Before changes, fetch and inspect upstream and the working tree. Coordinate overlapping edits; reconcile any upstream advancement before pushing. Preserve others' work.
- `docs/ROADMAP.md` tracks ordered phases; `docs/PROGRESS.md` is the concise live handoff ledger: current goal, decisions, done, test evidence, commit refs, blockers, next step. Update and commit it with each increment.
- Prioritize safe, small end-to-end milestones over speculative infrastructure. Continue autonomously unless a meaningful scientific, safety or infrastructure decision requires input.
- On review-only assignments follow `REVIEW.md`: inspect critically, record evidenced findings, **do not modify application code**, and push review notes to `main`. Implementers address findings with regression tests; keep review history.

## Definition of done
Behavior works in a representative end-to-end path, relevant tests pass, no known critical correctness issues remain, docs/ledger reflect reality, and the verified commit is pushed to `main`. State any coverage gaps honestly.
