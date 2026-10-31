# Roadmap

This is an outcome-based roadmap, not a fixed implementation blueprint. Update when evidence warrants.

- [ ] **Phase 0 — Foundation:** Repository contracts, independent review procedure, ledgers, test harness, small synthetic micrographs/masks, initial architecture assessment of reference repos.
- [ ] **Phase 1 — Local vertical slice:** Functional guided Windows-friendly Python UI and CLI/service core; prepare and validate pairs; reproducible split and augmentation previews; train a tiny segmentation model on CPU; evaluate and show fixed-sample prediction evolution.
- [ ] **Phase 2 — Local GPU research experience:** Detect CUDA and VRAM, choose safe presets, monitor resources and training live, checkpoint/resume, compare runs, evaluate numerical and UI regression paths.
- [ ] **Phase 3 — Collaboration:** Optional VM metadata/API, shared-drive manifests and deduplicated transfers, immutable dataset/run publication, concurrency and interruption recovery, local cache.
- [ ] **Phase 4 — Integration and deployment:** OnlineAnnotator export imports, reuse stable HydrideSegmentation algorithms, optional ml_server portal and ml_server_deploy integration, offline Windows packaging, operational documentation and independent scientific review.

Each phase must end with executed tests, reviewed gaps, concise docs/ledger updates and a pushed verified commit.
