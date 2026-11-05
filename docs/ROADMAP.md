# Roadmap

This is an outcome-based roadmap, not a fixed implementation blueprint. Update when evidence warrants.

- [x] **Phase 0 — Foundation:** *(done 2026-10-31 alongside Phase 1; reference repos inspected for export/training formats only)* Repository contracts, independent review procedure, ledgers, test harness, small synthetic micrographs/masks, initial architecture assessment of reference repos.
- [ ] **Phase 1 — Local vertical slice:** *(implemented and tested on Linux CPU 2026-10-31; awaiting Windows check and independent review)* Functional guided Windows-friendly Python UI and CLI/service core; prepare and validate pairs; reproducible split and augmentation previews; train a tiny segmentation model on CPU; evaluate and show fixed-sample prediction evolution.
- [ ] **Phase 2 — Local GPU research experience:** *(implemented and tested on CPU incl. simulated GPU telemetry 2026-10-31; needs a real NVIDIA GPU check)* Detect CUDA and VRAM, choose safe presets, monitor resources and training live, checkpoint/resume, compare runs, evaluate numerical and UI regression paths.
- [ ] **Phase 3 — Collaboration:** *(implemented and tested with local mock share and in-process service 2026-10-31; needs a real SMB share and VM check)* Optional VM metadata/API, shared-drive manifests and deduplicated transfers, immutable dataset/run publication, concurrency and interruption recovery, local cache.
- [ ] **Phase 4 — Integration and deployment:** *(implemented 2026-10-31: OnlineAnnotator import, HydrideSegmentation analysis adapter, deployment kit and runbook, registry page; office rehearsal, packaged executable and independent scientific review outstanding)* OnlineAnnotator export imports, reuse stable HydrideSegmentation algorithms, optional ml_server portal and ml_server_deploy integration, offline Windows packaging, operational documentation and independent scientific review.

Each phase must end with executed tests, reviewed gaps, concise docs/ledger updates and a pushed verified commit.
- [x] **Training usability review — 2026-11-05:** reproducible queued seed experiments and accurate Run check
  explanations for validation-only evaluation and checkpoint selection. Linux CPU/Qt regression coverage;
  office acceptance checks above remain outstanding.
- [ ] **Release 1.0.0rc1 - office pilot:** *(prepared 2026-11-04: accuracy details, installation self-test, roll-out plan `docs/ROLLOUT.md`, beginner's guide)* Phase A install and self-test on office workstations and share, Phase B pilot with scientific validation, go/no-go, Phase C general roll-out.
