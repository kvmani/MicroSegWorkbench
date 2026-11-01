# Independent review ledger

No independent reviews yet.

Append findings; do not erase history. Suggested record:

## RV-YYYYMMDD-001 — Short description
- **Status:** Open / Addressed / Verified / Deferred
- **Severity:** Critical / High / Medium / Low
- **Reviewer and date:**
- **Reviewed commit / scope:**
- **Type:** Confirmed defect / Unverified risk / Suggestion
- **Evidence and reproduction:**
- **Impact:**
- **Recommended action and verification:**
- **Fix commit and tests:**
- **Verification reviewer / evidence:**

## Independent repository review — 2026-11-01 (Codex)

**Reviewed baseline:** `47449da592abdb876bae7cfbbbd2a9750d65fb79` (`main`, v0.4.0).
**Protocol:** AGENTS.md / REVIEW.md; documentation-only review, no application fixes.
**Scope:** repository-wide assessment across data import/versioning, splits, augmentation/preprocessing,
metrics, real CPU training/evaluation/resume/prediction, collaboration/storage/service, Qt workflows,
plugins, reports and Windows/offline deployment configuration. Scientific integrity and publication boundaries
received the deepest independent testing. This is not a proof that every path is correct.

**Readiness: changes required.** Seven High and one Medium confirmed findings below. No Critical-severity defect
was established. High findings block the affected scientific/import/publication workflows and dependent phase
closure per REVIEW.md. Passing the existing suite does not resolve these independently reproduced defects.

### Test evidence and limits

- Environment: Linux x86_64, Python 3.12.14; torch 2.15.0+cpu, numpy 2.3.5, Pillow 12.3.0,
  PySide6 6.12.0, pytest 9.2.0, psutil 7.2.2. Qt ran offscreen.
- Full suite: `QT_QPA_PLATFORM=offscreen python -m pytest -q` -> **123 passed, 5 skipped in 132.39 s**.
  Two skips were the optional HydrideSegmentation tests (subsequently run successfully below); three require
  `MSW_TEST_SHARE` and were not run. One warning came from the environment's anyio pytest plugin deprecation.
- Independent adversarial checks: **10 passed in 2.93 s**; each pass confirms an undesirable behavior, rather than
  proving correctness. Exact runnable probes are retained in [REVIEW_REPRODUCTIONS_2026-11-01.md](REVIEW_REPRODUCTIONS_2026-11-01.md).
- Optional real HydrideSegmentation adapter: `MSW_HYDRIDESEG_PATH=/workspace/scratch/HydrideSegmentation python -m
  pytest -q tests/test_hydrideseg.py` -> **3 passed in 15.02 s**, including known-bar geometry and real training,
  inference and scientific analysis. Reference checkout `e6243cab9ae5c37587ec21605a7eb047bcd0e8b4`.
- OnlineAnnotator export interface inspected at `357c4243aecdb465a72e30906da381f537d633d1`
  (`src/online_annotator/services/exports.py`); bundle testing used synthetic fixtures, not a real user export.
  Neither reference repository was modified.
- Not executed here: Windows installer/frozen executable, interactive desktop, CUDA/AMP/real GPU OOM,
  actual SMB/CIFS filesystem or multi-machine contention, office VM deployment, air-gapped installation,
  real microscopy accuracy/calibration. Existing CI claims were read, not independently rerun in those environments.
  The `ml_server` / `ml_server_deploy` repositories were not audited; no integration changes were designed.
- Plausible unverified risks: GC vs active publication and crash windows across separate best/last checkpoint
  writes merit failure-injection testing. These are not counted as confirmed findings. The already documented
  trust-based share identity and absence of service TLS remain deployment constraints, not new discoveries.

## RV-20261101-001 — Dataset manifest changes can turn training data into reported test data
- **Status:** Open
- **Severity:** High
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; dataset integrity, resume and reevaluation.
- **Type:** Confirmed defect
- **Evidence and reproduction:** `core/dataset.py:290-301` verifies only file bytes against hashes supplied by the
  same mutable manifest. `training/trainer.py:435-446` opens the current dataset for reevaluation and checks only
  the class map; it does not compare the recorded manifest digest. The resume path at `trainer.py:215-219,270-283`
  also never enforces `run.json.dataset.manifest_sha256`. Probe `test_manifest_split_tamper_passes_integrity`
  changes all assignments to train and still receives `verify() == []`. Stronger probe
  `test_reevaluate_accepts_training_images_as_test` trains a real one-epoch model, swaps train/test assignments in
  `manifest.json` without changing any image/mask bytes, and successfully rewrites evaluation with those training
  stems labeled `split: test`.
- **Impact:** Immutable test-set and run-provenance guarantees fail; contaminated scores can be reported and
  published under the original dataset ID. Accidental metadata edits are sufficient; this does not require code execution.
- **Recommended action and verification:** Validate the canonical scientific manifest identity and group/split
  invariants when opening/verifying versions. Bind checkpoints/runs to an expected immutable manifest/content
  digest; enforce it before resume, reevaluation and publication. Merely hashing the current manifest without
  checking a previously trusted digest is insufficient. Regression tests must reject split, class-map, group,
  file-reference and expected-hash mutations; unchanged datasets must remain resumable and evaluable.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

## RV-20261101-002 — Mask edits after validation are published with stale shape and provenance
- **Status:** Open
- **Severity:** High
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; dataset creation.
- **Type:** Confirmed defect
- **Evidence and reproduction:** `core/dataset.py:329-350` rechecks the copied image hash but rereads the source
  mask without checking its validated hash or alignment. `test_changed_mask_accepted` validates 32x32 pairs,
  replaces one mask with a valid-class 16x16 PNG, then creates a version successfully. The manifest still says
  32x32 with the previous source-mask hash/class counts; the stored mask is 16x16 and `verify()` reports no problems.
  A same-size annotation edit is likewise not bound to the validated source or approval provenance.
- **Impact:** Invalid aligned datasets can be published, training can fail later, and annotation revisions/counts
  can be misattributed. The GUI separates Check folder from Create dataset version, making intervening edits realistic.
- **Recommended action and verification:** Snapshot/hash the exact source mask bytes used for decoding and match
  the validation hash; recheck dimensions and derive stored counts from those bytes. Fail atomically with an
  actionable revalidation message if input/provenance changed. Test same-size label edits, changed dimensions,
  and a concurrent source replacement; no version may appear on failure.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

## RV-20261101-003 — Version reuse can hide unapproved annotation provenance
- **Status:** Open
- **Severity:** High
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; content identity and approval provenance.
- **Type:** Confirmed defect
- **Evidence and reproduction:** `core/dataset.py:352-365` stores per-sample provenance but excludes it and approval
  status from dataset identity. `core/atomic.py:93-96` discards a new version if the target exists, without comparing
  manifests. `test_unapproved_provenance_reuses_approved_version` prepares approved inputs, changes only provenance
  to submitted/unapproved and validates again (the warning appears), then prepares under the same name. The call
  returns the same existing ID with `contains_unapproved_annotations: false` and the old approved provenance.
- **Impact:** An explicitly flagged unapproved import can become a dataset/report presented as using approved
  ground truth. Reversing the order also loses legitimate approval updates. Independent preparations on two
  machines get the same ID but different manifest bytes and can conflict during share publication.
- **Recommended action and verification:** Define an immutable identity for scientifically relevant annotation
  provenance/revisions/approval, or separate immutable pixel content from explicitly versioned provenance.
  Never silently discard conflicting provenance. Test approved -> submitted and submitted -> approved, changed
  annotator/source revision, repeated identical preparation and independent workstation publication.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

## RV-20261101-004 — Imported stems and shared file paths escape the staging directory
- **Status:** Open
- **Severity:** High
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; external bundle and share materialization boundaries.
- **Type:** Confirmed defect
- **Evidence and reproduction:** `integrations/onlineannotator.py:189,210-211` interpolates unvalidated
  `rec['stem']` into output paths. `_Source.read()` validates source paths but does not protect these destinations.
  `test_import_stem_writes_outside_stage` modifies a valid bundle stem to `../escaped`; import succeeds and writes
  `escaped.png` / `escaped_mask.png` beside the intended destination. Separately,
  `collab/share.py:306-319` trusts each record's `files` key; `test_share_materialize_traversal` passes
  `../escaped` and overwrites an existing file outside the private stage with a correctly hashed blob.
- **Impact:** A malformed or hostile export/shared record can overwrite writable files outside the destination.
  Hash verification does not establish path safety, and staging cleanup cannot roll back escaped writes.
- **Recommended action and verification:** Reject stems containing separators, traversal, absolute/drive/UNC
  syntax and Windows reserved names. Validate every manifest/record relative path against its intended root,
  including symlink resolution and Windows semantics, before any writes/downloads. Reject duplicate normalized
  destinations. Test POSIX and Windows traversal/absolute paths, case collisions and symlink escape; preexisting
  sentinel files outside the stage must remain unchanged and no destination should be published.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

## RV-20261101-005 — POSIX fallback overwrites an exclusively published record
- **Status:** Open
- **Severity:** High
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; immutable share publication on POSIX filesystems without hard links.
- **Type:** Confirmed defect (unsupported-filesystem condition and interleaving injected locally).
- **Evidence and reproduction:** `collab/fsops.py:61-70` falls back from unsupported `os.link` to an existence
  check followed by POSIX `os.rename`, which replaces an existing file. `test_posix_no_hardlinks_two_winners`
  injects `EOPNOTSUPP` and synchronizes two writers just before rename. Both `create_exclusive()` calls return
  True for different bytes under one path, and the second overwrites the first. Real office CIFS was not tested.
- **Impact:** On this supported fallback path, immutable dataset/run records can change under concurrent writers,
  with successful callers incorrectly believing their conflicting publication won. Review hashes/provenance can diverge.
- **Recommended action and verification:** Use a genuinely atomic no-replace primitive supported by the filesystem,
  or fail closed with clear unsupported-filesystem guidance. A check-then-rename fallback is insufficient.
  Tests must force the no-hard-links path under competing processes and prove exactly one winner, stable bytes,
  conflict reporting and no exposed partial records. Verify on the actual Linux-mounted office share.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

## RV-20261101-006 — Existing corrupt blobs are reused during successful publication
- **Status:** Open
- **Severity:** High
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; blob deduplication and publication integrity.
- **Type:** Confirmed defect
- **Evidence and reproduction:** `collab/blobstore.py:105-112,143-146` treats existence as verified content.
  `test_publish_reuses_corrupt_blob` preloads a real dataset image blob, corrupts its bytes at the digest path,
  then calls `SharedStore.publish_dataset()` on the intact local dataset. Publication succeeds and creates an
  immutable dataset record, while `store.check()` immediately reports the referenced corrupt image.
- **Impact:** Successful publication can advertise unusable authoritative datasets/models, despite having an intact
  source available. Later downloads fail; ordinary retry never repairs the reused blob. This violates the documented
  rule that all referenced blobs are verified before a record appears.
- **Recommended action and verification:** Verify reused blobs before committing a publication and handle corrupt
  content explicitly (fail publication or quarantine/repair atomically under appropriate coordination). Do not mutate
  a valid immutable blob. Test corrupt existing destinations, cached corrupt blobs, concurrent repair/publication,
  and ensure no new record references known bad bytes.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

## RV-20261101-007 — Prediction silently overwrites images with the same filename stem
- **Status:** Open
- **Severity:** High
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; inference exports and downstream scientific analysis.
- **Type:** Confirmed defect
- **Evidence and reproduction:** `training/predict.py:39-40,69-80` accepts all supported image extensions but names
  artifacts using only `path.stem`. `test_prediction_same_stem_overwrites` predicts real model outputs for
  `same.png` and `same.tif`: `predictions.json` reports two successful input records, but only one
  `same_pred_mask.png` and overlay exist. `integrations/hydrideseg.py:155-158` consequently reads the same final
  mask for both input records when analyzing predictions.
- **Impact:** Common mixed-format input folders lose predictions without an error. Scientific statistics can be
  attributed to the wrong input while the CLI reports complete success.
- **Recommended action and verification:** Preflight normalized output names and reject ambiguous inputs before
  inference, or use unique per-input artifact names recorded explicitly in the report and consumed downstream.
  Test same stem across extensions, case-insensitive Windows collisions and Unicode normalization collisions;
  every reported success must have a unique matching mask/overlay and correct downstream statistics.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

## RV-20261101-008 — Recovery cleanup deletes another active preparation/download
- **Status:** Open
- **Severity:** Medium
- **Reviewer and date:** Codex, 2026-11-01
- **Reviewed commit / scope:** `47449da`; workspace concurrency and interrupted-publication recovery.
- **Type:** Confirmed defect
- **Evidence and reproduction:** `core/atomic.py:106-117` deletes every `.staging-*` directory without checking
  age, owner or liveness. It is called before each preparation (`core/dataset.py:319`) and download
  (`collab/share.py:309`). `test_concurrent_prepare_deletes_active_stage` pauses one preparation at its first
  image copy, completes a second differently named dataset preparation in the same workspace, then releases
  the first. The first fails with FileNotFoundError because the second deleted its active stage.
- **Impact:** Two app/CLI instances sharing a local workspace can destroy each other's in-progress work and
  waste large transfers/preparations. Completed immutable outputs were not shown to be corrupted by this probe.
- **Recommended action and verification:** Track staging ownership/liveness or serialize operations using a local
  lock; reclaim only demonstrably abandoned work. Do not use mere directory prefix as evidence of interruption.
  Test two simultaneous preparations, two downloads, mixed operations, crash cleanup and stale-owner recovery.
- **Fix commit and tests:** Pending.
- **Verification reviewer / evidence:** Pending independent verification.

### Implementer handoff

Address 001–004 before relying on scientific provenance/imports, 005–006 before authoritative share publication,
and 007 before batch inference with mixed-format folders. Then fix 008. Add regression tests that assert the
safe behavior; the accompanying probes currently assert the defects. Record a fix commit and executed tests against
each ID. Retain all findings and close only after verification. Do not mark roadmap phases complete solely because
the current suite passes. Reconcile the stale Windows/SMB statements in PROGRESS/ROADMAP with dated evidence during
the next deployment handoff.
