# REVIEW.md — Independent Review Protocol

Purpose: a separate agent (Codex, Claude, Gemini or other) critically examines another agent's work, then records actionable, evidence-based findings without becoming the implementer.

1. Read `AGENTS.md`, `docs/SPECIFICATION.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`, and unresolved `docs/REVIEWS.md`. Limit exploration to the requested scope plus necessary integration boundaries.
2. Review scientific validity first: correct image/mask pairing and class maps, split leakage, augmentation transformations, metric definitions, reference calculations, repeatability and provenance.
3. As relevant, examine concurrency, network cache/synchronization and atomic publication, interruption recovery, CPU/GPU detection and out-of-memory behavior, Windows/offline deployment, security, UI discoverability, architecture, documentation and backward compatibility.
4. Run existing tests; design focused independent adversarial or numerical tests if practical. Distinguish **observed defects**, **plausible unverified risks** and **suggestions**. Never infer that passing tests prove scientific correctness.
5. Append findings to `docs/REVIEWS.md` with ID, date, reviewer, reviewed commit/scope, severity (Critical/High/Medium/Low), precise evidence (files/lines/reproduction), impact, fix suggestion and verification criteria. Note unreviewed areas and tests not executed.
6. Review-only means **no application-code edits**. Commit and push only review documentation to `main`, safely reconciling concurrent changes; never force-push.
7. Implementers record the fix commit and regression tests against each finding. Close only after verification, ideally by an independent reviewer; retain past findings for audit. Critical/High defects block affected functionality or dependent phases.

Conclude with a short readiness judgment and the most consequential unresolved risks. Prefer a handful of verified issues to speculative volume.
