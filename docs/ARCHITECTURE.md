# Architecture decisions

Status: Initial constraints only; the architecture is to emerge from verified increments.

The system has three logical responsibilities:
1. **Desktop data/compute client (Windows):** Guided UI, verified dataset cache, real local model training and telemetry, intermediate inference, reporting.
2. **Shared network storage:** Authoritative bulk data, immutable dataset snapshots, source annotations, published selected checkpoints and result artifacts.
3. **Optional VM coordination service:** Small database and API for experiment metadata, permissions, consistent registration and orchestration; no bulk data stored on VM.

Separate bulk-data traffic from metadata API traffic. Prefer content addressing, manifest-based syncing, staging and atomic publication. Avoid direct multi-client DB writes to a network-share SQLite file.

Do not choose a complex distributed framework before a tested vertical slice justifies it. Record significant design decisions with rationale, tradeoffs and migration implications here.
