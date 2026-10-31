# Deployment and operations runbook

Status: the wheelhouse build (section 1, CPU PyTorch) and the Windows installer (section 2) run in CI on a Windows
runner on every push (`.github/workflows/tests.yml`, job "Windows offline installer"). **Nothing has yet been
carried out in the office** (NVIDIA GPU, SMB share, Ubuntu VM, interactive desktop). Treat the first office run as a
rehearsal and record findings in `docs/PROGRESS.md`.

The office setup has three parts:

| Part | Holds | Never holds |
|---|---|---|
| Windows workstation | the app, a local **workspace** (dataset copies, runs in progress, download cache) | the authoritative copy of anything shared |
| Shared network folder | published dataset versions, runs, checkpoints and reviews (immutable) | databases, in-progress runs |
| Ubuntu VM (optional) | metadata service: small SQLite index of what is on the share, user tokens | images, masks, checkpoints |

Everything installs offline from files carried in from an internet-connected machine.

## 1. Build the offline package set (internet-connected machine)

Use the same Python minor version as the workstations (3.12 recommended).

```bat
rem Windows 64-bit wheels for the workstation
py -3.12 -m pip download -d wheelhouse -r deploy\requirements-desktop.txt --only-binary=:all: --platform win_amd64 --python-version 3.12
rem PyTorch: choose the CUDA build that matches the workstations' NVIDIA driver (see pytorch.org "previous versions"),
rem e.g. CUDA 12.x builds come from https://download.pytorch.org/whl/cu12x ; for CPU-only machines use .../whl/cpu
py -3.12 -m pip download -d wheelhouse torch --index-url https://download.pytorch.org/whl/cu124 --only-binary=:all: --platform win_amd64 --python-version 3.12
rem pip itself and the build backend used to install the workbench from source
py -3.12 -m pip download -d wheelhouse pip setuptools wheel
```

Copy `wheelhouse/` and a checkout (or release archive) of this repository to removable media or the internal
mirror. Optionally use `deploy/constraints-tested.txt` (`-c`) to pin the versions used in automated tests.

If the office has an internal PyPI mirror, `pip install` with `--index-url <mirror>` replaces the wheelhouse.

## 2. Install on a Windows workstation

Prerequisites: Python 3.12 (64-bit) from the internal software store; for GPU training, a current NVIDIA driver.

```powershell
cd <media>\MicroSegWorkbench\deploy\windows
powershell -ExecutionPolicy Bypass -File install.ps1 -Wheelhouse <media>\wheelhouse -Source <media>\MicroSegWorkbench
```

This creates `%LOCALAPPDATA%\MicroSegWorkbench\venv`, installs only from the wheelhouse, runs `msw devices`
(shows whether the GPU and CUDA are usable and why not) and adds a desktop shortcut. `MicroSegWorkbench.bat` is an
alternative launcher.

First start:
1. The workspace defaults to `%USERPROFILE%\MicroSegWorkbench-workspace` (change it at the top of the window, or set
   `MSW_WORKSPACE`). Keep it on a **local** disk with room for dataset copies and checkpoints.
2. Step 1 -> *Create demo data* -> step 3 -> *Start training* is a 1-minute self-test.
3. Command-line equivalent: `%LOCALAPPDATA%\MicroSegWorkbench\venv\Scripts\python -m microseg_workbench quickstart`.

Self-test of the installation (optional, needs `pytest` in the wheelhouse):
`python -m pytest -q <source>\tests` (about 2 minutes on CPU).

### GPU notes
- `msw devices` prints the GPU, its free memory and a plain-language CUDA diagnosis.
- Presets `standard` and `large_gpu` use the automatic batch size (largest that fits in GPU memory) and mixed
  precision. If memory runs out, the run stops with advice and can be resumed with a smaller batch size.
- GPU utilisation and temperature come from `nvidia-smi` (installed with the driver) or `pynvml` if present.

### Packaged executable (optional, not yet built)
The app supports being frozen into a single executable (e.g. with PyInstaller): the training child process is then
started as `<exe> --cli train ...` instead of `python -m microseg_workbench`. Building and signing such a package is
not part of this repository yet.

## 3. Prepare the shared network folder (once, administrator)

1. Create an empty folder on the share, e.g. `\\fileserver\research\MicroSegWorkbench`.
2. Initialise it from any workstation: `msw share init \\fileserver\research\MicroSegWorkbench`
   (or *6 Share -> Initialise new shared folder*).
3. Permissions (NTFS/SMB): researchers need **read** on everything and **create files/folders** in `blobs\`,
   `datasets\`, `runs\` and `reviews\`. Published items are never modified by the app, so users do not need
   modify/delete rights; withholding them protects published data from accidental deletion. Administrators keep
   full control.
4. Each user connects once: *6 Share -> Shared folder* (or `msw share config --path <share> --user <name>`).

Layout and guarantees are described in `docs/ARCHITECTURE.md` (AD-9, AD-10). No database file is ever placed on
the share.

Maintenance:
- `msw share check` re-hashes every published file (slow on large shares; run monthly or after storage incidents).
- `msw share gc --days 7` removes partial uploads left by interrupted transfers.
- `msw share gc --days 7 --unreferenced` lists files that no published record uses (left by publications that were
  interrupted after uploading); add `--apply` to delete them. Files younger than `--days` are never touched,
  because they may belong to a publication still in progress.
- Back up the share folder with the normal file-server backup; everything in it is immutable.

## 4. Optional metadata service on the Ubuntu VM

The service indexes the share's records, authenticates users with tokens and records who submitted what. It uses
only the Python standard library (no extra packages) and port **5080** (pytex already uses 8765 in the ML suite).

### 4a. As a component of the ML Server suite (`kvmani/ml_server_deploy`)
Proposed `manifest.yml` entry (to be added in that repository by its maintainers; not changed from here):

```yaml
  msw_metadata:
    repo: kvmani/MicroSegWorkbench
    ref: v0.4.0            # a tagged release of this repository (to be tagged)
    commit: ""
    dir: MicroSegWorkbench
    import_check: microseg_workbench
    src: src
    env: shared
    requirements: []       # standard library only
    unit: ml-platform-msw-metadata.service
    port: 5080
    health: /api/health
    start: >-
      {venv}/bin/python -m microseg_workbench.collab.server serve
      --db {root}/shared/data/msw/meta.sqlite --host {bind} --port {port} --public-read
    workdir: "{current}/apps/MicroSegWorkbench"
    environment:
      PYTHONPATH: "{current}/apps/MicroSegWorkbench/src"
```

The database path is under `shared/` (persistent across releases and rollbacks, on the VM's local disk).

### Portal tile (`kvmani/ml_server`)
Start the service with `--public-read` to let anyone on the intranet view a read-only registry page at
`http://<vm>:5080/` (approved models, all published runs, dataset versions; no images, no write access).
A catalog entry in `ml_server/src/ml_server/catalog.py` (maintainers' change, not made from here) could be:

```python
        {
            "id": "microseg-workbench",
            "name": "MicroSegWorkbench models",
            "summary": "Shared segmentation datasets, trained models and their review status.",
            "detail": ("Read-only registry of dataset versions and models published from the MicroSegWorkbench "
                       "desktop app, with test scores and independent approvals."),
            "category": "Microstructure",
            "tags": ["segmentation", "training", "models", "review", "hydride"],
            "state": "active",
            "owner": "MicroSegWorkbench",
            "href": os.getenv("MSW_REGISTRY_URL", "http://127.0.0.1:5080"),
            "icon": "micrograph-mark.svg",
            "internal": False,
        },
```

### 4b. Standalone
Copy the repository to `/opt/microsegworkbench`, then use `deploy/linux/msw-metadata.service` (adjust `User`).

### Users and tokens
```bash
python3 -m microseg_workbench.collab.server add-user alice --db /var/lib/msw/meta.sqlite   # prints alice's token
```
Give each person their token privately; they enter it in *6 Share -> Service token*. Re-running `add-user`
replaces (revokes) the old token. Tokens are stored only as SHA-256 hashes.

### Health, backup, recovery
- Health: `curl http://<vm>:5080/api/health`.
- Backup: `sqlite3 /var/lib/msw/meta.sqlite ".backup '/backup/meta-$(date +%F).sqlite'"` (safe while running).
- If the database is lost, start with an empty one, re-create users, and have one user run `msw share sync`: the
  share is authoritative and the index is rebuilt from it (submitter identities of old records then show as
  unverified).
- The service has no TLS of its own; keep it on the intranet, behind the existing gateway if one is introduced.

## 5. Upgrades

- Workstations: build a new wheelhouse if dependencies changed, re-run `install.ps1` (in-place upgrade; workspaces
  untouched). Dataset versions and runs created by older versions remain readable.
- Share: formats are versioned (`msw-share.json`, record `schema` fields); a newer app refuses formats it does not
  know rather than guessing.
- VM service: replace the source and restart the unit; the database schema is created if missing.

## 6. Troubleshooting

| Symptom | Check |
|---|---|
| "No NVIDIA GPU detected" | `msw devices` diagnosis: CPU-only torch wheel installed, driver missing/outdated |
| Training stops with "ran out of memory" | Resume or restart with a smaller batch size / patch size, or use automatic batch size |
| Run shows "interrupted" | The computer or app stopped during training; *4 Results -> Resume training* |
| "not a MicroSegWorkbench shared folder" | Wrong path, or the share was never initialised (`msw share init`) |
| Download fails half-way | Just retry: completed files are cached and the partial file resumes |
| "already published with different content" | Someone published a different item under the same id; ids never change meaning - investigate before forcing anything |
| Metadata service unavailable | The share still works; run *Sync with service* / `msw share sync` once it is back |
| HydrideSegmentation statistics "not available" | Install scipy, scikit-image, matplotlib and set `MSW_HYDRIDESEG_PATH` to its checkout |
