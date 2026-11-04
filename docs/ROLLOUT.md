# Roll-out plan (release candidate 1.0.0rc1)

How to take MicroSegWorkbench from "tested on synthetic data" to everyday use in the office, safely and in small
steps. Technical installation details are in [DEPLOYMENT](DEPLOYMENT.md); this document is the plan, the
sign-off checklist and the support arrangements.

## Where we start
- **Ready:** the full workflow (data check -> dataset version -> preprocessing/augmentation -> training -> test
  evaluation with accuracy details -> comparison -> segmentation of new images -> publish/review/fetch through the
  shared folder). About 200 automated tests pass on Windows and Linux CI, including an offline install and a
  Windows SMB loopback share.
- **Not yet proven in the office** (the pilot exists to close these): a real NVIDIA GPU workstation, the real file
  server (permissions, latency, several users at once), the Ubuntu VM service, a real OnlineAnnotator export,
  interactive use by researchers (screen scaling, fonts), and an independent scientific review of the metrics
  and models on real micrographs.

## Roles
| Role | Who | Responsibility |
|---|---|---|
| Owner | project lead | go/no-go decisions, priorities, approves the sign-off |
| IT / administrator | office IT | wheelhouse, installs, share folder and permissions, optional VM service, backups |
| Pilot users | 2-3 researchers | real work during the pilot, observations, issue reports |
| Scientific reviewer | someone not involved in training | checks metrics and models against manual measurements |

## Phase A - preparation (IT, about one day)
1. Build the offline wheelhouse on an internet-connected machine ([DEPLOYMENT section 1](DEPLOYMENT.md)); for GPU
   workstations include the **CUDA** PyTorch wheel that matches the installed NVIDIA driver.
2. Create and initialise the shared folder, set permissions (create but not modify/delete for researchers),
   include it in the file-server backup ([DEPLOYMENT section 3](DEPLOYMENT.md)).
3. Optional: install the metadata service on the VM, create user tokens, schedule the SQLite backup (section 4).
4. Once per shared folder, with an account that may create and delete folders in it:
   `msw selftest --share <share> --share-write` - publishes and downloads through a temporary folder inside the
   share (removed afterwards; real records untouched). This proves the file server supports what publishing needs.
5. Install on the first GPU workstation (`install.ps1`), connect the shared folder, then run
   `msw selftest` (or *Help -> Check this installation*). Expected: **RESULT: READY**, Compute device **PASS**
   with the GPU named, Shared folder (read) **PASS**. This check only reads the share. Keep the saved report
   (`<workspace>\selftest-reports\`).
6. Repeat step 5 on every pilot workstation; on CPU-only machines the device check is a WARN, which is acceptable
   for segmenting images but not for training real models.

## Phase B - pilot (2-3 users, 2-4 weeks)
1. **Onboarding (30 minutes per user):** walk through the beginner's guide (`docs/guide/`) and the *Try it in two
   minutes* section of the in-app Help with demo data; then each user's own small folder.
2. **Real work:** each pilot user prepares one real dataset version (several specimens, test specimens not used
   for training), trains U-Net with the *Standard* preset on the GPU, records observations, publishes; a second
   pilot user reviews.
3. **Scientific validation** (scientific reviewer, before any model is used for reported results):
   - compare predicted area fractions (`predictions.csv`, Accuracy details) with manual or established
     measurements on images *not* used for training;
   - check the weakest test images and the boundary F1 of the feature classes;
   - repeat training with 3 seeds (Advanced -> Repeat with seeds) and confirm that differences between settings
     are larger than the seed spread;
   - record the conclusion as the review comment in step 6.
4. **Collect feedback** continuously (see Support). Fix blocking issues before Phase C; others go to the backlog.

### Go / no-go for general roll-out
All must hold:
- every pilot workstation self-test READY, and GPU training of a real dataset completed;
- publish, review and download worked between at least two workstations on the real share;
- no open issue that loses data, corrupts results or shows unreviewed results as approved;
- the scientific reviewer agrees that at least one model's area fractions match the reference within the
  tolerance the group needs (state it, e.g. +/- 1 percentage point), on images not used for training;
- pilot users can do the workflow without help, using the guide and the in-app Help.

## Phase C - general roll-out
1. Install on the remaining workstations (self-test READY on each, reports kept).
2. One 30-minute group session with the guide; name a pilot user as the first contact person.
3. After two weeks: short review of issues and usage; update the guide where users struggled.

## Sign-off checklist
| # | Item | How to verify | Done |
|---|---|---|---|
| 1 | Wheelhouse built and stored with its `constraints-tested.txt` | file list | |
| 2 | Share initialised, permissions set, in backup | `msw share status` from two PCs; IT backup report | |
| 2a | Share supports publishing | `msw selftest --share-write` report: write test PASS | |
| 3 | Each workstation self-test READY | saved `selftest-reports\*.txt` | |
| 4 | GPU used on GPU workstations | self-test "Compute device: PASS ... GPU" | |
| 5 | Real dataset version created from real images and masks | dataset id, Data step shows no errors | |
| 6 | Real model trained, evaluated, reviewed by a second person | run report (Export report), APPROVED in step 6 | |
| 7 | Scientific validation recorded | reviewer's comment, comparison table | |
| 8 | VM service (if used) healthy and backed up | `curl http://<vm>:5080/api/health`, backup file | |
| 9 | Users onboarded | attendance | |
| 10 | Support contact and feedback route announced | e-mail/notice | |

## Support and feedback
- First contact: the named pilot user; then the owner.
- An issue report should contain: what you did, what you expected, the message shown, the **run id** or dataset
  id, the app version (window title, or `msw --version`), and for installation problems the latest self-test report.
- Useful files to attach: `runs\<run>\status.json`, `events.jsonl`, the exported run report; never send images
  that may not leave the office.
- Collect reports in one place (an issue list or a shared document) and review them weekly during the pilot.

## Data management
- The shared folder is the authoritative store for dataset versions, published models, notes and reviews; it is
  append-only and backed up with the file server. Never edit or delete inside it by hand.
- Workspaces are local working copies and are **not** backed up: publish anything worth keeping.
- Original micrographs and annotations stay in their usual office locations (or OnlineAnnotator); dataset
  versions are verified copies with fingerprints.
- Monthly: `msw share check`; after interrupted transfers: `msw share gc --days 7`.

## Upgrades and rollback
- Upgrade: install the new wheelhouse with `install.ps1`, then run the self-test. Workspaces, the share and the
  service database are kept; formats are versioned and older items stay readable.
- Rollback: re-install the previous wheelhouse. Items created by a newer version that the old one does not
  understand are refused with a message, never misread.
- Keep the previous wheelhouse until the new version has run for two weeks.

## Known limitations at roll-out
- Data loading uses one background thread; on very fast GPUs with large patches the CPU may limit speed.
- Identity on the share is the user name each person enters (no login); the optional service uses tokens.
- The transformer network (SegFormer-lite) has no pretrained weights (offline); it needs more data and epochs than
  U-Net and is mainly for comparison.
- Input checks on new images cover bit depth and brightness only, not magnification or contrast; users must use
  models only for micrographs like the training images.
- Boundary F1 uses a fixed 2-pixel tolerance; it compares edges, not physical widths.
