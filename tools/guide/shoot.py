"""Drive the real app offscreen through a simulated office scenario and save screenshots.

SIMULATED: synthetic micrographs, a local folder standing in for the office network drive (/mnt/office-share),
three colleagues' workspaces on one machine. Training is real (CPU).
"""
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from microseg_workbench.collab.share import SharedStore  # noqa: E402
from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset  # noqa: E402
from microseg_workbench.core.workspace import Workspace  # noqa: E402
from microseg_workbench.gui.app import MainWindow  # noqa: E402

BUILD = Path(os.environ.get("MSW_GUIDE_BUILD", Path(__file__).parent / "build"))
OUT = BUILD / "shots"
OUT.mkdir(parents=True, exist_ok=True)
SHARE_ROOT = Path(os.environ.get("MSW_GUIDE_SHARE", "/mnt/office-share"))  # simulated network drive
STORE = SHARE_ROOT / "MicroSegWorkbench"
RAW = SHARE_ROOT / "micrographs" / "Zr-hydride-batch-01"
HOMES = Path(os.environ.get("MSW_GUIDE_HOMES", "/srv/workstations"))  # simulated local disks
os.environ["MSW_USER"] = "alice"

app = QApplication([])


def wait(cond, t=900):
    end = time.monotonic() + t
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.02)
    raise SystemExit("timeout")


def settle(n=30):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def est_wait(train):
    settle(60)
    wait(lambda: train.estimate is not None and not train.estimating and not train.estimate_timer.isActive())
    settle(20)


def shot(win, name, widget=None):
    settle()
    (widget or win).grab().save(str(OUT / f"{name}.png"))
    print("saved", name, flush=True)


def window(user):
    ws = Workspace(HOMES / user / "MicroSegWorkspace").ensure()
    os.environ["MSW_USER"] = user
    win = MainWindow(ws, interactive=False)
    win.resize(1440, 900)
    win.show()
    settle()
    return win, ws


USERS = ("alice", "bob", "carol", "dave")


def reset():
    """Remove only what an earlier run of this script created (never the drive or home roots themselves)."""
    for p in [STORE, SHARE_ROOT / "micrographs"] + [HOMES / u for u in USERS]:
        shutil.rmtree(p, ignore_errors=True)
    RAW.parent.mkdir(parents=True)
    write_synthetic_dataset(RAW, SyntheticSpec(specimens=6, images_per_specimen=4, size=160, seed=11))
    SharedStore.create(STORE)


reset()
# ---------------------------------------------------------------- Alice: data
win, ws = window("alice")
data = win.data_page
# make two typical mistakes: an image without a mask, a mask with an undeclared value
good_mask = np.asarray(Image.open(RAW / "S04_0002_mask.png")).copy()
bad = good_mask.copy()
bad[5:9, 5:9] = 7
Image.fromarray(bad).save(RAW / "S04_0002_mask.png")
(RAW / "S05_0003_mask.png").rename(RAW.parent / "S05_0003_mask.png")
data.folder.setText(str(RAW))
data.check_folder()
wait(lambda: data.report is not None and not data.busy and data.thumbs_ready)
shot(win, "01_data_problems")
Image.fromarray(good_mask).save(RAW / "S04_0002_mask.png")
(RAW.parent / "S05_0003_mask.png").rename(RAW / "S05_0003_mask.png")
data.check_folder()
wait(lambda: data.report is not None and data.report.ok and not data.busy and data.thumbs_ready)
data.gallery.setCurrentRow(5)
data.name.setText("zr-hydride")
shot(win, "02_data_ready")
data.create_version()
wait(lambda: data.last_version_id is not None and not data.busy)
win.refresh_status()
shot(win, "03_dataset_version")

# ---------------------------------------------------------------- step 2
win.next_btn.click()
aug = win.augment_page
aug.main_tabs.setCurrentIndex(0)
aug.preview()
aug.border_editor.drag_edge_to("bottom", 148)
aug.refresh_preprocess()
shot(win, "04_preprocess")
aug.resize_factor.setValue(0.25)
aug.refresh_preprocess()
shot(win, "05_preprocess_resize_warning")
aug.resize_factor.setValue(1.0)
aug.border[1].setValue(0)
aug.refresh_preprocess()
aug.main_tabs.setCurrentIndex(1)
aug.use_preset("standard")
aug.preview()
shot(win, "06_augment_builder")
aug.tabs.setCurrentIndex(1)
aug.show_steps(3)
shot(win, "07_augment_steps")
aug.run_check()
wait(lambda: not aug.checking)
aug.tabs.setCurrentIndex(3)
shot(win, "08_augment_check")
aug.tabs.setCurrentIndex(0)
win.refresh_status()

# ---------------------------------------------------------------- step 3
win.go_to(2)
train = win.train_page
train.preset.setCurrentIndex(train.preset.findData("quick_cpu"))
train.network.setCurrentIndex(train.network.findData("unet"))
train.epochs.setValue(25)
train.device.setCurrentIndex(train.device.findData("cpu"))
win.refresh_status()
train.refresh_estimate()
est_wait(train)
shot(win, "09_train_setup")
train.network.setCurrentIndex(train.network.findData("segformer"))
train.refresh_estimate()
est_wait(train)
shot(win, "10_train_network_choice")
train.network.setCurrentIndex(train.network.findData("unet"))
train.refresh_estimate()
est_wait(train)
win.advanced_btn.click()
shot(win, "11_train_advanced")
win.beginner_btn.click()
train.start()
wait(lambda: len(train.epochs_seen) >= 12 or train.final_state is not None)
shot(win, "12_train_live")
train.tabs.setCurrentIndex(1)
shot(win, "13_train_resources")
train.tabs.setCurrentIndex(0)
wait(lambda: train.final_state is not None)
assert train.final_state == "completed", train.stderr
unet_id = train.series_runs[-1]

# transformer on the same dataset version, through the same page
train.network.setCurrentIndex(train.network.findData("segformer"))
train.epochs.setValue(40)
train.start()
wait(lambda: train.final_state is None)
wait(lambda: train.final_state is not None)
seg_id = train.series_runs[-1]
print("runs", unet_id, seg_id, flush=True)

# ---------------------------------------------------------------- step 4
win.go_to(3)
res = win.results_page
res.refresh()
for i in range(res.runs.count()):
    if res.runs.item(i).data(256).endswith(unet_id):
        res.runs.setCurrentRow(i)
res.result_tabs.setCurrentIndex(0)
shot(win, "14_results")
res.result_tabs.setCurrentIndex(1)
shot(win, "15b_results_accuracy")
res.result_tabs.setCurrentIndex(2)
shot(win, "15_results_overlays")
res.note_edit.setPlainText("Checked all 4 test overlays: thin hydrides continuous, pores complete. "
                           "Boundaries of thick hydrides about 1 px wide. Fine for area fractions, not for widths.")
res.add_observation()
res.result_tabs.setCurrentIndex(5)
shot(win, "16_results_observations")
res.result_tabs.setCurrentIndex(6)
shot(win, "17_results_model_card")

# ---------------------------------------------------------------- step 5
win.go_to(4)
cmp = win.compare_page
cmp.refresh()
cmp.set_checked([unet_id, seg_id])
cmp.compare_tabs.setCurrentIndex(0)
shot(win, "18_compare_scores")
cmp.compare_tabs.setCurrentIndex(2)
shot(win, "19_compare_curves")
cmp.compare_tabs.setCurrentIndex(4)
cmp.run_same_image()
wait(lambda: not cmp.si_busy)
shot(win, "20_compare_same_image")

# ---------------------------------------------------------------- step 6: Alice publishes
win.go_to(5)
sp = win.share_page
sp.path.setText(str(STORE))
sp.user.setText("alice")
sp.save_settings()
sp.publish_dataset()
wait(lambda: not sp.busy)
sp.local_run.setCurrentIndex(sp.local_run.findData(unet_id))
sp.publish_note.setPlainText("U-Net, quick CPU preset, 25 epochs. Test feature IoU good; thin hydrides continuous. "
                             "Use for area fractions of batch-01-like micrographs.")
shot(win, "21_share_publish")
sp.publish_run()
wait(lambda: not sp.busy)
sp.runs.selectRow(0)
shot(win, "22_share_published")
win.close()

# ---------------------------------------------------------------- Bob reviews
bwin, bws = window("bob")
bob = bwin.share_page
bwin.go_to(5)
bob.path.setText(str(STORE))
bob.user.setText("bob")
bob.save_settings()
bob.runs.selectRow(0)
bob.download_run()
wait(lambda: not bob.busy)
bob.runs.selectRow(0)
bob.decision.setCurrentIndex(bob.decision.findData("approved"))
bob.comment.setPlainText("Re-checked overlays of S03 test images; agree with Alice's limits (not for widths).")
bob.submit_review()
bob.runs.selectRow(0)
shot(bwin, "23_share_review")
bwin.close()

# ---------------------------------------------------------------- Carol fetches and uses
cwin, cws = window("carol")
carol = cwin.share_page
cwin.go_to(5)
carol.path.setText(str(STORE))
carol.user.setText("carol")
carol.save_settings()
carol.approved_only.setChecked(True)
carol.runs.selectRow(0)
shot(cwin, "24_share_fetch")
new_imgs = SHARE_ROOT / "micrographs" / "Zr-hydride-batch-02-unlabelled"
new_imgs.mkdir(parents=True)
for p in sorted(RAW.glob("S0[12]_*.png")):
    if not p.stem.endswith("_mask"):
        shutil.copy(p, new_imgs / p.name.replace("S0", "B2-S0"))
src = np.asarray(Image.open(RAW / "S06_0001.png")).astype(np.float32)
Image.fromarray(np.clip(src * 0.35 + 165, 0, 255).astype(np.uint8)).save(new_imgs / "B2-S09_0001-overexposed.png")
out = HOMES / "carol" / "results" / "batch-02-predictions"
out.parent.mkdir(parents=True)
carol.download_and_use(carol.selected_run_id(), new_imgs, out)
wait(lambda: not carol.busy and (out / "predictions.json").exists() and not cwin.results_page.predicting)
cwin.results_page.result_tabs.setCurrentIndex(6)
shot(cwin, "25_carol_results")
print("carol warning:", cwin.ctx.last_error, flush=True)
from PySide6.QtWidgets import QMessageBox  # noqa: E402

box = QMessageBox(QMessageBox.Icon.Warning, "Check these images", cwin.ctx.last_error, parent=cwin)
box.show()
settle()
box.grab().save(str(OUT / "25b_warning_box.png"))
box.close()
cwin.results_page.result_tabs.setCurrentIndex(2)
shot(cwin, "25c_carol_overlay")
import subprocess  # noqa: E402

run_id = carol.selected_run_id() or unet_id
cli = []
for args in (["models"], ["predict", unet_id, str(new_imgs), "--out", str(HOMES / "carol" / "results" / "cli-run"),
                          "--device", "cpu"]):
    cmd = [sys.executable, "-m", "microseg_workbench", "--workspace", str(cws.root)] + args
    res_ = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MSW_USER": "carol"})
    cli.append("> msw " + " ".join(args).replace(str(SHARE_ROOT), "S:").replace(str(HOMES / "carol"), "C:\\Users\\carol"))
    cli.extend(res_.stdout.rstrip().splitlines())
    cli.append("")
(OUT / "cli.txt").write_text("\n".join(cli))
cwin.go_to(6)
shot(cwin, "26_help")
cwin.help_page.run_selftest()
wait(lambda: not cwin.help_page.selftest_busy)
shot(cwin, "29_help_selftest")
cwin.close()

# ---------------------------------------------------------------- Dave: a dataset that cannot train
dave_raw = SHARE_ROOT / "micrographs" / "Zr-hydride-batch-03"
write_synthetic_dataset(dave_raw, SyntheticSpec(specimens=5, images_per_specimen=3, size=128, seed=5))
for m in dave_raw.glob("*_mask.png"):
    if not m.name.startswith("S03"):
        arr = np.asarray(Image.open(m)).copy()
        arr[arr == 2] = 0
        Image.fromarray(arr).save(m)
dwin, dws = window("dave")
dd = dwin.data_page
dd.folder.setText(str(dave_raw))
dd.check_folder()
wait(lambda: dd.report is not None and not dd.busy and dd.thumbs_ready)
dd.seed.setValue(1)
dd.name.setText("batch-03")
dd.create_version()
wait(lambda: dd.last_version_id is not None and not dd.busy)
shot(dwin, "27_health_block_data")
dwin.go_to(2)
dwin.refresh_status()
shot(dwin, "28_health_block_train")
dwin.close()
print("DONE", flush=True)
