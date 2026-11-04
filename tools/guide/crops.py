import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

D = Path(os.environ.get("MSW_GUIDE_BUILD", Path(__file__).parent / "build"))
HOMES = Path(os.environ.get("MSW_GUIDE_HOMES", "/srv/workstations"))
S, C, G = D / "shots", D / "crops", D / "grid"
C.mkdir(parents=True, exist_ok=True)
G.mkdir(parents=True, exist_ok=True)
CROPS = {
    "full_data_ready": ("02_data_ready", (0, 0, 1572, 900)),
    "tour": ("03_dataset_version", (0, 0, 1572, 900)),
    "data_problems": ("01_data_problems", (205, 140, 980, 800)),
    "data_ready": ("02_data_ready", (980, 200, 1572, 800)),
    "dataset_version": ("03_dataset_version", (980, 392, 1572, 805)),
    "health_data": ("27_health_block_data", (980, 690, 1572, 812)),
    "health_train": ("28_health_block_train", (214, 160, 655, 760)),
    "preprocess": ("05_preprocess_resize_warning", (214, 190, 1556, 780)),
    "augment_builder": ("06_augment_builder", (214, 190, 1556, 712)),
    "augment_steps": ("07_augment_steps", (214, 540, 1556, 770)),
    "augment_check": ("08_augment_check", (214, 540, 1556, 770)),
    "train_setup": ("09_train_setup", (214, 160, 655, 760)),
    "train_advanced": ("11_train_advanced", (214, 160, 655, 800)),
    "train_live": ("12_train_live", (655, 160, 1556, 800)),
    "train_resources": ("13_train_resources", (655, 160, 1556, 800)),
    "results": ("14_results", (598, 145, 1556, 440)),
    "results_overlays": ("15_results_overlays", (598, 300, 1556, 800)),
    "results_observations": ("16_results_observations", (598, 300, 1556, 800)),
    "results_card": ("17_results_model_card", (598, 300, 1556, 670)),
    "compare_scores": ("18_compare_scores", (205, 145, 1215, 800)),
    "compare_same_image": ("20_compare_same_image", (533, 195, 1556, 745)),
    "share_connect": ("21_share_publish", (205, 160, 715, 360)),
    "share_runpub": ("21_share_publish", (720, 160, 1556, 800)),
    "share_review_panel": ("23_share_review", (205, 370, 715, 800)),
    "share_review_table": ("23_share_review", (730, 540, 1556, 595)),
    "share_fetch": ("24_share_fetch", (720, 485, 1556, 720)),
    "help": ("26_help", (205, 90, 1556, 800)),
    "results_accuracy": ("15b_results_accuracy", (598, 300, 1556, 800)),
    "selftest": ("29_help_selftest", (205, 520, 1556, 800)),
}
sizes = {}
for name, (src, box) in CROPS.items():
    im = Image.open(S / f"{src}.png").convert("RGB").crop(box)
    im.save(C / f"{name}.png")
    sizes[name] = list(im.size)

# carol: warning dialog above the prediction overlay of the over-exposed image (output file, enlarged 2x)
warn = Image.open(S / "25b_warning_box.png").convert("RGB")
ov = Image.open(HOMES / "carol/results/batch-02-predictions/B2-S09_0001-overexposed_overlay.png").convert("RGB")
ov = ov.resize((ov.width * 2, ov.height * 2), Image.Resampling.NEAREST)
W = max(warn.width, ov.width) + 40
canvas = Image.new("RGB", (W, warn.height + ov.height + 60), "white")
canvas.paste(warn, ((W - warn.width) // 2, 15))
canvas.paste(ov, ((W - ov.width) // 2, warn.height + 40))
canvas.save(C / "carol_results.png")
sizes["carol_results"] = list(canvas.size)
(C / "manifest.json").write_text(json.dumps(sizes, indent=1))

# preview copies with a fraction grid for placing markers
for name in sizes:
    im = Image.open(C / f"{name}.png").convert("RGB")
    d = ImageDraw.Draw(im)
    for i in range(1, 10):
        x, y = im.width * i / 10, im.height * i / 10
        d.line([(x, 0), (x, im.height)], fill=(0, 200, 255), width=1)
        d.line([(0, y), (im.width, y)], fill=(0, 200, 255), width=1)
        d.text((x + 2, 2), f".{i}", fill=(255, 0, 255))
        d.text((2, y + 2), f".{i}", fill=(255, 0, 255))
    im.save(G / f"{name}.png")
print(sizes)
