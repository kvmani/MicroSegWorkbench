"""OnlineAnnotator export import: styles, integrity checks, provenance, annotator splits."""

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from microseg_workbench.cli import main
from microseg_workbench.core.dataset import create_dataset_version, validate_folder
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig, SplitError
from microseg_workbench.integrations.onlineannotator import OAImportError, import_bundle
from oa_fixture import make_bundle


@pytest.mark.parametrize("style", ["indexed", "binary", "red", "colour"])
def test_every_mask_style_imports_exact_class_numbers(tmp_path, style):
    bundle = make_bundle(tmp_path / "b.zip", style=style)
    report = import_bundle(bundle, tmp_path / "out")
    expected_values = [0, 2, 5] if style in ("indexed", "colour") else [0, 2]
    assert report.class_map.values == expected_values
    mask = np.asarray(Image.open(tmp_path / "out" / "S1_1_mask.png"))
    assert set(np.unique(mask)) <= set(expected_values) and (mask == 2).any()
    v = validate_folder(tmp_path / "out", DEFAULT_GROUP_REGEX)
    assert v.ok, v.issues
    assert v.provenance["samples"]["S1_1"]["approved_by"] == "rev@lab"


def test_extracted_folder_and_split_folders_layout(tmp_path):
    splits = {f"S{s}_{i}": ("train" if s <= 2 else "val" if s == 3 else "test") for s in range(1, 5) for i in (1, 2)}
    bundle = make_bundle(tmp_path / "b.zip", layout="split_folders", splits=splits)
    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(tmp_path / "extracted")
    report = import_bundle(tmp_path / "extracted", tmp_path / "out")
    assert report.splits == {"train": 4, "val": 2, "test": 2}


def test_tampered_file_and_inconsistent_counts_abort(tmp_path):
    bundle = make_bundle(tmp_path / "b.zip")
    with zipfile.ZipFile(bundle) as zf:
        files = {n: zf.read(n) for n in zf.namelist()}
    files["pairs/S1_1.png"] = files["pairs/S2_1.png"]
    with zipfile.ZipFile(tmp_path / "t.zip", "w") as zf:
        for n, b in files.items():
            zf.writestr(n, b)
    with pytest.raises(OAImportError, match="SHA-256"):
        import_bundle(tmp_path / "t.zip", tmp_path / "out")
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob(".staging-*"))

    manifest = json.loads(files["manifest.json"])
    manifest["images"][0]["class_pixels"]["2"] += 1
    files["manifest.json"] = json.dumps(manifest).encode()
    files["pairs/S1_1.png"] = zipfile.ZipFile(bundle).read("pairs/S1_1.png")
    with zipfile.ZipFile(tmp_path / "c.zip", "w") as zf:
        for n, b in files.items():
            zf.writestr(n, b)
    with pytest.raises(OAImportError, match="inconsistent"):
        import_bundle(tmp_path / "c.zip", tmp_path / "out")


def test_unknown_colour_and_wrong_schema(tmp_path):
    bundle = make_bundle(tmp_path / "b.zip", style="colour")
    with zipfile.ZipFile(bundle) as zf:
        files = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(files["manifest.json"])
    manifest["classes"][1]["color"] = "#00FF00"  # pore colour no longer matches the masks
    files["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(tmp_path / "u.zip", "w") as zf:
        for n, b in files.items():
            zf.writestr(n, b)
    with pytest.raises(OAImportError, match="matches no class"):
        import_bundle(tmp_path / "u.zip", tmp_path / "out")
    manifest["schema"] = "something/2"
    with zipfile.ZipFile(tmp_path / "s.zip", "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(OAImportError, match="Unsupported manifest schema"):
        import_bundle(tmp_path / "s.zip", tmp_path / "out2")


def test_unapproved_masks_flow_into_dataset_and_annotator_splits(tmp_path):
    good = {f"S{s}_{i}": ("train" if s <= 2 else "val" if s == 3 else "test") for s in range(1, 5) for i in (1, 2)}
    bundle = make_bundle(tmp_path / "b.zip", unapproved=("S2_1",), splits=good)
    report = import_bundle(bundle, tmp_path / "out")
    assert report.unapproved == ["S2_1"] and report.warnings
    v = validate_folder(tmp_path / "out", DEFAULT_GROUP_REGEX)
    assert "unapproved_annotations" in {i.code for i in v.issues} and v.ok
    version = create_dataset_version(v, SplitConfig(source="annotator"), tmp_path / "ds", "oa")
    assert version.manifest["contains_unapproved_annotations"] is True
    assert {s["stem"]: s["split"] for s in version.samples()} == good
    assert version.samples()[0]["annotation_provenance"]["annotated_by"] == "ann@lab"
    assert version.manifest["annotation_source"]["project"]["name"] == "Zr hydrides"

    leaky = dict(good, S1_2="test")  # same specimen S1 in train and test
    import_bundle(make_bundle(tmp_path / "l.zip", splits=leaky), tmp_path / "leaky")
    v2 = validate_folder(tmp_path / "leaky", DEFAULT_GROUP_REGEX)
    with pytest.raises(SplitError, match="same group"):
        create_dataset_version(v2, SplitConfig(source="annotator"), tmp_path / "ds", "oa")
    import_bundle(make_bundle(tmp_path / "n.zip"), tmp_path / "nosplit")
    v3 = validate_folder(tmp_path / "nosplit", DEFAULT_GROUP_REGEX)
    with pytest.raises(SplitError, match="no train/val/test split"):
        create_dataset_version(v3, SplitConfig(source="annotator"), tmp_path / "ds", "oa")


def test_cli_import_and_prepare(tmp_path, capsys):
    good = {f"S{s}_{i}": ("train" if s <= 2 else "val" if s == 3 else "test") for s in range(1, 5) for i in (1, 2)}
    bundle = make_bundle(tmp_path / "b.zip", splits=good)
    ws = str(tmp_path / "ws")
    assert main(["--workspace", ws, "import-oa", str(bundle), "--quiet"]) == 0
    folder = tmp_path / "ws" / "imports" / "b"
    assert main(["--workspace", ws, "prepare", str(folder), "--name", "oa", "--split-source", "annotator"]) == 0
    assert main(["--workspace", ws, "import-oa", str(bundle), "--quiet"]) == 1  # destination exists
