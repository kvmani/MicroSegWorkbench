"""Synthetic Windows-pilot regression; no office images or exports required."""
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oa_fixture import make_bundle
from microseg_workbench.core.dataset import DatasetVersion, create_dataset_version, validate_folder
from microseg_workbench.core.imageio import load_image, load_mask_values
from microseg_workbench.core.splits import DEFAULT_GROUP_REGEX, SplitConfig
from microseg_workbench.integrations.onlineannotator import OAImportError, import_bundle


@pytest.fixture
def pilot_bundle(tmp_path):
    original = make_bundle(tmp_path / 'original.zip', specimens=13, per=7)
    bundle = tmp_path / 'pilot.zip'
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(bundle, 'w') as dst:
        manifest = json.loads(src.read('manifest.json'))
        stems = ['Zr2p5Nb_Before_10X', 'Zr2p5Nb_before_10X', 'Zr2p5Nb_before_10X__2']
        for i, rec in enumerate(manifest['images']):
            # Keep distinct original member paths even when display stems collide.
            rec['stem'] = stems[i] if i < 3 else rec['stem']
            rec['split'] = 'train' if i < 73 else ('val' if i < 82 else 'test')
            rec['version_status'] = 'submitted' if i == 1 else 'approved'
            if i == 1:
                rec['approved_by'] = rec['approved_at'] = None
            image = src.read(rec['image_file'])
            if i < 18:
                import io
                gray = np.asarray(Image.open(io.BytesIO(image)))
                rgb = np.stack([gray, gray // 2, 255 - gray], axis=-1)
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format='PNG')
                image = buf.getvalue()
            rec['image_sha256'] = hashlib.sha256(image).hexdigest()
            dst.writestr(rec['image_file'], image)
            dst.writestr(rec['mask_file'], src.read(rec['mask_file']))
        dst.writestr('manifest.json', json.dumps(manifest))
    return bundle


def test_pilot_cli_integrity_and_inference(pilot_bundle, tmp_path, capsys):
    from microseg_workbench.cli import main
    from microseg_workbench.core.workspace import Workspace
    from microseg_workbench.training.config import TrainConfig
    from microseg_workbench.training.trainer import run_training
    from microseg_workbench.training.predict import predict_folder
    from microseg_workbench.training.multi_predict import compare_models_on_image

    folder = tmp_path / 'imported'
    assert main(['import-oa', str(pilot_bundle), '--out', str(folder), '--quiet']) == 0
    assert 'Renamed 1 pair(s)' in capsys.readouterr().out
    provenance = json.loads((folder / 'provenance.json').read_text())
    assert len(provenance['samples']) == 91
    with zipfile.ZipFile(pilot_bundle) as src:
        original_records = json.loads(src.read('manifest.json'))['images']
    names = ['Zr2p5Nb_Before_10X', 'Zr2p5Nb_before_10X__3', 'Zr2p5Nb_before_10X__2'] + [
        r['stem'] for r in original_records[3:]]
    assert set(names) == set(provenance['samples'])
    assert names[:3] == ['Zr2p5Nb_Before_10X', 'Zr2p5Nb_before_10X__3', 'Zr2p5Nb_before_10X__2']
    assert len(set(n.casefold() for n in names)) == 91
    assert provenance['unapproved'] == ['Zr2p5Nb_before_10X__3']
    with zipfile.ZipFile(pilot_bundle) as src:
        records = json.loads(src.read('manifest.json'))['images']
        for name, rec in zip(names, records):
            p = provenance['samples'][name]
            assert p['original_stem'] == rec['stem'] and p['output_stem'] == name
            assert p['image_file'] == rec['image_file'] and p['mask_file'] == rec['mask_file']
            for key in ('version', 'version_status', 'approved_by', 'approved_at', 'split'):
                assert p[key] == rec[key]
            assert (folder / f'{name}.png').read_bytes() == src.read(rec['image_file'])
            assert (folder / f'{name}_mask.png').read_bytes() == src.read(rec['mask_file'])
    assert len({provenance['samples'][n]['image_sha256'] for n in names[:3]}) == 3
    assert len({provenance['samples'][n]['mask_png_sha256'] for n in names[:3]}) == 3
    assert any(i.code == 'mixed_channels' for i in validate_folder(folder, DEFAULT_GROUP_REGEX).errors)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    ws = Workspace(tmp_path / 'ws').ensure()
    assert main(['--workspace', str(ws.root), 'prepare', str(folder), '--name', 'pilot',
                 '--channel-policy', 'grayscale', '--split-source', 'image',
                 '--train', '0.8', '--val', '0.1', '--test', '0.1', '--seed', '42']) == 0
    version = DatasetVersion.open(next(ws.datasets_dir.iterdir()))
    assert version.manifest['split_counts'] == {'train': 73, 'val': 9, 'test': 9}
    assert not version.verify()
    for sample in version.samples():
        with Image.open(sample['source_image']) as image:
            expected = np.asarray(image.convert('L'))
        actual = load_image(version.image_path(sample)).array[..., 0]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(load_mask_values(version.mask_path(sample)),
                                      load_mask_values(sample['source_mask']))
        assert actual.shape == (48, 48)
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    run = run_training(TrainConfig(dataset_id=version.dataset_id, epochs=1, base_channels=4,
                                  depth=1, patches_per_image=1, num_threads=1, device='cpu'), ws)
    assert run.status()['state'] == 'completed'
    new = tmp_path / 'new'; new.mkdir()
    Image.fromarray(np.full((48, 48, 3), [10, 80, 200], np.uint8)).save(new / 'rgb.png')
    prediction = predict_folder(run, new, tmp_path / 'prediction', device='cpu')
    assert not prediction['errors'] and len(prediction['images']) == 1
    assert prediction['channel_policy'] == 'grayscale'
    assert compare_models_on_image([run], new / 'rgb.png')['models']
    # Older checkpoints omit the policy and retain native-channel rejection.
    import torch
    checkpoint = torch.load(run.checkpoints / 'best.pt', weights_only=True)
    checkpoint.pop('channel_policy')
    torch.save(checkpoint, run.checkpoints / 'legacy.pt')
    legacy = predict_folder(run, new, tmp_path / 'legacy-prediction', checkpoint='legacy', device='cpu')
    assert not legacy['images'] and 'expects 1' in legacy['errors'][0]['error']


def test_ambiguous_member_and_hash_abort(pilot_bundle, tmp_path):
    with zipfile.ZipFile(pilot_bundle, 'a') as zf:
        member = json.loads(zf.read('manifest.json'))['images'][0]['image_file']
        with pytest.warns(UserWarning):
            zf.writestr(member, b'wrong')
    with pytest.raises(OAImportError, match='ambiguous'):
        import_bundle(pilot_bundle, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_grayscale_preserves_16bit_and_reports_mixed_depth(synthetic_folder):
    paths = sorted(p for p in synthetic_folder.glob('*.png') if not p.stem.endswith('_mask'))
    image = np.asarray(Image.open(paths[0])).astype(np.uint16) * 257
    Image.fromarray(image).save(paths[0])
    loaded = load_image(paths[0], 'grayscale')
    assert loaded.array.dtype == np.uint16
    np.testing.assert_array_equal(loaded.array[..., 0], image)
    assert any(i.code == 'mixed_bit_depth' for i in validate_folder(
        synthetic_folder, DEFAULT_GROUP_REGEX, 'grayscale').errors)
    for path in paths[1:]:
        pixels = np.asarray(Image.open(path)).astype(np.uint16) * 257
        Image.fromarray(pixels).save(path)
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX, 'grayscale')
    assert report.ok
    version = create_dataset_version(report, SplitConfig(), synthetic_folder.parent / 'versions', 'sixteen')
    for sample in version.samples():
        assert sample['dtype'] == 'uint16' and sample['channels'] == 1
        assert version.image_path(sample).read_bytes() == Path(sample['source_image']).read_bytes()
    assert not version.verify()


def _wait(app, condition):
    end = time.monotonic() + 120
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return
        time.sleep(.01)
    raise AssertionError('GUI operation timed out')


@pytest.mark.gui
def test_pilot_gui(pilot_bundle, tmp_path):
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    from microseg_workbench.gui.app import MainWindow
    from microseg_workbench.core.workspace import Workspace
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Workspace(tmp_path / 'ws'), interactive=False)
    win.show()
    data = win.data_page
    data.import_bundle(pilot_bundle)
    _wait(app, lambda: data.report is not None and not data.busy and data.thumbs_ready)
    assert not data.report.ok
    assert any('Renamed 1' in w for w in data.import_warnings)
    data.grayscale.setChecked(True)
    assert win.augment_page.grayscale.isChecked()
    data.check_folder()
    _wait(app, lambda: data.report is not None and not data.busy and data.thumbs_ready)
    assert data.report.ok
    data.split_source.setCurrentIndex(data.split_source.findData('image'))
    data.train.setValue(.8); data.val.setValue(.1); data.test.setValue(.1)
    assert '73 images' in data.split_preview.text()
    data.create_version()
    _wait(app, lambda: data.last_version_id is not None and not data.busy)
    assert data.last_version_id in data.result.text()
    assert win.train_page.dataset.currentData() == data.last_version_id
    data.copy_id.click()
    assert app.clipboard().text() == data.last_version_id
    win.go_to(1)
    win.augment_page.run_check()
    _wait(app, lambda: not win.augment_page.checking)
    win.go_to(2)
    train = win.train_page
    train.epochs.setValue(1)
    train.device.setCurrentIndex(train.device.findData('cpu'))
    train.start()
    _wait(app, lambda: train.final_state is not None)
    assert train.final_state == 'completed', (train.stderr, win.ctx.last_error)
    from microseg_workbench.training.runs import RunDir
    run = RunDir(next(p for p in win.ctx.workspace.runs_dir.iterdir() if not p.name.startswith('.')))
    new = tmp_path / 'gui-new'; new.mkdir()
    Image.fromarray(np.full((48, 48, 3), [30, 90, 160], np.uint8)).save(new / 'rgb.png')
    win.go_to(3)
    win.results_page.predict(run, new, tmp_path / 'gui-predictions')
    _wait(app, lambda: not win.results_page.predicting)
    assert not win.results_page.last_prediction['errors']
    assert len(win.results_page.last_prediction['images']) == 1
    from microseg_workbench.cli import main
    assert main(['--workspace', str(win.ctx.workspace.root), 'prepare', data.folder.text(),
                 '--name', 'cli-version', '--channel-policy', 'grayscale', '--split-source', 'image']) == 0
    win.go_to(2)  # discover the CLI version through the normal navigation path
    ids = [win.train_page.dataset.itemData(i) for i in range(win.train_page.dataset.count())]
    cli_id = next(i for i in ids if i.startswith('cli-version-'))
    win.train_page.dataset.setCurrentIndex(win.train_page.dataset.findData(cli_id))
    assert win.train_page.dataset.currentData() == cli_id
    win.close()


def test_native_dataset_backward_compatibility(synthetic_folder, tmp_path):
    report = validate_folder(synthetic_folder, DEFAULT_GROUP_REGEX)
    version = create_dataset_version(report, SplitConfig(), tmp_path / 'versions', 'native')
    manifest = version.manifest
    manifest.pop('channel_policy')
    for sample in manifest['samples']:
        sample.pop('channel_policy')
        sample.pop('source_image_sha256')
    (version.root / 'manifest.json').write_text(json.dumps(manifest))
    assert not DatasetVersion.open(version.root).verify()


def test_zip_and_extracted_names_match(pilot_bundle, tmp_path):
    extracted = tmp_path / 'extracted'
    with zipfile.ZipFile(pilot_bundle) as zf:
        zf.extractall(extracted)
    a = import_bundle(pilot_bundle, tmp_path / 'zip-import')
    b = import_bundle(extracted, tmp_path / 'folder-import')
    assert a.images == b.images == 91 and a.warnings == b.warnings
    for p in a.folder.iterdir():
        if p.name != 'provenance.json':
            assert p.read_bytes() == (b.folder / p.name).read_bytes()
