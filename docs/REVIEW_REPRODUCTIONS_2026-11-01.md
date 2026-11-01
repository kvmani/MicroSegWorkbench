# Independent adversarial reproductions — 2026-11-01

Reviewed commit: `47449da592abdb876bae7cfbbbd2a9750d65fb79`.

These are **defect-confirming probes**, not correctness regression tests: passing means the described defect
exists. Implementers must invert the relevant expectation after fixing each issue. No application or repository
test code was changed for this review. All inputs and writes are synthetic and confined to pytest temporary folders.

From the repository root, install the normal test/training dependencies. Save the following block to a temporary
file named `msw_adversarial.py` outside the repository, then run:

```sh
python -m pytest -q /absolute/path/to/msw_adversarial.py
```

This uses `tests/oa_fixture.py` from the reviewed repository. Linux/POSIX is needed for the no-hard-links race probe.
Observed: **10 passed in 2.93 s**. The last test performs real one-epoch CPU training and then demonstrates test-set
contamination during reevaluation. The POSIX race probe injects an unsupported-hard-links error and a deterministic
thread interleaving; it does not claim that a real office SMB server was exercised.

```python
import sys, json, zipfile, errno, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
sys.path.insert(0, str(Path.cwd() / 'tests'))
from oa_fixture import make_bundle
from microseg_workbench.core.synthetic import SyntheticSpec, write_synthetic_dataset
from microseg_workbench.core.dataset import validate_folder, create_dataset_version, DatasetVersion
from microseg_workbench.core.splits import SplitConfig, DEFAULT_GROUP_REGEX
from microseg_workbench.core.atomic import write_json_atomic
from microseg_workbench.core.imageio import encode_png
from microseg_workbench.integrations.onlineannotator import import_bundle
from microseg_workbench.core.workspace import Workspace
from microseg_workbench.collab.share import SharedStore

def dataset(tmp_path):
    folder = tmp_path/'source'
    write_synthetic_dataset(folder, SyntheticSpec(specimens=3, images_per_specimen=1, size=32))
    return folder, validate_folder(folder, DEFAULT_GROUP_REGEX)

def test_changed_mask_accepted(tmp_path):
    folder, report = dataset(tmp_path)
    pair = report.pairs[0]
    pair.mask_path.write_bytes(encode_png(np.zeros((16, 16), np.uint8)))
    v = create_dataset_version(report, SplitConfig(), tmp_path/'versions', 'd')
    s = next(s for s in v.samples() if s['stem']==pair.stem)
    from PIL import Image
    assert Image.open(v.mask_path(s)).size == (16,16)
    assert (s['height'],s['width']) == (32,32)
    assert v.verify() == []

def test_unapproved_provenance_reuses_approved_version(tmp_path):
    folder, report = dataset(tmp_path)
    samples = {p.stem: {'version_status':'approved'} for p in report.pairs}
    prov = {'schema':'microseg-workbench.provenance/1', 'samples':samples, 'unapproved':[]}
    write_json_atomic(folder/'provenance.json', prov)
    first = create_dataset_version(validate_folder(folder,DEFAULT_GROUP_REGEX),SplitConfig(),tmp_path/'versions','d')
    stem = report.pairs[0].stem
    prov['unapproved']=[stem]; prov['samples'][stem]['version_status']='submitted'
    write_json_atomic(folder/'provenance.json', prov)
    second_report=validate_folder(folder,DEFAULT_GROUP_REGEX)
    assert any(i.code=='unapproved_annotations' for i in second_report.warnings)
    second = create_dataset_version(second_report,SplitConfig(),tmp_path/'versions','d')
    assert first.dataset_id==second.dataset_id
    assert second.manifest['contains_unapproved_annotations'] is False

def test_manifest_split_tamper_passes_integrity(tmp_path):
    _, report = dataset(tmp_path)
    v=create_dataset_version(report,SplitConfig(),tmp_path/'versions','d')
    for s in v.manifest['samples']: s['split']='train'
    write_json_atomic(v.root/'manifest.json',v.manifest)
    assert DatasetVersion.open(v.root).verify()==[]

def test_import_stem_writes_outside_stage(tmp_path):
    src=make_bundle(tmp_path/'original.zip',specimens=1,per=1)
    with zipfile.ZipFile(src) as z: files={n:z.read(n) for n in z.namelist()}
    m=json.loads(files['manifest.json']); m['images'][0]['stem']='../escaped'
    files['manifest.json']=json.dumps(m).encode()
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as z:
        for n,b in files.items(): z.writestr(n,b)
    import_bundle(tmp_path/'bad.zip',tmp_path/'imported')
    assert (tmp_path/'escaped.png').exists()
    assert (tmp_path/'escaped_mask.png').exists()

def test_share_materialize_traversal(tmp_path):
    store=SharedStore.create(tmp_path/'share'); ws=Workspace(tmp_path/'ws').ensure()
    src=tmp_path/'payload'; src.write_bytes(b'overwritten')
    digest=store.blobs.put_file(src)
    outside=ws.datasets_dir/'escaped'; outside.write_bytes(b'original')
    store._materialize({'../escaped':digest}, ws, ws.datasets_dir, 'safe', None)
    assert outside.read_bytes()==b'overwritten'

def test_posix_no_hardlinks_two_winners(tmp_path,monkeypatch):
    import microseg_workbench.collab.fsops as f
    barrier=threading.Barrier(2); rename=f.os.rename
    def no_link(*a,**k): raise OSError(errno.EOPNOTSUPP,'not supported')
    def race_rename(src,dst): barrier.wait(timeout=5); return rename(src,dst)
    monkeypatch.setattr(f.os,'link',no_link); monkeypatch.setattr(f.os,'rename',race_rename)
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda data:f.create_exclusive(tmp_path/'record',data),[b'A',b'B']))
    assert results==[True,True]

def test_publish_reuses_corrupt_blob(tmp_path):
    _,report=dataset(tmp_path)
    v=create_dataset_version(report,SplitConfig(),tmp_path/'versions','d')
    store=SharedStore.create(tmp_path/'share')
    src=v.image_path(v.samples()[0]); digest=store.blobs.put_file(src)
    store.blobs.path(digest).write_bytes(b'corrupt')
    store.publish_dataset(v)
    assert store.check()

def test_prediction_same_stem_overwrites(tmp_path):
    import torch
    from PIL import Image
    from microseg_workbench.training.model import UNet, ModelConfig
    from microseg_workbench.training.engine import save_checkpoint
    from microseg_workbench.training.runs import RunDir
    from microseg_workbench.training.predict import predict_folder
    torch.set_num_threads(1)
    _,r=dataset(tmp_path)
    run=RunDir(tmp_path/'run').create()
    model=UNet(ModelConfig(1,3,4,1))
    save_checkpoint(run.checkpoints/'best.pt',model,{'class_map':r.class_map.to_dict(),'epoch':1})
    inputs=tmp_path/'inputs'; inputs.mkdir()
    Image.fromarray(np.zeros((32,32),np.uint8)).save(inputs/'same.png')
    Image.fromarray(np.full((32,32),255,np.uint8)).save(inputs/'same.tif')
    report=predict_folder(run,inputs,tmp_path/'out',device='cpu')
    assert len(report['images'])==2
    assert len(list((tmp_path/'out').glob('*_pred_mask.png')))==1

def test_concurrent_prepare_deletes_active_stage(tmp_path,monkeypatch):
    import microseg_workbench.core.dataset as d
    _,report=dataset(tmp_path)
    started=threading.Event(); release=threading.Event(); copy=d.shutil.copyfile
    def paused_copy(src,dst):
        if '.staging-first-' in str(dst):
            started.set(); assert release.wait(10)
        return copy(src,dst)
    monkeypatch.setattr(d.shutil,'copyfile',paused_copy)
    with ThreadPoolExecutor(2) as pool:
        future=pool.submit(create_dataset_version,report,SplitConfig(),tmp_path/'versions','first')
        assert started.wait(5)
        create_dataset_version(report,SplitConfig(),tmp_path/'versions','second')
        release.set()
        with pytest.raises(FileNotFoundError): future.result()

def test_reevaluate_accepts_training_images_as_test(tmp_path):
    from microseg_workbench.training.trainer import run_training, evaluate_run
    from microseg_workbench.training.config import TrainConfig
    ws=Workspace(tmp_path/'ws').ensure()
    _,report=dataset(tmp_path)
    v=create_dataset_version(report,SplitConfig(),ws.datasets_dir,'d')
    run=run_training(TrainConfig(dataset_id=v.dataset_id,epochs=1,batch_size=1,base_channels=4,depth=1,
                                 patches_per_image=1,fixed_samples=0,num_threads=1,device='cpu'),ws)
    trained={s['stem'] for s in v.samples('train')}
    for s in v.manifest['samples']:
        if s['stem'] in trained: s['split']='test'
        elif s['split']=='test': s['split']='train'
    write_json_atomic(v.root/'manifest.json',v.manifest)
    result=evaluate_run(run,ws)
    assert trained==set(result['images'])
    assert result['split']=='test'
```
