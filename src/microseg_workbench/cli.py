"""Command-line interface. Every UI action has a CLI equivalent for scripting and tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .core.augment import AugmentConfig
from .core.dataset import DatasetVersion, create_dataset_version, list_dataset_versions, validate_folder
from .core.imageio import encode_png, load_image, load_mask_values, to_unit_float
from .core.render import augmentation_preview
from .core.splits import DEFAULT_GROUP_REGEX, SplitConfig
from .core.synthetic import SyntheticSpec, write_synthetic_dataset
from .core.workspace import Workspace


def _ws(args: argparse.Namespace) -> Workspace:
    return (Workspace(Path(args.workspace)) if args.workspace else Workspace.default()).ensure()


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def cmd_demo_data(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else _ws(args).demo_dir
    stems = write_synthetic_dataset(out, SyntheticSpec(args.specimens, args.per_specimen, args.size, args.seed))
    print(f"Wrote {len(stems)} SYNTHETIC image/mask pairs to {out}")
    return 0


def _print_report(report) -> None:
    print(report.summary())
    if report.class_map:
        print("Classes: " + ", ".join(f"{c.value}={c.name}" for c in report.class_map.classes)
              + f"  (mask encoding: {report.class_map.mask_encoding})")
    for issue in report.issues:
        where = f"[{issue.stem}] " if issue.stem else ""
        print(f"  {issue.severity.upper():7s} {issue.code}: {where}{issue.message}")


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_folder(Path(args.folder), args.group_regex)
    _print_report(report)
    groups: dict[str, int] = {}
    for p in report.pairs:
        if p.ok:
            groups[p.group] = groups.get(p.group, 0) + 1
    print("Groups: " + ", ".join(f"{g} ({n})" for g, n in sorted(groups.items())))
    return 0 if report.ok else 1


def cmd_prepare(args: argparse.Namespace) -> int:
    ws = _ws(args)
    report = validate_folder(Path(args.folder), args.group_regex)
    _print_report(report)
    if not report.ok:
        print("Not creating a dataset version: fix the errors above first.", file=sys.stderr)
        return 1
    split = SplitConfig(train=args.train, val=args.val, test=args.test, seed=args.seed, group_regex=args.group_regex)
    version = create_dataset_version(report, split, ws.datasets_dir, args.name)
    print(f"Dataset version: {version.dataset_id}")
    print(f"Split counts: {version.manifest['split_counts']}")
    for group, splits in version.manifest["groups"].items():
        print(f"  group {group}: {splits[0]}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    for v in list_dataset_versions(_ws(args).datasets_dir):
        print(f"{v.dataset_id}  {v.manifest['split_counts']}  classes={v.class_map.names}")
    return 0


def cmd_preview_aug(args: argparse.Namespace) -> int:
    from .training.data import value_to_channel_lut

    version = DatasetVersion.open(_ws(args).datasets_dir / args.dataset)
    samples = version.samples("train") or version.samples()
    sample = next((s for s in samples if s["stem"] == args.stem), None) if args.stem else samples[0]
    if sample is None:
        print(f"No sample {args.stem!r} in {args.dataset}", file=sys.stderr)
        return 1
    cfg = AugmentConfig.from_dict(json.loads(Path(args.augment).read_text())) if args.augment else AugmentConfig()
    image = to_unit_float(load_image(version.image_path(sample)).array)
    labels = value_to_channel_lut(version.class_map)[load_mask_values(version.mask_path(sample))]
    grid, captions = augmentation_preview(image, labels, version.class_map, cfg, seed=args.seed, count=args.count)
    Path(args.out).write_bytes(encode_png(grid))
    print(f"Preview of {sample['stem']} written to {args.out}")
    for i, ops in enumerate(captions):
        print(f"  column {i}: {', '.join(ops)}")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    from .training.device import choose_device, detect_devices

    for d in detect_devices():
        print(d.label())
    print(choose_device("auto")[2])
    return 0


def _train_config(args: argparse.Namespace):
    from .training.config import TrainConfig, preset_config

    if args.config:
        payload = json.loads(Path(args.config).read_text())
        payload["dataset_id"] = args.dataset
        return TrainConfig.from_dict(payload)
    overrides = {k: getattr(args, k) for k in ("epochs", "batch_size", "learning_rate", "seed", "device",
                                               "num_threads") if getattr(args, k) is not None}
    return preset_config(args.preset, args.dataset, **overrides)


def _print_event(event: dict[str, Any]) -> None:
    kind = event["type"]
    if kind == "start":
        print(f"{event['note']}  ({event['parameters']:,} parameters, {event['steps_per_epoch']} steps/epoch)")
    elif kind == "epoch":
        res = event["resources"]
        mem = f"  RAM {res['process_rss_mb']} MB" if "process_rss_mb" in res else ""
        print(f"epoch {event['epoch']}/{event['epochs']}  train loss {_fmt(event['train_loss'])}  "
              f"val loss {_fmt(event['val_loss'])}  val mIoU(fg) {_fmt(event['val_mean_iou_foreground'])}"
              f"{'  *best*' if event['is_best'] else ''}  {event['epoch_seconds']:.1f}s{mem}", flush=True)
    elif kind == "evaluation" and event["metrics"]:
        m = event["metrics"]
        print(f"{event['split']} set: mIoU {_fmt(m['mean_iou'])}  mIoU(fg) {_fmt(m['mean_iou_foreground'])}  "
              f"pixel accuracy {_fmt(m['pixel_accuracy'])}  [UNREVIEWED]")
    elif kind == "end" and event["state"] != "completed":
        print(f"Run {event['state']}: {event.get('error', '')}")


def cmd_train(args: argparse.Namespace) -> int:
    from .training.trainer import run_training

    cfg = _train_config(args)
    try:
        run = run_training(cfg, _ws(args), run_id=args.run_id, on_event=None if args.quiet else _print_event)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    print(f"Run directory: {run.root}  state: {run.status().get('state')}")
    return 0 if run.status().get("state") in ("completed", "cancelled") else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .training.runs import RunDir
    from .training.trainer import evaluate_run

    ws = _ws(args)
    report = evaluate_run(RunDir(ws.runs_dir / args.run), ws)
    m = report["metrics"]
    print(f"Run {args.run}: {report['split']} split, checkpoint epoch {report['checkpoint_epoch']}  [UNREVIEWED]")
    if m:
        for c in m["per_class"]:
            print(f"  {c['name']:15s} IoU {_fmt(c['iou'])}  Dice {_fmt(c['dice'])}  "
                  f"precision {_fmt(c['precision'])}  recall {_fmt(c['recall'])}")
        print(f"  mean IoU {_fmt(m['mean_iou'])}  pixel accuracy {_fmt(m['pixel_accuracy'])}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    from .training.runs import list_runs

    for run in list_runs(_ws(args).runs_dir):
        st = run.status()
        print(f"{run.run_id}  {st.get('state')}  {run.info().get('dataset', {}).get('dataset_id')}  {st.get('message', '')}")
    return 0


def cmd_quickstart(args: argparse.Namespace) -> int:
    ws = _ws(args)
    folder = ws.demo_dir
    write_synthetic_dataset(folder, SyntheticSpec())
    print(f"1/3 Generated SYNTHETIC demo data in {folder}")
    report = validate_folder(folder, DEFAULT_GROUP_REGEX)
    if not report.ok:
        _print_report(report)
        return 1
    version = create_dataset_version(report, SplitConfig(), ws.datasets_dir, "demo")
    print(f"2/3 Dataset version {version.dataset_id}: {version.manifest['split_counts']}")
    print("3/3 Training quick CPU preset")
    args.dataset = version.dataset_id
    return cmd_train(args)


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui.app import main as gui_main

    return gui_main(["--workspace", args.workspace] if args.workspace else [])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="msw", description="MicroSegWorkbench: microstructure segmentation workbench")
    p.add_argument("--workspace", help="workspace folder (default: $MSW_WORKSPACE or ~/MicroSegWorkbench-workspace)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("demo-data", help="write synthetic micrographs and masks")
    s.add_argument("--out")
    s.add_argument("--specimens", type=int, default=5)
    s.add_argument("--per-specimen", type=int, default=4)
    s.add_argument("--size", type=int, default=96)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo_data)

    for name, func, helptext in (("validate", cmd_validate, "check an image/mask folder"),
                                 ("prepare", cmd_prepare, "validate and create an immutable dataset version")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("folder")
        s.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX,
                       help="regex whose first group names the specimen/source group (default: text before first _)")
        if name == "prepare":
            s.add_argument("--name", required=True)
            s.add_argument("--train", type=float, default=0.6)
            s.add_argument("--val", type=float, default=0.2)
            s.add_argument("--test", type=float, default=0.2)
            s.add_argument("--seed", type=int, default=42)
        s.set_defaults(func=func)

    sub.add_parser("datasets", help="list dataset versions").set_defaults(func=cmd_datasets)
    sub.add_parser("devices", help="show CPU/GPU devices").set_defaults(func=cmd_devices)
    sub.add_parser("runs", help="list training runs").set_defaults(func=cmd_runs)

    s = sub.add_parser("preview-aug", help="write an augmentation preview PNG")
    s.add_argument("dataset")
    s.add_argument("--out", required=True)
    s.add_argument("--stem")
    s.add_argument("--augment", help="JSON file with augmentation options")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--count", type=int, default=6)
    s.set_defaults(func=cmd_preview_aug)

    for name, func in (("train", cmd_train), ("quickstart", cmd_quickstart)):
        s = sub.add_parser(name, help="train a model" if name == "train" else
                           "synthetic data -> dataset -> quick CPU training -> evaluation")
        if name == "train":
            s.add_argument("dataset")
        s.add_argument("--preset", default="quick_cpu")
        s.add_argument("--config", help="JSON training config (overrides preset)")
        s.add_argument("--epochs", type=int)
        s.add_argument("--batch-size", type=int)
        s.add_argument("--learning-rate", type=float)
        s.add_argument("--seed", type=int)
        s.add_argument("--device", choices=("auto", "cpu", "cuda"))
        s.add_argument("--num-threads", type=int)
        s.add_argument("--run-id")
        s.add_argument("--quiet", action="store_true")
        s.set_defaults(func=func, config=None)

    s = sub.add_parser("evaluate", help="(re)evaluate a run's best checkpoint")
    s.add_argument("run")
    s.set_defaults(func=cmd_evaluate)

    sub.add_parser("gui", help="open the desktop application").set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
