"""Command-line interface. Every UI action has a CLI equivalent for scripting and tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


from .core.augment import AugmentConfig
from .core.dataset import DatasetVersion, create_dataset_version, list_dataset_versions, validate_folder
from .core.imageio import encode_png, load_image, load_mask_values, to_unit_float
from .core.render import augmentation_preview
from .core.splits import DEFAULT_GROUP_REGEX, SplitConfig, SplitError
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
    split = SplitConfig(train=args.train, val=args.val, test=args.test, seed=args.seed, group_regex=args.group_regex,
                        source=args.split_source)
    try:
        version = create_dataset_version(report, split, ws.datasets_dir, args.name)
    except SplitError as exc:
        print(f"Cannot split: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset version: {version.dataset_id}")
    print(f"Split counts: {version.manifest['split_counts']}")
    for group, splits in version.manifest["groups"].items():
        print(f"  group {group}: {splits[0]}")
    return 0


def cmd_import_oa(args: argparse.Namespace) -> int:
    from .integrations.onlineannotator import OAImportError, import_bundle

    out = Path(args.out) if args.out else _ws(args).root / "imports" / Path(args.bundle).stem
    try:
        report = import_bundle(Path(args.bundle), out, progress=None if args.quiet else print)
    except OAImportError as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {report.images} image(s) ({report.mask_style} masks) into {report.folder}")
    print("Classes: " + ", ".join(f"{c.value}={c.name}" for c in report.class_map.classes))
    print(f"Splits assigned in OnlineAnnotator: {report.splits}")
    for w in report.warnings:
        print(f"WARNING: {w}")
    print(f"Next: msw prepare {report.folder} --name <name> [--split-source annotator]")
    return 0


def cmd_hydride_stats(args: argparse.Namespace) -> int:
    from .integrations.hydrideseg import HydrideSegUnavailable, analyse_predictions, analyse_run

    ws = _ws(args)
    try:
        if args.predictions:
            report = analyse_predictions(Path(args.predictions), args.class_name, args.um_per_px)
            for name, stats in report["images"].items():
                m = stats["scalar_metrics"]
                print(f"{name}: " + ", ".join(f"{k} {v:.4g}" for k, v in list(m.items())[:6]
                                              if isinstance(v, (int, float))))
        else:
            report = analyse_run(ws.runs_dir / args.run, ws.datasets_dir, args.class_name, args.um_per_px)
            for stem, d in report["per_image_distances"].items():
                print(f"{stem}: area fraction truth {d['mask_area_fraction_gt']:.4f} pred "
                      f"{d['mask_area_fraction_pred']:.4f}, count {d['hydride_count_gt']:.0f}/"
                      f"{d['hydride_count_pred']:.0f}, orientation KS {d['hydride_orientation_ks']:.3f}, "
                      f"size KS {d['hydride_size_ks']:.3f}")
    except (HydrideSegUnavailable, ValueError, FileNotFoundError) as exc:
        print(f"Cannot analyse: {exc}", file=sys.stderr)
        return 1
    print(f"Computed with {report['computed_with']}  [UNREVIEWED]")
    return 0


def cmd_architectures(args: argparse.Namespace) -> int:
    from .training.architectures import ARCHITECTURES

    for a in ARCHITECTURES.values():
        print(f"{a.name:12s} {a.title}\n{'':12s} {a.description}\n{'':12s} size knobs: {a.size_help} (from {a.module})")
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
    from .training.device import choose_device, cuda_diagnosis, detect_devices
    from .training.monitor import sample_system

    for d in detect_devices():
        print(d.label() + (f"  compute capability {d.compute_capability}" if d.compute_capability else ""))
    print(cuda_diagnosis())
    print(choose_device("auto")[2])
    snap = sample_system()
    for g in snap.get("gpus", []):
        tag = "  [SIMULATED]" if g["simulated"] else ""
        print(f"GPU {g['index']} {g['name']}: utilisation {g['utilization_percent']}%, memory "
              f"{g['memory_used_mb']}/{g['memory_total_mb']} MB ({g['source']}){tag}")
    return 0


def _train_config(args: argparse.Namespace):
    from .training.config import TrainConfig, preset_config

    if args.config:
        payload = json.loads(Path(args.config).read_text())
        payload["dataset_id"] = args.dataset
        return TrainConfig.from_dict(payload)
    overrides = {k: getattr(args, k) for k in ("architecture", "epochs", "batch_size", "learning_rate", "seed", "device",
                                               "num_threads", "mixed_precision", "early_stopping_patience") if getattr(args, k) is not None}
    return preset_config(args.preset, args.dataset, **overrides)


def _print_event(event: dict[str, Any]) -> None:
    kind = event["type"]
    if kind == "start":
        amp = ", mixed precision" if event.get("mixed_precision") else ""
        print(f"{event['note']}  ({event['parameters']:,} parameters, batch size {event.get('batch_size')}{amp}, "
              f"{event['steps_per_epoch']} steps/epoch)")
    elif kind == "resume":
        print(f"Resuming after epoch {event['from_epoch']}. {event['note']}")
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
    from dataclasses import replace

    from .training.trainer import run_training

    cfg = _train_config(args)
    repeats = max(1, getattr(args, "repeat_seeds", 1) or 1)
    if repeats > 1 and args.run_id:
        print("--run-id cannot be combined with --repeat-seeds", file=sys.stderr)
        return 1
    ws = _ws(args)
    runs = []
    for k in range(repeats):
        run_cfg = replace(cfg, seed=cfg.seed + k)
        if repeats > 1:
            print(f"--- run {k + 1}/{repeats} (seed {run_cfg.seed}) ---")
        try:
            run = run_training(run_cfg, ws, run_id=args.run_id, on_event=None if args.quiet else _print_event)
        except Exception as exc:
            print(f"Training failed: {exc}", file=sys.stderr)
            return 1
        print(f"Run directory: {run.root}  state: {run.status().get('state')}")
        if run.status().get("state") != "completed":
            return 0 if run.status().get("state") == "cancelled" else 1
        runs.append(run)
    if len(runs) > 1:
        from .training.compare import compare_runs

        summary = compare_runs(runs)["seed_summary"]
        if summary:
            st = summary["test_miou_fg"] or summary["test_miou"]
            print(f"Over {len(runs)} seeds: test mIoU(fg) {_fmt(st['mean'])} +/- {_fmt(st['sd'])} "
                  f"(min {_fmt(st['min'])}, max {_fmt(st['max'])})  [UNREVIEWED]")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from .training.trainer import ResumeError, resume_training

    try:
        run = resume_training(_ws(args), args.run, on_event=None if args.quiet else _print_event)
    except ResumeError as exc:
        print(f"Cannot resume: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    print(f"Run directory: {run.root}  state: {run.state()}")
    return 0 if run.state() in ("completed", "cancelled") else 1


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
        resumable = "  (resumable: msw resume " + run.run_id + ")" if run.resumable() else ""
        print(f"{run.run_id}  {run.state()}  {run.info().get('dataset', {}).get('dataset_id')}  "
              f"{st.get('message', '')}{resumable}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .training.compare import comparison_csv, compare_runs
    from .training.runs import RunDir

    ws = _ws(args)
    runs = [RunDir(ws.runs_dir / r) for r in args.runs]
    missing = [r.run_id for r in runs if not r.info()]
    if missing:
        print(f"Unknown run(s): {', '.join(missing)}", file=sys.stderr)
        return 1
    comparison = compare_runs(runs)
    for w in comparison["warnings"]:
        print(f"WARNING: {w}")
    for s in comparison["runs"]:
        classes = "  ".join(f"{k} {_fmt(v)}" for k, v in s["test_class_iou"].items())
        print(f"{s['run_id']}  [{s['dataset_id']}]  test mIoU {_fmt(s['test_miou'])}  "
              f"mIoU(fg) {_fmt(s['test_miou_fg'])}  best val {_fmt(s['best_val_miou_fg'])} @ {s['best_val_epoch']}  "
              f"{s['train_seconds']}s  | {classes}")
    if comparison.get("seed_summary"):
        st = comparison["seed_summary"]["test_miou_fg"] or comparison["seed_summary"]["test_miou"]
        print(f"Runs differ only by seed: test mIoU(fg) {_fmt(st['mean'])} +/- {_fmt(st['sd'])} over {st['n']} runs")
    if comparison["differences"]:
        print("Settings that differ:")
        for k, values in comparison["differences"].items():
            print(f"  {k}: " + " | ".join(str(v) for v in values))
    print("All scores are UNREVIEWED.")
    if args.csv:
        Path(args.csv).write_text(comparison_csv(comparison), encoding="utf-8")
        print(f"CSV written to {args.csv}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .training.predict import PredictionError, predict_folder
    from .training.runs import RunDir

    ws = _ws(args)
    try:
        report = predict_folder(RunDir(ws.runs_dir / args.run), Path(args.input), Path(args.out),
                                checkpoint=args.checkpoint, device=args.device,
                                progress=lambda i, n, name: print(f"[{i}/{n}] {name}", flush=True))
    except PredictionError as exc:
        print(f"Cannot predict: {exc}", file=sys.stderr)
        return 1
    for err in report["errors"]:
        print(f"SKIPPED {err['file']}: {err['error']}")
    print(f"{len(report['images'])} prediction(s) written to {args.out}  [UNREVIEWED]")
    return 0 if report["images"] else 1


def _share(args: argparse.Namespace):
    from .collab.settings import CollabSettings
    from .collab.share import SharedStore

    ws = _ws(args)
    settings = CollabSettings.load(ws)
    path = getattr(args, "share", None) or settings.share_path
    if not path:
        raise SystemExit("No shared folder configured. Run: msw share config --path <shared folder>")
    return ws, settings, SharedStore.open(Path(path))


def cmd_share(args: argparse.Namespace) -> int:
    from .collab.settings import CollabSettings
    from .collab.share import SharedStore, ShareError
    from .training.runs import RunDir

    action = args.share_command
    try:
        if action == "init":
            store = SharedStore.create(Path(args.path))
            print(f"Shared folder ready: {store.root}")
            return 0
        if action == "config":
            ws = _ws(args)
            settings = CollabSettings.load(ws)
            for key in ("path", "user", "service", "token"):
                value = getattr(args, key)
                if value is not None:
                    setattr(settings, {"path": "share_path", "service": "service_url", "token": "service_token"}
                            .get(key, key), value)
            if settings.share_path:
                SharedStore.open(Path(settings.share_path))  # validate before saving
            settings.save(ws)
            print(f"Shared folder: {settings.share_path or '-'}  user: {settings.user_name()}  "
                  f"service: {settings.service_url or '-'}")
            return 0
        ws, settings, store = _share(args)
        user = settings.user_name()
        progress = None if args.quiet else (lambda m: print(m, flush=True))
        if action == "status":
            print(f"Shared folder {store.root}")
            print("Datasets:")
            for d in store.list_datasets():
                local = "local copy" if (ws.datasets_dir / d["dataset_id"]).exists() else "not downloaded"
                print(f"  {d['dataset_id']}  {d['split_counts']}  by {d['published_by']} {d['published_at'][:10]}"
                      f"  ({local})")
            print("Runs:")
            for r in store.list_runs():
                print(f"  {r['run_id']}  [{r['review'].upper()}]  dataset {r['dataset_id']}  test mIoU "
                      f"{_fmt(r['summary']['mean_iou'])}  by {r['published_by']}")
            return 0
        if action == "publish-dataset":
            version = DatasetVersion.open(ws.datasets_dir / args.dataset)
            rec = store.publish_dataset(version, user=user, progress=progress)
            print(f"Published dataset {rec['dataset_id']} ({rec['bytes']:,} bytes)")
        elif action == "publish-run":
            rec = store.publish_run(RunDir(ws.runs_dir / args.run), tuple(args.checkpoints.split(",")), user=user,
                                    progress=progress)
            print(f"Published run {rec['run_id']} (unreviewed until someone approves it)")
        elif action == "pull-dataset":
            version = store.pull_dataset(args.dataset, ws, progress=progress)
            print(f"Dataset available locally: {version.root}")
        elif action == "pull-run":
            run = store.pull_run(args.run, ws, progress=progress)
            print(f"Run available locally: {run.root}")
        elif action == "review":
            review = store.add_review(args.run, args.decision, args.comment or "", reviewer=user)
            print(f"Recorded: {review['decision']} by {review['reviewer']}")
        elif action == "check":
            problems = store.check()
            for p in problems:
                print(f"PROBLEM: {p}")
            print("Share is consistent." if not problems else f"{len(problems)} problem(s) found.")
            return 0 if not problems else 1
        elif action == "sync":
            if not settings.service_url:
                print("No metadata service configured.", file=sys.stderr)
                return 1
            from .collab.client import ServiceClient

            n = ServiceClient(settings.service_url, settings.service_token).sync_from_share(store)
            print(f"Sent {n} record(s) to {settings.service_url}.")
            return 0
        elif action == "gc":
            removed = store.blobs.remove_stale_partials(args.days * 86400)
            print(f"Removed {len(removed)} stale partial upload(s).")
            if args.unreferenced:
                found = store.remove_unreferenced_blobs(args.days * 86400, apply=args.apply)
                size = sum(n for _, n in found)
                verb = "Deleted" if args.apply else "Would delete (run again with --apply)"
                print(f"{verb}: {len(found)} unreferenced file(s), {size:,} bytes, older than {args.days} days.")
        _register_with_service(settings, store, action, args)
    except ShareError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _register_with_service(settings, store, action: str, args: argparse.Namespace) -> None:
    """Best-effort index update in the optional metadata service (the share stays authoritative)."""
    if not settings.service_url or action not in ("publish-dataset", "publish-run", "review"):
        return
    from .collab.client import update_service

    print(update_service(settings, store))


def cmd_report(args: argparse.Namespace) -> int:
    from .training.report import write_report
    from .training.runs import RunDir

    run = RunDir(_ws(args).runs_dir / args.run)
    if not run.info():
        print(f"Unknown run {args.run}", file=sys.stderr)
        return 1
    print(f"Report written to {write_report(run)}")
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
            s.add_argument("--split-source", choices=("group", "annotator"), default="group",
                           help="group: seeded split by group; annotator: splits assigned in OnlineAnnotator")
        s.set_defaults(func=func)

    s = sub.add_parser("hydride-stats", help="HydrideSegmentation feature statistics (optional integration)")
    s.add_argument("run", nargs="?", help="run whose test predictions are compared with ground truth")
    s.add_argument("--predictions", help="instead: a 'msw predict' output folder")
    s.add_argument("--class", dest="class_name", default="hydride")
    s.add_argument("--um-per-px", type=float, help="spatial calibration (micrometres per pixel)")
    s.set_defaults(func=cmd_hydride_stats)

    s = sub.add_parser("import-oa", help="import an OnlineAnnotator export (ZIP or folder)")
    s.add_argument("bundle")
    s.add_argument("--out", help="new folder (default: <workspace>/imports/<bundle name>)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_import_oa)

    sub.add_parser("datasets", help="list dataset versions").set_defaults(func=cmd_datasets)
    sub.add_parser("architectures", help="list available networks").set_defaults(func=cmd_architectures)
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
        s.add_argument("--architecture", help="network, e.g. unet or segformer (msw architectures lists them)")
        s.add_argument("--epochs", type=int)
        s.add_argument("--batch-size", type=int, help="0 = automatic (fits GPU memory)")
        s.add_argument("--mixed-precision", choices=("auto", "on", "off"))
        s.add_argument("--early-stopping", dest="early_stopping_patience", type=int,
                       help="stop after N epochs without validation improvement")
        s.add_argument("--learning-rate", type=float)
        s.add_argument("--seed", type=int)
        s.add_argument("--device", choices=("auto", "cpu", "cuda"))
        s.add_argument("--num-threads", type=int)
        s.add_argument("--run-id")
        s.add_argument("--repeat-seeds", type=int, default=1,
                       help="train N runs with seeds seed, seed+1, ... and report mean +/- sd")
        s.add_argument("--quiet", action="store_true")
        s.set_defaults(func=func, config=None)

    s = sub.add_parser("predict", help="segment new images with a trained run")
    s.add_argument("run")
    s.add_argument("input", help="folder of images")
    s.add_argument("--out", required=True, help="new output folder")
    s.add_argument("--checkpoint", choices=("best", "last"), default="best")
    s.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("report", help="write a self-contained HTML report of a run")
    s.add_argument("run")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("compare", help="compare runs side by side")
    s.add_argument("runs", nargs="+")
    s.add_argument("--csv", help="also write the table as CSV")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("resume", help="continue a cancelled, failed or interrupted run")
    s.add_argument("run")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_resume)

    s = sub.add_parser("evaluate", help="(re)evaluate a run's best checkpoint")
    s.add_argument("run")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("share", help="shared office folder: publish, download, review")
    s.add_argument("--share", help="shared folder path (default: from workspace settings)")
    s.add_argument("--quiet", action="store_true")
    ss = s.add_subparsers(dest="share_command", required=True)
    x = ss.add_parser("init", help="prepare an empty shared folder")
    x.add_argument("path")
    x = ss.add_parser("config", help="set shared folder, user name and optional service")
    x.add_argument("--path")
    x.add_argument("--user")
    x.add_argument("--service")
    x.add_argument("--token")
    ss.add_parser("status", help="list shared datasets and runs")
    x = ss.add_parser("publish-dataset")
    x.add_argument("dataset")
    x = ss.add_parser("publish-run")
    x.add_argument("run")
    x.add_argument("--checkpoints", default="best", help="comma list: best,last")
    x = ss.add_parser("pull-dataset")
    x.add_argument("dataset")
    x = ss.add_parser("pull-run")
    x.add_argument("run")
    x = ss.add_parser("review")
    x.add_argument("run")
    x.add_argument("--decision", required=True, choices=("approved", "rejected", "needs_changes"))
    x.add_argument("--comment")
    ss.add_parser("check", help="verify every published file")
    x = ss.add_parser("gc", help="remove stale partial uploads (and optionally unreferenced files)")
    x.add_argument("--days", type=float, default=7)
    x.add_argument("--unreferenced", action="store_true", help="also list files no published record uses")
    x.add_argument("--apply", action="store_true", help="actually delete the unreferenced files listed")
    ss.add_parser("sync", help="re-send all shared records to the metadata service")
    s.set_defaults(func=cmd_share)

    sub.add_parser("gui", help="open the desktop application").set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    from .plugins import load_plugins

    load_plugins()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
