"""Run the BRACE-Net train/validation/test protocol across multiple seeds.

Each seed receives an isolated metrics directory. The script then writes
aggregate mean and standard-deviation statistics for the complete suite.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, pstdev


ROOT = Path(__file__).resolve().parent


def _cfg_task_name(cfg_path):
    path = Path(cfg_path)
    if len(path.parts) >= 2:
        return path.parent.name
    return "segmentation"


def _cfg_basename(cfg_path):
    return Path(cfg_path).stem


def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _as_cli_opt(key, value):
    if value is None:
        return None
    return f"{key}={value}"


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _append_log(log_path, command, returncode=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(command) + "\n")
        if returncode is not None:
            f.write(f"# returncode: {returncode}\n")


def _run(command, env, log_path, dry_run=False):
    _append_log(log_path, command)
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    _append_log(log_path, command, completed.returncode)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _flatten_numeric(prefix, value, out):
    number = _numeric(value)
    if number is not None:
        out[prefix] = number
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(child_prefix, child, out)


def _collect_metrics(seed_dir):
    files = {
        "best_val": "best_val_metrics.json",
        "val": "val_val_metrics.json",
        "test": "val_test_metrics.json",
    }
    collected = {}
    flat = {}
    for section, filename in files.items():
        path = seed_dir / filename
        if not path.exists():
            continue
        payload = _read_json(path)
        collected[section] = payload
        _flatten_numeric(section, payload, flat)
    return collected, flat


def _build_summary(seed_flats, ddof):
    by_metric = {}
    for seed, flat in seed_flats.items():
        for metric, value in flat.items():
            by_metric.setdefault(metric, []).append({"seed": seed, "value": value})

    summary = {}
    for metric, entries in sorted(by_metric.items()):
        values = [entry["value"] for entry in entries]
        if len(values) == 1:
            std_value = 0.0
        elif ddof == 0:
            std_value = pstdev(values)
        else:
            std_value = stdev(values)
        summary[metric] = {
            "mean": mean(values),
            "std": std_value,
            "n": len(values),
            "ddof": ddof,
            "values": entries,
        }
    return summary


def _write_summary_csv(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n", "ddof", "values"])
        writer.writeheader()
        for metric, stats in summary.items():
            writer.writerow(
                {
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "n": stats["n"],
                    "ddof": stats["ddof"],
                    "values": json.dumps(stats["values"], ensure_ascii=False),
                }
            )


def _checkpoint_from_seed_dir(seed_dir):
    best_metrics = seed_dir / "best_val_metrics.json"
    if not best_metrics.exists():
        raise FileNotFoundError(f"Missing best validation metrics: {best_metrics}")
    checkpoint = _read_json(best_metrics).get("checkpoint")
    if not checkpoint:
        raise RuntimeError(f"No checkpoint field found in {best_metrics}")
    checkpoint_path = ROOT / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint from {best_metrics} does not exist: {checkpoint_path}")
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser("Run BRACE-Net across multiple training seeds")
    parser.add_argument(
        "--cfg",
        default="configs/zpvehicles/brace_net.yaml",
        help="Segmentation config path",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to run")
    parser.add_argument(
        "--suite-name",
        default=None,
        help="Name under output/metrics/<task>. Default: <cfg>-multiseed-seeds...-timestamp",
    )
    parser.add_argument("--output-root", default="output/metrics", help="Root directory for metric outputs")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--epochs", type=int, default=None, help="Forwarded as epochs=<value>")
    parser.add_argument("--batch-size", type=int, default=None, help="Forwarded as batch_size=<value>")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Forwarded as val_batch_size=<value>")
    parser.add_argument("--num-workers", type=int, default=None, help="Forwarded as dataloader.num_workers=<value>")
    parser.add_argument("--num-points", type=int, default=None, help="Forwarded as dataset.common.num_points=<value>")
    parser.add_argument("--skip-val", action="store_true", help="Skip explicit best-checkpoint val evaluation")
    parser.add_argument("--skip-test", action="store_true", help="Skip held-out test evaluation")
    parser.add_argument(
        "--std-ddof",
        type=int,
        choices=[0, 1],
        default=1,
        help="0 for population std, 1 for sample std. Default: 1",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write command log only")
    args, extra_opts = parser.parse_known_args()
    return args, extra_opts


def main():
    args, extra_opts = parse_args()
    task_name = _cfg_task_name(args.cfg)
    seed_text = "-".join(str(seed) for seed in args.seeds)
    suite_name = args.suite_name or f"{_cfg_basename(args.cfg)}-multiseed-seeds{seed_text}-{_timestamp()}"
    suite_dir = (ROOT / args.output_root / task_name / suite_name).resolve()
    command_log = suite_dir / "commands.log"

    base_opts = [
        _as_cli_opt("epochs", args.epochs),
        _as_cli_opt("batch_size", args.batch_size),
        _as_cli_opt("val_batch_size", args.val_batch_size),
        _as_cli_opt("dataloader.num_workers", args.num_workers),
        _as_cli_opt("dataset.common.num_points", args.num_points),
        "wandb.use_wandb=False",
    ]
    base_opts = [opt for opt in base_opts if opt is not None] + extra_opts

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    suite_dir.mkdir(parents=True, exist_ok=True)
    seed_flats = {}
    seed_records = {}

    for seed in args.seeds:
        seed_dir = suite_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics_opt = f"metrics_output_dir={seed_dir}"
        seed_opt = f"seed={seed}"

        train_cmd = [
            args.python,
            "train.py",
            "--cfg",
            args.cfg,
            seed_opt,
            metrics_opt,
            *base_opts,
        ]
        _run(train_cmd, env, command_log, args.dry_run)

        if args.dry_run:
            checkpoint = f"<checkpoint from {seed_dir / 'best_val_metrics.json'}>"
        else:
            checkpoint = _checkpoint_from_seed_dir(seed_dir)

        if not args.skip_val:
            val_cmd = [
                args.python,
                "train.py",
                "--cfg",
                args.cfg,
                "mode=val",
                seed_opt,
                f"pretrained_path={checkpoint}",
                metrics_opt,
                *base_opts,
            ]
            _run(val_cmd, env, command_log, args.dry_run)

        if not args.skip_test:
            test_cmd = [
                args.python,
                "train.py",
                "--cfg",
                args.cfg,
                "mode=val",
                seed_opt,
                f"pretrained_path={checkpoint}",
                "dataset.val.split=test",
                metrics_opt,
                *base_opts,
            ]
            _run(test_cmd, env, command_log, args.dry_run)

        if args.dry_run:
            continue

        collected, flat = _collect_metrics(seed_dir)
        seed_flats[seed] = flat
        seed_records[str(seed)] = {
            "seed": seed,
            "output_dir": str(seed_dir),
            "checkpoint": checkpoint,
            "metrics": collected,
        }
        _write_json(seed_dir / "seed_run_summary.json", seed_records[str(seed)])

    plan_payload = {
        "cfg": args.cfg,
        "seeds": args.seeds,
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "extra_opts": base_opts,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        _write_json(suite_dir / "planned_multiseed_run.json", plan_payload)
        print(f"Dry run complete. Planned commands are in: {command_log}")
        return

    summary = _build_summary(seed_flats, args.std_ddof)
    payload = {
        **plan_payload,
        "std_ddof": args.std_ddof,
        "summary": summary,
        "seeds": seed_records,
    }
    _write_json(suite_dir / "multiseed_summary.json", payload)
    _write_summary_csv(suite_dir / "multiseed_summary.csv", summary)
    print(f"Wrote summary: {suite_dir / 'multiseed_summary.json'}")
    print(f"Wrote summary: {suite_dir / 'multiseed_summary.csv'}")


if __name__ == "__main__":
    main()
