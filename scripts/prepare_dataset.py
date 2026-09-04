#!/usr/bin/env python3
"""Convert ZPVehicles fence-truck labels into binary truck-bed labels.

The source dataset stores one sample per directory, with ``labeled_3D.txt`` rows
formatted as ``x y z label``. We remap one or more source labels to
``truck_bed=1`` and everything else to ``non_bed=0``, then save each sample as a
compressed NPZ file that later dataloaders can sample from directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a binary truck-bed/non-bed ZPVehicles dataset."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/raw/Fence_truck"),
        help="Source Fence_truck directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for converted NPZ files.",
    )
    parser.add_argument(
        "--positive-labels",
        dest="truck_bed_labels",
        type=int,
        nargs="+",
        default=[2],
        help="Original point labels treated as truck-bed positives.",
    )
    parser.add_argument(
        "--feature",
        choices=("xyz",),
        default="xyz",
        help="Feature array saved as x. pos is always xyz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output NPZ files.",
    )
    return parser.parse_args()


def iter_label_files(src: Path) -> Iterable[Path]:
    for sample_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        path = sample_dir / "labeled_3D.txt"
        if path.is_file():
            yield path


def load_labeled_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xyz: List[List[float]] = []
    labels: List[int] = []
    with path.open("r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
                labels.append(int(parts[3]))
            except ValueError:
                continue
    if not xyz:
        raise ValueError(f"{path}: no usable labeled rows found")
    return np.asarray(xyz, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def remap_binary(labels: np.ndarray, truck_bed_labels: Sequence[int]) -> np.ndarray:
    positive = np.isin(labels, np.asarray(truck_bed_labels, dtype=np.int64))
    return positive.astype(np.uint8)


def convert_one(
    label_path: Path,
    dst_samples: Path,
    truck_bed_labels: Sequence[int],
    feature: str,
    overwrite: bool,
) -> Dict[str, object]:
    sample_name = label_path.parent.name
    out_path = dst_samples / f"{sample_name}.npz"
    if out_path.exists() and not overwrite:
        with np.load(out_path) as data:
            y = data["y"]
            orig_labels = data["orig_labels"] if "orig_labels" in data else None
        counts = Counter(orig_labels.tolist()) if orig_labels is not None else {}
        positive = int(y.sum())
        total = int(y.shape[0])
        return {
            "sample": sample_name,
            "label_file": str(label_path),
            "output": str(out_path),
            "num_points": total,
            "num_positive": positive,
            "positive_ratio": positive / total if total else 0.0,
            "orig_label_counts": dict(sorted(counts.items())),
            "skipped_existing": True,
        }

    xyz, orig_labels = load_labeled_points(label_path)
    y = remap_binary(orig_labels, truck_bed_labels)
    x = xyz if feature == "xyz" else xyz

    np.savez_compressed(
        out_path,
        pos=xyz.astype(np.float32),
        x=x.astype(np.float32),
        y=y.astype(np.uint8),
        orig_labels=orig_labels.astype(np.int16),
        source_label_file=np.asarray(str(label_path)),
    )

    counts = Counter(orig_labels.tolist())
    positive = int(y.sum())
    total = int(y.shape[0])
    return {
        "sample": sample_name,
        "label_file": str(label_path),
        "output": str(out_path),
        "num_points": total,
        "num_positive": positive,
        "positive_ratio": positive / total if total else 0.0,
        "orig_label_counts": dict(sorted(counts.items())),
        "skipped_existing": False,
    }


def write_manifest(dst: Path, rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    manifest = {
        "label_map": {"0": "non_bed", "1": "truck_bed"},
        "source": str(args.src),
        "truck_bed_labels": list(args.truck_bed_labels),
        "feature": args.feature,
        "samples": list(rows),
    }
    with (dst / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    csv_path = dst / "manifest.csv"
    fieldnames = [
        "sample",
        "label_file",
        "output",
        "num_points",
        "num_positive",
        "positive_ratio",
        "orig_label_counts",
        "skipped_existing",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(args.src)

    args.dst.mkdir(parents=True, exist_ok=True)
    samples_dir = args.dst / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    label_paths = list(iter_label_files(args.src))
    if not label_paths:
        raise FileNotFoundError(f"{args.src}: no labeled_3D.txt files found")

    rows = []
    for label_path in label_paths:
        row = convert_one(
            label_path=label_path,
            dst_samples=samples_dir,
            truck_bed_labels=args.truck_bed_labels,
            feature=args.feature,
            overwrite=args.overwrite,
        )
        rows.append(row)
        print(
            f"{row['sample']}: points={row['num_points']} "
            f"truck_bed={row['num_positive']} ratio={row['positive_ratio']:.4f} "
            f"labels={row['orig_label_counts']} -> {row['output']}"
        )

    write_manifest(args.dst, rows, args)
    print(f"Wrote {len(rows)} samples to {args.dst}")
    print(f"Label map: 0=non_bed, 1=truck_bed from source labels {list(args.truck_bed_labels)}")


if __name__ == "__main__":
    main()
