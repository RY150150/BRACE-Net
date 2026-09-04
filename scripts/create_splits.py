#!/usr/bin/env python3
"""Create train/validation/test splits for converted truck-bed samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed truck-bed samples into train/validation/test sets."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed"),
        help="Converted binary dataset root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split generation.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    return parser.parse_args()


def sample_names(samples_dir: Path) -> List[str]:
    return sorted(path.stem for path in samples_dir.glob("*.npz"))


def choose_split(
    names: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio}"
        )
    if len(names) < 3:
        raise ValueError("Need at least 3 samples to produce train/val/test splits")

    rng = np.random.default_rng(seed)
    order = np.asarray(list(names), dtype=object)
    rng.shuffle(order)

    n = len(order)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = n - n_train - n_val

    while n_test < 1:
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
        n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError("Could not allocate at least one sample to each split")

    split_map: Dict[str, str] = {}
    for idx, sample in enumerate(order.tolist()):
        if idx < n_train:
            split_map[sample] = "train"
        elif idx < n_train + n_val:
            split_map[sample] = "val"
        else:
            split_map[sample] = "test"
    return split_map


def write_splits(data_root: Path, split_map: Dict[str, str]) -> None:
    split_lists = {"train": [], "val": [], "test": []}
    for sample, split in sorted(split_map.items()):
        split_lists[split].append(sample)

    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, names in split_lists.items():
        with (splits_dir / f"{split}.txt").open("w") as fh:
            for sample in names:
                fh.write(f"{sample}\n")

    with (data_root / "splits.json").open("w") as fh:
        json.dump(split_lists, fh, indent=2)

    csv_path = data_root / "splits.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample", "split"])
        writer.writeheader()
        for sample, split in sorted(split_map.items()):
            writer.writerow({"sample": sample, "split": split})


def main() -> None:
    args = parse_args()
    samples_dir = args.data_root / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(samples_dir)

    names = sample_names(samples_dir)
    if not names:
        raise FileNotFoundError(f"{samples_dir}: no .npz samples found")

    split_map = choose_split(
        names=names,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    write_splits(args.data_root, split_map)
    counts = {split: list(split_map.values()).count(split) for split in ("train", "val", "test")}
    print(
        f"Wrote splits for {len(names)} samples under {args.data_root / 'splits'} "
        f"(train={counts['train']}, val={counts['val']}, test={counts['test']})"
    )


if __name__ == "__main__":
    main()
