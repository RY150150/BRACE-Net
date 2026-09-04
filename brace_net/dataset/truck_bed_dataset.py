from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .build import DATASETS


@DATASETS.register_module()
class TruckBedDataset(Dataset):
    """NPZ dataset for truck-bed versus non-bed segmentation.

    Directory layout:
        data_root/
          samples/*.npz
          splits/train.txt
          splits/val.txt
          splits/test.txt

    Each NPZ file stores:
        pos: [N, 3]
        x:   [N, 3]
        y:   [N]
    """

    classes = ["non_bed", "truck_bed"]
    num_classes = 2
    gravity_dim = 2

    def __init__(
        self,
        data_root,
        split="train",
        num_points=4096,
        normalize=True,
        augmentation=None,
        transform=None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = int(num_points)
        self.normalize = normalize
        self.augmentation = augmentation or {}
        self.use_augmentation = bool(self.augmentation.get("enabled", False)) and split == "train"
        self.transform = transform

        split_name = "val" if split == "test" and not (self.data_root / "splits" / "test.txt").exists() else split
        split_file = self.data_root / "splits" / f"{split_name}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)

        with split_file.open("r") as fh:
            self.sample_names = [line.strip() for line in fh if line.strip()]
        self.sample_paths = [self.data_root / "samples" / f"{name}.npz" for name in self.sample_names]

        missing = [str(path) for path in self.sample_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing NPZ samples: {missing[:5]}")

        self.num_per_class = self._count_labels()

    def __len__(self):
        return len(self.sample_paths)

    def _count_labels(self):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for path in self.sample_paths:
            with np.load(path) as data:
                labels = data["y"].astype(np.int64)
            counts += np.bincount(labels, minlength=self.num_classes)[: self.num_classes]
        return counts

    def _sample_indices(self, num_available, idx):
        replace = num_available < self.num_points
        if self.split == "train":
            return np.random.choice(num_available, self.num_points, replace=replace)

        rng = np.random.default_rng(idx)
        return rng.choice(num_available, self.num_points, replace=replace)

    def _normalize_pos(self, pos):
        pos = pos - pos.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(pos, axis=1))
        if scale > 0:
            pos = pos / scale
        return pos

    def _augment_pos(self, pos):
        if self.augmentation.get("rotate_z", True):
            angle = np.random.uniform(-np.pi, np.pi)
            cosval, sinval = np.cos(angle), np.sin(angle)
            rot = np.asarray(
                [
                    [cosval, -sinval, 0.0],
                    [sinval, cosval, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            pos = pos @ rot.T

        scale = self.augmentation.get("scale", [1.0, 1.0])
        if scale is not None:
            scale_min, scale_max = float(scale[0]), float(scale[1])
            if scale_max > 0 and scale_min != scale_max:
                pos = pos * np.random.uniform(scale_min, scale_max)

        shift = self.augmentation.get("shift", [0.0, 0.0, 0.0])
        if shift is not None:
            shift = np.asarray(shift, dtype=np.float32)
            if np.any(shift > 0):
                pos = pos + np.random.uniform(-shift, shift).astype(np.float32)

        jitter_sigma = float(self.augmentation.get("jitter_sigma", 0.0))
        jitter_clip = float(self.augmentation.get("jitter_clip", 0.0))
        if jitter_sigma > 0:
            noise = np.random.normal(0.0, jitter_sigma, pos.shape).astype(np.float32)
            if jitter_clip > 0:
                noise = np.clip(noise, -jitter_clip, jitter_clip)
            pos = pos + noise

        dropout_ratio = float(self.augmentation.get("dropout_ratio", 0.0))
        if dropout_ratio > 0:
            keep = np.random.random(len(pos)) > dropout_ratio
            if np.any(~keep) and np.any(keep):
                replacement = np.random.choice(np.flatnonzero(keep), size=np.sum(~keep), replace=True)
                pos = pos.copy()
                pos[~keep] = pos[replacement]

        return pos.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        path = self.sample_paths[idx]
        with np.load(path) as data:
            pos = data["pos"].astype(np.float32)
            x = data["x"].astype(np.float32)
            y = data["y"].astype(np.int64)

        choice = self._sample_indices(len(pos), idx)
        pos = pos[choice]
        x = x[choice]
        y = y[choice]

        if self.normalize:
            pos = self._normalize_pos(pos)
            x = pos.copy()

        if self.use_augmentation:
            pos = self._augment_pos(pos)
            x = pos.copy()

        data = {
            "pos": pos,
            "x": x,
            "y": y,
        }
        if self.transform is not None:
            data = self.transform(data)
        return data
