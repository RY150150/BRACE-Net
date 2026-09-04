# BRACE-Net

Official multi-seed training code for **BRACE-Net: Lightweight
False-Positive-Sensitive LiDAR Point-Cloud Segmentation for Truck-Bed
Perception**.

BRACE-Net combines three components around a lightweight hierarchical point
encoder:

- GeoLite coordinate-prior injection at encoder stages 0-4;
- training-only boundary supervision with 16-nearest-neighbor targets;
- a 64x64 XY/XZ/YZ Tri-Plane residual-logit branch.

This repository contains the final model configuration and the authorized,
processed dataset used by the paper. It does not include pretrained weights,
checkpoints, raw source files, paper drafts, web applications, or alternative
ablation models.

## Environment

The reported environment used Python 3.8.10, PyTorch 1.10.0, CUDA 11.3, and a
single NVIDIA RTX 3090 GPU.

```bash
conda create -n brace-net python=3.8.10 -y
conda activate brace-net
bash install.sh
```

The CUDA extension build requires a working CUDA 11.3 toolchain and a compiler
compatible with the selected PyTorch wheel.

## Dataset

The ready-to-use processed dataset is included under `data/processed/`. It
contains all 106 fence-truck samples used by the paper, totaling 10,352,125
annotated points, together with the exact 74/16/16 train/validation/test split.
No separate preparation step is required before running multi-seed training.

Each `.npz` file contains only the arrays read by the training pipeline:

```text
pos: float32 [N, 3]
x:   float32 [N, 3]
y:   uint8   [N]
```

The original point clouds come from ZPVehicles, and the task-specific
truck-compartment annotations follow WLTCL. Their processed binary derivative
is redistributed here with authorization confirmed by the repository
maintainer. See `data/processed/README.md` for source citations and terms.

To regenerate the processed files from an authorized copy of the source data,
place its `Fence_truck` directory under `data/raw/` and run:

```bash
python scripts/prepare_dataset.py --overwrite
cp data/splits/*.txt data/processed/splits/
```

Source label `2` is mapped to `truck_bed`; labels `0`, `1`, and `3` are mapped
to `non_bed`.

## Multi-Seed Training

Run the paper protocol for seeds 0, 1, and 2:

```bash
python train_multiseed.py
```

The default configuration uses 80 epochs, 4096 sampled points, batch size 8,
AdamW with learning rate 0.001, cosine scheduling, boundary-loss weight 0.2,
and no data augmentation. To select a GPU explicitly:

```bash
python train_multiseed.py --cuda-visible-devices 0
```

The suite writes per-seed validation/test records plus aggregate JSON and CSV
summaries under `output/metrics/zpvehicles/`. Checkpoints are generated locally
under `log/` and are ignored by Git.

To inspect the complete command sequence without training:

```bash
python train_multiseed.py --dry-run
```

## Final Model Configuration

The authoritative configuration is `configs/zpvehicles/brace_net.yaml`, merged
recursively with `configs/default.yaml` and
`configs/zpvehicles/default.yaml`. The registered model is `BRACENet`, with a
`BRACEEncoder`, `HierarchicalPointDecoder`, main segmentation head, training-only
boundary head, and zero-initialized residual-logit head.

## License and Attribution

BRACE-Net additions are released under the MIT License. The hierarchical point
backbone and CUDA operators include modified third-party code; see
`THIRD_PARTY_NOTICES.md` and `licenses/third-party-backbone-MIT.txt`.
