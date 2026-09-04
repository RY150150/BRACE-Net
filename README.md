# BRACE-Net

Official multi-seed training code for **BRACE-Net: Lightweight
False-Positive-Sensitive LiDAR Point-Cloud Segmentation for Truck-Bed
Perception**.

BRACE-Net combines three components around a lightweight hierarchical point
encoder:

- GeoLite coordinate-prior injection at encoder stages 0-4;
- training-only boundary supervision with 16-nearest-neighbor targets;
- a 64x64 XY/XZ/YZ Tri-Plane residual-logit branch.

This repository contains the final model configuration used by the paper. It
does not include pretrained weights, checkpoints, point clouds, restricted
annotations, paper drafts, web applications, or alternative ablation models.

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

## Data Preparation

The code expects task annotations in the ZPVehicles fence-truck layout, with
one `labeled_3D.txt` file per sample and rows formatted as `x y z label`.
ZPVehicles is public, while the task-specific annotations used by the paper
require permission from their authors and are not redistributed here.

Place authorized source files under `data/raw/Fence_truck/`, then run:

```bash
python scripts/prepare_dataset.py
mkdir -p data/processed/splits
cp data/splits/*.txt data/processed/splits/
```

The committed split lists reproduce the paper's vehicle-instance-level
74/16/16 train/validation/test partition. Source label `2` is mapped to
`truck_bed`; labels `0`, `1`, and `3` are mapped to `non_bed`.

Each generated `.npz` file contains:

```text
pos: float32 [N, 3]
x:   float32 [N, 3]
y:   integer [N]
```

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
