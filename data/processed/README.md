# Processed Truck-Bed Dataset

This directory contains the ready-to-use binary point-cloud segmentation data
used in the BRACE-Net paper:

- `samples/`: 106 compressed NumPy files;
- `splits/`: the fixed 74/16/16 train/validation/test partition;
- 10,352,125 annotated points in total;
- class `0`: non-bed; class `1`: truck-bed/compartment.

Each sample stores only `pos`, `x`, and `y`. The files contain no original
source paths, auxiliary metadata, model predictions, or weights. The default
configuration at `configs/zpvehicles/default.yaml` reads this directory
directly.

## Provenance

The source point clouds are from:

Z. Ye, Z. Wang, X. Chen, T. Zhou, C. Yu, J. Guo, and J. Li, "ZPVehicles: A
Dataset of Large Vehicle 3D Point Cloud Data," IEEE MetroAutomotive, 2023,
pp. 234-239. https://doi.org/10.1109/MetroAutomotive57488.2023.10219120

The task-specific truck-compartment annotations follow:

G. Sun, M. Li, D. Liu, M. Liu, B. Wu, and Y. Zhang, "WLTCL: Wide-Field-of-View
3-D LiDAR Truck Compartment Automatic Localization System," IEEE Transactions
on Instrumentation and Measurement, vol. 74, pp. 1-14, 2025.
https://doi.org/10.1109/TIM.2025.3564017

The processed derivative is redistributed with authorization confirmed by the
repository maintainer. The repository's MIT software license applies to the
code, not automatically to the underlying point clouds or annotations. Users
remain responsible for complying with the applicable source-data and
annotation terms and for citing both works above.
