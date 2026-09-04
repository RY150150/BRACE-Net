"""Tri-plane residual feature projection used by BRACE-Net."""

import torch
import torch.nn as nn


class _ViewCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        return self.net(features)


class TriPlaneProjectionModule(nn.Module):
    """Project point features to XY/XZ/YZ grids and gather them back."""

    def __init__(self, channels, grid_size=64, dropout=0.1):
        super().__init__()
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")

        self.view_cnns = nn.ModuleList([_ViewCNN(self.channels) for _ in range(3)])
        self.fusion = nn.Sequential(
            nn.Conv1d(self.channels * 3, self.channels, 1, bias=False),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if dropout and float(dropout) > 0 else nn.Identity(),
        )

    def _normalize_to_grid(self, xyz):
        xyz_min = xyz.amin(dim=1, keepdim=True)
        xyz_max = xyz.amax(dim=1, keepdim=True)
        xyz_range = (xyz_max - xyz_min).clamp_min(1e-6)
        normalized = (xyz - xyz_min) / xyz_range
        grid = torch.floor(normalized * (self.grid_size - 1)).long()
        return grid.clamp_(0, self.grid_size - 1)

    def _project_view(self, point_features, grid_uv, view_cnn):
        batch_size, channels, num_points = point_features.shape
        grid_size = self.grid_size
        flat_index = grid_uv[..., 0] * grid_size + grid_uv[..., 1]
        expanded_index = flat_index.unsqueeze(1).expand(-1, channels, -1)

        plane = point_features.new_zeros(batch_size, channels, grid_size * grid_size)
        plane.scatter_add_(2, expanded_index, point_features)
        counts = point_features.new_zeros(batch_size, 1, grid_size * grid_size)
        counts.scatter_add_(2, flat_index.unsqueeze(1), point_features.new_ones(batch_size, 1, num_points))
        plane = plane / counts.clamp_min(1.0)

        plane = view_cnn(plane.view(batch_size, channels, grid_size, grid_size))
        return plane.view(batch_size, channels, grid_size * grid_size).gather(2, expanded_index)

    def forward(self, xyz, point_features):
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError("xyz must have shape [B, N, 3]")
        if point_features.dim() != 3:
            raise ValueError("point_features must have shape [B, C, N]")
        if xyz.size(0) != point_features.size(0) or xyz.size(1) != point_features.size(2):
            raise ValueError("xyz and point_features must share batch size and point count")

        grid = self._normalize_to_grid(xyz)
        axes = ((0, 1), (0, 2), (1, 2))
        view_features = [
            self._project_view(point_features, grid[..., list(view_axes)], view_cnn)
            for view_axes, view_cnn in zip(axes, self.view_cnns)
        ]
        return point_features + self.fusion(torch.cat(view_features, dim=1))
