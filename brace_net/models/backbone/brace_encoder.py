"""Coordinate-prior encoder used by BRACE-Net."""
from typing import Iterable

import torch
import torch.nn as nn

from ..build import MODELS
from ..layers import create_convblock1d
from .hierarchical_point_backbone import HierarchicalPointEncoder


class GeometryPriorEncoder(nn.Module):
    """Encode explicit per-point geometry priors with a tiny 1D MLP.

    The priors are computed from the current stage xyz coordinates only, so the
    module does not introduce KNN, ball query, or attention overhead.
    """

    def __init__(
        self,
        out_channels: int,
        num_priors: int = 5,
        hidden_dim: int = 16,
        norm_args=None,
        act_args=None,
        conv_args=None,
        eps: float = 1.0e-6,
    ):
        super().__init__()
        assert num_priors > 0, "geo_prior_dim must be positive"
        self.num_priors = int(num_priors)
        self.eps = float(eps)
        conv_kwargs = dict(conv_args) if conv_args is not None else {}
        hidden_dim = int(hidden_dim)
        out_channels = int(out_channels)

        self.mlp = nn.Sequential(
            create_convblock1d(
                self.num_priors,
                hidden_dim,
                norm_args=norm_args,
                act_args=act_args,
                **conv_kwargs,
            ),
            create_convblock1d(
                hidden_dim,
                out_channels,
                norm_args=norm_args,
                act_args=act_args,
                **conv_kwargs,
            ),
        )

    @staticmethod
    def _to_bnc_xyz(xyz: torch.Tensor) -> torch.Tensor:
        if xyz.dim() != 3:
            raise AssertionError(f"xyz must be a 3D tensor, got shape {tuple(xyz.shape)}")
        if xyz.shape[-1] == 3:
            return xyz
        if xyz.shape[1] == 3:
            return xyz.transpose(1, 2).contiguous()
        raise AssertionError(f"xyz must have a coordinate dimension of 3, got {tuple(xyz.shape)}")

    def _compute_priors(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz = self._to_bnc_xyz(xyz)
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        xyz_min = xyz.amin(dim=1)
        xyz_max = xyz.amax(dim=1)
        xyz_range = xyz_max - xyz_min

        x_norm = 2.0 * (x - xyz_min[:, 0:1]) / (xyz_range[:, 0:1] + self.eps) - 1.0
        y_norm = 2.0 * (y - xyz_min[:, 1:2]) / (xyz_range[:, 1:2] + self.eps) - 1.0
        z_norm = (z - xyz_min[:, 2:3]) / (xyz_range[:, 2:3] + self.eps)

        center_xy = xyz[..., :2].mean(dim=1, keepdim=True)
        rho = torch.sqrt(torch.sum((xyz[..., :2] - center_xy) ** 2, dim=-1).clamp_min(0.0))
        rho_norm = rho / (rho.amax(dim=1, keepdim=True) + self.eps)

        height_centered = z - z.mean(dim=1, keepdim=True)
        priors = [x_norm, y_norm, z_norm, rho_norm, height_centered]

        if self.num_priors <= len(priors):
            priors = priors[: self.num_priors]
        else:
            pad = priors[-1].new_zeros(priors[-1].shape)
            priors = priors + [pad] * (self.num_priors - len(priors))
        return torch.stack(priors, dim=1).contiguous()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._compute_priors(xyz))


class GeometryGuidedFusion(nn.Module):
    """Fuse point features with encoded geometry priors.

    Default fusion is a lightweight gated residual:
        out = point_feat + alpha * sigmoid(dwconv(geo_feat)) * geo_feat
    """

    def __init__(
        self,
        channels: int,
        geo_prior_dim: int = 5,
        geo_hidden_dim: int = 16,
        geo_fusion: str = "gated_add",
        norm_args=None,
        act_args=None,
        conv_args=None,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.geo_fusion = geo_fusion
        self.geo_encoder = GeometryPriorEncoder(
            out_channels=self.channels,
            num_priors=geo_prior_dim,
            hidden_dim=geo_hidden_dim,
            norm_args=norm_args,
            act_args=act_args,
            conv_args=conv_args,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        if geo_fusion == "gated_add":
            self.gate = nn.Sequential(
                nn.Conv1d(self.channels, self.channels, kernel_size=1, groups=self.channels, bias=True),
                nn.Sigmoid(),
            )
        elif geo_fusion == "add":
            self.gate = None
        else:
            raise ValueError(f"Unsupported geo_fusion: {geo_fusion}")

    def _to_bcn_feature(self, feat: torch.Tensor):
        if feat.dim() != 3:
            raise AssertionError(f"point feature must be a 3D tensor, got {tuple(feat.shape)}")
        if feat.shape[1] == self.channels:
            return feat, False
        if feat.shape[-1] == self.channels:
            return feat.transpose(1, 2).contiguous(), True
        raise AssertionError(
            f"point feature channel mismatch: expected {self.channels}, got shape {tuple(feat.shape)}"
        )

    def forward(self, xyz: torch.Tensor, point_feat: torch.Tensor) -> torch.Tensor:
        feat, transposed = self._to_bcn_feature(point_feat)
        geo_feat = self.geo_encoder(xyz)
        if geo_feat.shape[-1] != feat.shape[-1]:
            raise AssertionError(
                f"geometry and point features must share N, got {geo_feat.shape[-1]} and {feat.shape[-1]}"
            )

        if self.geo_fusion == "gated_add":
            feat = feat + self.alpha * self.gate(geo_feat) * geo_feat
        else:
            feat = feat + self.alpha * geo_feat

        if transposed:
            feat = feat.transpose(1, 2).contiguous()
        return feat


@MODELS.register_module()
class BRACEEncoder(HierarchicalPointEncoder):
    """Hierarchical point encoder with lightweight coordinate priors."""

    def __init__(
        self,
        *args,
        geo_prior: bool = True,
        geo_prior_dim: int = 5,
        geo_hidden_dim: int = 16,
        geo_fusion: str = "gated_add",
        geo_stages: Iterable[int] = (0, 1, 2, 3, 4),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.geo_prior = bool(geo_prior)
        stage_iter = range(len(self.channel_list)) if geo_stages is None else geo_stages
        self.geo_stages = tuple(int(stage) for stage in stage_iter)

        if not self.geo_prior:
            self.geo_fusions = nn.ModuleDict()
            return

        invalid_stages = [stage for stage in self.geo_stages if stage < 0 or stage >= len(self.channel_list)]
        if invalid_stages:
            raise ValueError(
                f"geo_stages {invalid_stages} are invalid for {len(self.channel_list)} encoder stages"
            )

        self.geo_fusions = nn.ModuleDict(
            {
                str(stage): GeometryGuidedFusion(
                    channels=self.channel_list[stage],
                    geo_prior_dim=geo_prior_dim,
                    geo_hidden_dim=geo_hidden_dim,
                    geo_fusion=geo_fusion,
                    norm_args=self.norm_args,
                    act_args=self.act_args,
                    conv_args=self.conv_args,
                )
                for stage in self.geo_stages
            }
        )

    def _fuse_stage(self, stage: int, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        key = str(stage)
        if key not in self.geo_fusions:
            return feat
        return self.geo_fusions[key](xyz, feat)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, "keys"):
            p0, f0 = p0["pos"], p0.get("x", None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
            f0 = self._fuse_stage(i, p0, f0)
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, "keys"):
            p0, f0 = p0["pos"], p0.get("x", None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            _f = self._fuse_stage(i, _p, _f)
            p.append(_p)
            f.append(_f)
        return p, f
