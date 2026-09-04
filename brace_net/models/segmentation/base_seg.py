"""Final BRACE-Net segmentation model."""

import copy
import logging

import torch
import torch.nn as nn

from ..build import MODELS, build_model_from_cfg
from ..layers import TriPlaneProjectionModule, create_convblock1d


@MODELS.register_module()
class BRACENet(nn.Module):
    """BRACE-Net with coordinate priors, boundary supervision, and RLF."""

    def __init__(
        self,
        encoder_args,
        decoder_args,
        cls_args,
        boundary_args,
        triplane_grid_size=64,
        triplane_dropout=0.1,
        triplane_logit_alpha_init=0.01,
        return_aux_in_eval=False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logging.warning("Unused BRACE-Net options: %s", sorted(kwargs))

        self.encoder = build_model_from_cfg(encoder_args)

        decoder_cfg = copy.deepcopy(encoder_args)
        decoder_cfg.update(decoder_args)
        decoder_cfg.encoder_channel_list = self.encoder.channel_list
        self.decoder = build_model_from_cfg(decoder_cfg)
        feature_channels = self.decoder.out_channels

        main_head_cfg = copy.deepcopy(cls_args)
        main_head_cfg.in_channels = feature_channels
        self.head = build_model_from_cfg(main_head_cfg)

        self.triplane_projection = TriPlaneProjectionModule(
            feature_channels,
            grid_size=triplane_grid_size,
            dropout=triplane_dropout,
        )
        residual_head_cfg = copy.deepcopy(cls_args)
        residual_head_cfg.in_channels = feature_channels
        self.triplane_head = build_model_from_cfg(residual_head_cfg)
        self._zero_init_last_conv1d(self.triplane_head)
        self.triplane_logit_alpha = nn.Parameter(
            torch.tensor(float(triplane_logit_alpha_init), dtype=torch.float32)
        )

        boundary_head_cfg = copy.deepcopy(boundary_args)
        boundary_head_cfg.in_channels = feature_channels
        self.boundary_head = build_model_from_cfg(boundary_head_cfg)
        self.return_aux_in_eval = bool(return_aux_in_eval)

    @staticmethod
    def _zero_init_last_conv1d(module):
        last_conv = None
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv1d):
                last_conv = submodule
        if last_conv is None:
            raise ValueError("The residual logit head must contain a Conv1d layer")
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, data):
        points, features = self.encoder.forward_seg_feat(data)
        features = self.decoder(points, features).squeeze(-1)

        main_logits = self.head(features)
        triplane_features = self.triplane_projection(points[0], features)
        residual_logits = self.triplane_head(triplane_features)
        segmentation_logits = main_logits + self.triplane_logit_alpha * residual_logits

        if not self.training and not self.return_aux_in_eval:
            return segmentation_logits
        return {
            "seg_logits": segmentation_logits,
            "boundary_logits": self.boundary_head(features),
        }


@MODELS.register_module()
class SegHead(nn.Module):
    """Point-wise segmentation head."""

    def __init__(
        self,
        num_classes,
        in_channels,
        mlps=None,
        norm_args=None,
        act_args=None,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logging.warning("Unused segmentation-head options: %s", sorted(kwargs))
        norm_args = norm_args or {"norm": "bn1d"}
        act_args = act_args or {"act": "relu"}
        hidden_channels = [in_channels] if mlps is None else list(mlps)
        channels = [in_channels, *hidden_channels, num_classes]

        layers = []
        for index in range(len(channels) - 2):
            layers.append(
                create_convblock1d(
                    channels[index],
                    channels[index + 1],
                    norm_args=norm_args,
                    act_args=act_args,
                )
            )
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(create_convblock1d(channels[-2], channels[-1], act_args=None))
        self.head = nn.Sequential(*layers)

    def forward(self, features):
        return self.head(features)
