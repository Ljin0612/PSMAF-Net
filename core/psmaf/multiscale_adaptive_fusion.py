"""Adaptive RGB-infrared fusion at one feature-pyramid level."""

import torch
from torch import nn

from .pseudo_semantic_guidance import PseudoSemanticGuidance


class MultiScaleAdaptiveFusion(nn.Module):
    """Fuse one equally-shaped RGB/IR feature pair without resizing it."""

    def __init__(self, channels: int, use_psg: bool = True):
        super().__init__()
        self.use_psg = use_psg
        self.guidance = PseudoSemanticGuidance(channels)
        self.residual = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor):
        if rgb_feat.shape != ir_feat.shape:
            raise ValueError("RGB and IR features must have identical shapes")
        if self.use_psg:
            attention, weights = self.guidance(rgb_feat, ir_feat)
            alpha = weights[:, 0:1] * attention
            beta = weights[:, 1:2] * attention
        else:
            attention = rgb_feat.new_ones((rgb_feat.shape[0], 1, *rgb_feat.shape[-2:]))
            weights = rgb_feat.new_full((rgb_feat.shape[0], 2, 1, 1), 0.5)
            alpha, beta = weights[:, 0:1], weights[:, 1:2]
        residual = self.residual(torch.cat((rgb_feat, ir_feat), dim=1))
        return alpha * rgb_feat + beta * ir_feat + (1.0 - attention) * residual

