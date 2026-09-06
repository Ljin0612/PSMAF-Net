"""Task-independent, multi-level PSMAF fusion wrapper."""

from collections.abc import Sequence

import torch
from torch import nn

from .multiscale_adaptive_fusion import MultiScaleAdaptiveFusion


class PSMAFFusion(nn.Module):
    """Fuse aligned pyramid levels for detection or segmentation."""

    def __init__(self, channels: Sequence[int], use_psg=True, use_msaf=True, fusion_mode="psmaf"):
        super().__init__()
        if fusion_mode not in {"psmaf", "add", "concat"}:
            raise ValueError("fusion_mode must be one of: psmaf, add, concat")
        self.channels = tuple(channels)
        self.use_msaf = use_msaf
        self.fusion_mode = fusion_mode
        self.levels = nn.ModuleList([MultiScaleAdaptiveFusion(c, use_psg) for c in channels])
        self.concat_projections = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in channels])

    def forward(self, rgb_features, ir_features):
        if len(rgb_features) != len(self.channels) or len(ir_features) != len(self.channels):
            raise ValueError(f"expected {len(self.channels)} RGB and IR feature levels")
        fused = []
        for rgb, ir, adaptive, projection in zip(rgb_features, ir_features, self.levels, self.concat_projections):
            if rgb.shape != ir.shape:
                raise ValueError("corresponding RGB and IR feature levels must match")
            if self.fusion_mode == "concat":
                value = projection(torch.cat((rgb, ir), dim=1))
            elif self.fusion_mode == "add" or not self.use_msaf:
                value = rgb + ir
            else:
                value = adaptive(rgb, ir)
            fused.append(value)
        return tuple(fused)
