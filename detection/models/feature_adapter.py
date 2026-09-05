"""Convert UNIV encoder stages into a Detectron2-compatible feature pyramid."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


FEATURE_NAMES = ("p2", "p3", "p4", "p5")
FEATURE_STRIDES = (4, 8, 16, 32)


class UNIVFeatureAdapter(nn.Module):
    """Project the three native ConvMAE stages and derive a stride-32 level.

    The official UNIV ConvMAE encoder exposes stride 4, 8, and 16 stages.  This
    adapter leaves that source untouched, projects the stages to a common channel
    width, and derives ``p5`` from ``p4`` for Detectron2's shared RPN head.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 384, 768),
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("UNIVFeatureAdapter requires exactly three encoder stages")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        self.out_channels = int(out_channels)
        self.projections = nn.ModuleList(
            nn.Conv2d(int(channels), self.out_channels, kernel_size=1)
            for channels in in_channels
        )
        self.p5 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, stages: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(stages) != 3:
            raise ValueError(f"expected three UNIV stages, received {len(stages)}")
        projected = [layer(feature) for layer, feature in zip(self.projections, stages)]
        features = dict(zip(FEATURE_NAMES[:3], projected))
        features["p5"] = self.p5(features["p4"])
        return features
