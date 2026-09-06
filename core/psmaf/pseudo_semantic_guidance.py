"""Label-free pseudo-semantic guidance for paired feature maps."""

import torch
from torch import nn


class PseudoSemanticGuidance(nn.Module):
    """Estimate foreground attention and global modality reliability.

    The estimates are learned only from the two feature tensors; no annotation is
    an input to this module.
    """

    def __init__(self, channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or max(16, channels // 2)
        self.shared = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.attention = nn.Conv2d(hidden, 1, 1)
        self.reliability = nn.Conv2d(hidden, 2, 1)

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor):
        if rgb_feat.shape != ir_feat.shape:
            raise ValueError(f"paired features must have equal shapes, got {rgb_feat.shape} and {ir_feat.shape}")
        latent = self.shared(torch.cat((rgb_feat, ir_feat), dim=1))
        attention = self.attention(latent).sigmoid()
        reliability = self.reliability(latent).mean((2, 3), keepdim=True).softmax(dim=1)
        return attention, reliability
