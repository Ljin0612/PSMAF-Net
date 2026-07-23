"""Shared multi-scale adaptation interface for PSMAF-Net.

This module is a core PSMAF placeholder for upstream shared multi-scale
cross-modal adaptation. It is deliberately not a detection-specific neck and is
not coupled to any segmentation head.
"""

from torch import nn


class MultiScaleAdaptation(nn.Module):
    """Placeholder for shared multi-scale cross-modal feature adaptation.

    Future implementations will adapt fused RGB-IR features across scales before
    they are consumed by downstream tasks. Detection and segmentation pipelines
    should call the same core PSMAF output features from this boundary.
    """

    def __init__(self):
        """Initialize the placeholder module without learnable layers."""
        super().__init__()

    def forward(self, features):
        """Return features unchanged until adaptation logic is implemented."""
        return features


# Backward-compatible alias for the earlier scaffold spelling.
MultiscaleAdaptation = MultiScaleAdaptation


__all__ = ["MultiScaleAdaptation", "MultiscaleAdaptation"]
