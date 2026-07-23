"""Multi-scale adaptation module placeholders for PSMAF-Net.

This module is a core PSMAF component and intentionally has no dependency on
any detection-specific neck or segmentation-specific head.
"""

from torch import nn


class MultiScaleAdaptation(nn.Module):
    """Interface placeholder for shared multi-scale cross-modal adaptation.

    Future versions will adapt shared upstream features across multiple scales
    before downstream tasks consume them.  It is not a detection-specific neck;
    detection and segmentation should both use the same core PSMAF outputs.
    """

    def forward(self, features):
        """Return features unchanged until the research implementation lands."""
        return features
