"""Pseudo-semantic guidance interface for PSMAF-Net.

This module intentionally contains only a lightweight placeholder. It defines the
future integration point for pseudo-semantic-label-guided cross-modal region
alignment while keeping the current scaffold independent from downstream task
heads.
"""

from torch import nn


class PseudoSemanticGuidance(nn.Module):
    """Placeholder for pseudo-semantic-label-guided RGB-IR alignment.

    Future implementations will use pseudo semantic priors to guide cross-modal
    region alignment between RGB and infrared features. The module must remain a
    core fusion component and must not depend on detection or segmentation heads.
    """

    def __init__(self):
        """Initialize the placeholder module without learnable layers."""
        super().__init__()

    def forward(self, rgb_feat, ir_feat, pseudo_semantic_prior=None):
        """Return RGB and infrared features unchanged.

        Args:
            rgb_feat: RGB/visible feature tensor or feature collection.
            ir_feat: Infrared feature tensor or feature collection.
            pseudo_semantic_prior: Optional pseudo semantic prior reserved for
                future cross-modal region alignment.

        Returns:
            The unchanged ``rgb_feat`` and ``ir_feat`` pair.
        """
        return rgb_feat, ir_feat


__all__ = ["PseudoSemanticGuidance"]
