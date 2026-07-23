"""Pseudo-semantic guidance module placeholders for PSMAF-Net.

This module intentionally defines only a stable interface.  It must remain
independent from downstream detection and segmentation heads so both tasks can
reuse the same PSMAF feature outputs in later implementations.
"""

from torch import nn


class PseudoSemanticGuidance(nn.Module):
    """Interface placeholder for pseudo-semantic guided cross-modal alignment.

    Future versions will use pseudo-semantic labels/priors to guide regional
    alignment between RGB and infrared feature maps.  The current scaffold is an
    identity pass-through that preserves inputs unchanged.
    """

    def forward(self, rgb_feat, ir_feat, pseudo_semantic_prior=None):
        """Return RGB and infrared features unchanged.

        Args:
            rgb_feat: RGB modality features from an upstream backbone.
            ir_feat: Infrared modality features from an upstream backbone.
            pseudo_semantic_prior: Optional pseudo-semantic prior reserved for
                future cross-modal region alignment.

        Returns:
            dict: Unmodified features and optional prior under stable keys.
        """
        return {
            "rgb_feat": rgb_feat,
            "ir_feat": ir_feat,
            "pseudo_semantic_prior": pseudo_semantic_prior,
        }
