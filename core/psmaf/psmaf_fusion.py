"""PSMAF fusion interface placeholder.

The fusion module composes pseudo-semantic guidance and shared multi-scale
adaptation as a task-agnostic core feature interface. It intentionally has no
dependency on detection heads or segmentation heads.
"""

from torch import nn

from .multiscale_adaptation import MultiScaleAdaptation
from .pseudo_semantic_guidance import PseudoSemanticGuidance


class PSMAFFusion(nn.Module):
    """Task-agnostic PSMAF fusion boundary for future RGB-IR feature modules.

    Downstream detection and segmentation code should consume the same output
    features exposed by this core module. The current placeholder preserves the
    input/output behavior while establishing the composition points for future
    pseudo-semantic guidance and multi-scale adaptation implementations.
    """

    def __init__(self):
        """Create placeholder submodules for the future PSMAF pipeline."""
        super().__init__()
        self.pseudo_semantic_guidance = PseudoSemanticGuidance()
        self.multiscale_adaptation = MultiScaleAdaptation()

    def forward(self, rgb_feat, ir_feat, pseudo_semantic_prior=None):
        """Return RGB and infrared features unchanged.

        Args:
            rgb_feat: RGB/visible feature tensor or feature collection.
            ir_feat: Infrared feature tensor or feature collection.
            pseudo_semantic_prior: Optional pseudo semantic prior reserved for
                future pseudo-semantic-guided alignment.

        Returns:
            The unchanged ``rgb_feat`` and ``ir_feat`` pair.
        """
        guided_features = self.pseudo_semantic_guidance(
            rgb_feat,
            ir_feat,
            pseudo_semantic_prior=pseudo_semantic_prior,
        )
        return self.multiscale_adaptation(guided_features)


__all__ = ["PSMAFFusion"]
