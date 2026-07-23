"""PSMAF fusion interface placeholders.

The fusion module composes core PSMAF building blocks without depending on any
specific detection head or segmentation head.
"""

from torch import nn

from .multiscale_adaptation import MultiScaleAdaptation
from .pseudo_semantic_guidance import PseudoSemanticGuidance


class PSMAFFusion(nn.Module):
    """Placeholder interface for PSMAF feature fusion.

    The class wires pseudo-semantic guidance and multi-scale adaptation behind a
    shared core interface.  The current implementation preserves input/output
    behavior so future detection and segmentation pipelines can call the same
    ``core/psmaf`` feature interface without task-specific coupling.
    """

    def __init__(self):
        """Create placeholder PSMAF submodules."""
        super().__init__()
        self.pseudo_semantic_guidance = PseudoSemanticGuidance()
        self.multiscale_adaptation = MultiScaleAdaptation()

    def forward(self, rgb_feat, ir_feat, pseudo_semantic_prior=None):
        """Return fused-feature placeholders without changing inputs.

        Args:
            rgb_feat: RGB modality features from an upstream backbone.
            ir_feat: Infrared modality features from an upstream backbone.
            pseudo_semantic_prior: Optional pseudo-semantic prior reserved for
                future guided alignment.

        Returns:
            dict: Unmodified features from the placeholder PSMAF interface.
        """
        guided_features = self.pseudo_semantic_guidance(
            rgb_feat=rgb_feat,
            ir_feat=ir_feat,
            pseudo_semantic_prior=pseudo_semantic_prior,
        )
        return self.multiscale_adaptation(guided_features)
