"""Top-level PSMAF-Net placeholder.

The full Pseudo-Semantic guided Multi-scale Adaptive Fusion Network is intentionally
not implemented in the repository scaffold. This placeholder documents the intended
integration boundary for future work.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PSMAFNetPlaceholder:
    """Configuration marker for the future PSMAF-Net model."""

    backbone: str = "UNIV"
    fusion_goal: str = "RGB-IR pseudo-semantic multi-scale adaptive fusion"

    def describe(self) -> str:
        """Return a human-readable summary of the planned model."""
        return f"PSMAF-Net placeholder using {self.backbone}: {self.fusion_goal}"
