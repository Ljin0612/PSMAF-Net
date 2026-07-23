# PSMAF design

PSMAF-Net is the formal model line in this repository. Its core goal is to build
an upstream multimodal fusion model on top of the reproduced UNIV baseline, then
validate that upstream representation on downstream detection and segmentation
tasks.

## Core ideas

1. Pseudo-semantic labels guide cross-modal region alignment so that infrared
   and visible features are encouraged to match at semantically meaningful
   regions instead of only at raw pixel locations.
2. Multi-scale adaptive fusion strengthens the representation of objects,
   boundaries, and background context at different scales.
3. The PSMAF module is an upstream core component located under `core/psmaf`.
4. Detection and segmentation are downstream validation tasks used to evaluate
   the upstream PSMAF-Net representation.

## Task boundary

The model design should keep upstream fusion independent from downstream task
heads. Detection and segmentation may require task-specific configs, datasets,
and training scripts, but those downstream routes should consume the shared
UNIV/PSMAF feature export contract rather than redefining the core PSMAF module.
