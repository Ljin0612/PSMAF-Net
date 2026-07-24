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

## Complex-scenario detection validation

M3FD detection is one of the core downstream validations for PSMAF-Net. The
recommended detection route is to export UNIV-original and PSMAF-Net fused
features or images with the same interface, then train and evaluate a consistent
Mask R-CNN detection head on each input setting.

The expected advantage of pseudo-semantic label guidance should be validated
especially on low-light, nighttime, occluded, small-object, and visible-light
degraded scenes. In these complex scenarios, pseudo-semantic supervision is
expected to help the model focus on semantically related regions, reduce
background interference introduced by simple RGB/IR concatenation, and use
pseudo-semantic priors to guide cross-modal region alignment between infrared
and visible features.

Recommended M3FD detection comparisons:

1. IR-only Mask R-CNN.
2. RGB-only Mask R-CNN.
3. RGB-IR early fusion + Mask R-CNN.
4. RGB-IR late fusion + Mask R-CNN.
5. UNIV-original + Mask R-CNN.
6. PSMAF-Net + Mask R-CNN.

Besides overall detection metrics, the evaluation should include subset analysis
for low-light/nighttime, occlusion, small-object, and visible-light degradation
cases to verify whether PSMAF-Net improves robustness in the intended complex
conditions.

## Complex-scenario segmentation validation

The segmentation validation should follow the same principle: keep the upstream
UNIV/PSMAF representation fixed behind the shared feature export contract, then
compare downstream segmentation heads under matched training and evaluation
settings.

The recommended sequence is:

1. First reproduce the MSRS-IR UNIV-original + UPerNet segmentation baseline.
2. Then replace the upstream representation with PSMAF-Net under the same
   UPerNet evaluation protocol.
3. Finally extend the study to more complex RGB-IR semantic segmentation
   datasets, such as FMB or MFNet, when their pixel-level semantic masks and
   train/validation splits are available.

Segmentation evaluation should report mIoU, mAcc, per-class IoU, and
complex-scenario subset analysis. The subset analysis should focus on the same
failure modes as detection, including low-light/nighttime scenes, occlusion,
small objects, and visible-light degradation, to test whether pseudo-semantic
label guidance improves semantic region focus, suppresses RGB/IR concatenation
background noise, and strengthens pseudo-semantic-prior-based cross-modal
alignment.

Dataset suitability must be checked before using any dataset for standard
semantic segmentation. If a dataset does not provide pixel-level masks, it must
not be used as a standard semantic segmentation evaluation set; it can only be
used for detection experiments or for auxiliary pseudo-label analysis.

## Task boundary

The model design should keep upstream fusion independent from downstream task
heads. Detection and segmentation may require task-specific configs, datasets,
and training scripts, but those downstream routes should consume the shared
UNIV/PSMAF feature export contract rather than redefining the core PSMAF module.
