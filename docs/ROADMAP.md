# PSMAF-Net roadmap

## Phase 1: clean scaffold

- Keep the repository independent from legacy `Ljin0612/M3FD` experiment code.
- Provide an original UNIV integration entry under `third_party/UNIV/`.
- Provide lightweight environment checks and dry-run launchers.
- Reserve module boundaries for PSMAF-Net without implementing the full model.

## Phase 2: upstream fusion model

- Integrate the official UNIV backbone and checkpoints supplied by the project owner.
- Add pseudo-semantic guidance and multi-scale adaptive fusion modules.
- Define feature export contracts for downstream tasks.

## Phase 3: downstream validation

- Detection: Mask R-CNN / Detectron2.
- Segmentation: UperNet / MMSegmentation.
