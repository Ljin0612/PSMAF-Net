# PSMAF-Net project plan

## Repository status

This repository is the formal PSMAF-Net model repository and is the mainline
location for current PSMAF-Net development. The legacy `Ljin0612/M3FD`
repository is treated only as an old exploratory codebase and is no longer
maintained as the main development line.

## Environment objective

PSMAF-Net keeps one lightweight dependency entry point around the bundled
original UNIV code, Detectron2 detection experiments, and MMSegmentation
segmentation experiments. The repository-level files are intentionally
conservative:

- `requirements.txt` records the original UNIV package versions that are known
  from `third_party/UNIV/requirements.txt` where such versions are available.
- `environment.yml` creates a minimal Conda environment and delegates package
  installation to `requirements.txt`.
- Detectron2, MMCV, and MMSegmentation are documented as downstream framework
  entry points because their installable builds depend on CUDA, PyTorch, and
  platform compatibility.
- `tools/check_env.py` checks the active environment only; it never installs,
  upgrades, or modifies dependencies.

## Dependency policy

1. Preserve original UNIV version notes first. The bundled UNIV README records a
   `python=3.6` Conda environment, and the bundled UNIV requirements record
   package pins such as `torch==2.4.1`, `torchvision==0.19.1`,
   `numpy==1.23.5`, `opencv-python==4.10.0.84`, `timm==1.0.12`, and
   `mmengine==0.10.5`.
2. Avoid broad unpinned upgrades. New versions should be introduced only when a
   specific downstream compatibility issue requires them.
3. Keep framework-specific installation decisions explicit. Detectron2 wheels
   and MMCV/MMSegmentation combinations should be selected for the exact CUDA
   and PyTorch runtime in use.
4. Treat environment checks as diagnostics. Missing optional dependencies should
   be printed as `missing` and should not stop the script.

## Requirements layering

- `third_party/UNIV/requirements.txt` is the complete official UNIV dependency
  snapshot imported with the upstream source.
- Top-level `requirements.txt` is the current lightweight PSMAF-Net entry point;
  it prioritizes core dependencies and is not necessarily identical to a full
  official UNIV pretraining environment.
- To run complete official UNIV pretraining, start from
  `third_party/UNIV/requirements.txt` and create a dedicated environment matched
  to the server CUDA and PyTorch versions.
- Do not blindly upgrade Detectron2, MMCV, or MMSegmentation; each must be
  matched to the target CUDA/PyTorch runtime.

## Initial milestones

### Milestone 1: dependency entry points

- Maintain top-level `requirements.txt` and `environment.yml` as the primary
  setup references.
- Use `python tools/check_env.py` after environment creation to report the
  active Python, PyTorch, CUDA, torchvision, timm, numpy, OpenCV, Detectron2,
  MMCV, MMEngine, and MMSegmentation status.

### Milestone 2: UNIV integration boundary

- Keep the original UNIV source under `third_party/UNIV/`.
- Prefer wrapping UNIV entry points from PSMAF-Net scripts instead of editing
  upstream files unless required for compatibility.
- Document any divergence from original UNIV versions before changing pins.

### Milestone 3: downstream task setup

- Detection route: connect the UNIV/PSMAF backbone export to Mask R-CNN through
  Detectron2 configs and training scripts.
- Segmentation route: connect the UNIV/PSMAF backbone export to UPerNet through
  MMSegmentation configs and training scripts.
- Validate each route independently before combining task-level results.
