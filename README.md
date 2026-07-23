# PSMAF-Net

**Pseudo-Semantic guided Multi-scale Adaptive Fusion Network**

中文名：**伪语义引导的多尺度自适应跨模态融合网络**

PSMAF-Net is a clean research repository for an RGB-IR upstream cross-modal fusion model built on top of the original UNIV foundation-model route. The fused representation will be evaluated through two downstream tasks:

- Object detection: Mask R-CNN / Detectron2 route, following the original UNIV detection setting.
- Semantic segmentation: UperNet / MMSegmentation route, following the original UNIV segmentation setting.

## Current scope

This repository currently provides only the initial formal project scaffold:

- Original UNIV entry point under `third_party/UNIV/`.
- Environment sanity checks.
- Detection and segmentation launcher placeholders.
- PSMAF-Net module placeholders.

It intentionally does **not** include a complete PSMAF-Net implementation, datasets, checkpoints, full training runs, experiment logs, or legacy YOLO-style adapter code.

## Repository layout

```text
configs/
  detection/       # Detectron2 / Mask R-CNN configuration placeholders
  segmentation/    # MMSegmentation / UperNet configuration placeholders
docs/              # Design notes and project roadmap
psmaf_net/         # PSMAF-Net Python package placeholders
scripts/           # Environment checks and task entry points
third_party/UNIV/  # Original UNIV code entry or integration notes
tests/             # Lightweight scaffold tests
```

## Quick checks

```bash
python scripts/check_env.py
python -m pytest tests
```

## Entry points

```bash
# Upstream UNIV pretraining entry check
python scripts/run_univ_pretrain.py --dry-run

# Downstream detection placeholder
python scripts/run_detection.py --dry-run

# Downstream segmentation placeholder
python scripts/run_segmentation.py --dry-run
```

## Data and artifact policy

Do not commit datasets, model weights, full training outputs, or generated experiment artifacts. The `.gitignore` blocks common heavy outputs such as `*.pth`, `*.pt`, `*.ckpt`, `runs/`, `results/`, `logs/`, and dataset folders.
