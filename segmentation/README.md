# Segmentation

This directory defines the semantic segmentation downstream-task entry point for
UNIV-style infrared feature learning.

The original UNIV semantic segmentation downstream setting uses **UPerNet** on
top of the UNIV backbone and is expected to be integrated through
**MMSegmentation**. This repository intentionally keeps the current layer as a
clear, auditable entry point rather than pretending that a complete training
pipeline is already wired.

## Scope

Included now:

- Dataset-layout documentation for infrared semantic segmentation datasets.
- Lightweight dataset layout validation.
- YAML configuration stubs recording the paper-facing segmentation settings.
- Train/eval command-line entry points with explicit TODOs for future
  MMSegmentation integration.

Not included yet:

- A runnable MMSegmentation config conversion layer.
- UPerNet model construction code.
- UNIV checkpoint loading for segmentation.
- Training or evaluation loops.

## Paper-facing settings recorded here

- Head: UPerNet
- Input resolution: 512 x 512
- Optimizer: AdamW
- Weight decay: 0.05
- Learning rate schedule: Poly
- `ignore_index`: 255

## Files

- `configs/msrs_ir_upernet_univ.yaml`: MSRS infrared segmentation entry config.
- `configs/mfnet_ir_upernet_univ.yaml`: MFNet infrared segmentation entry config.
- `datasets/README.md`: expected dataset layout and split notes.
- `adapters/README.md`: future adapter boundaries for MMSegmentation.
- `scripts/check_seg_dataset.py`: lightweight dataset layout checker.
- `scripts/train_upernet_univ.py`: future training entry point.
- `scripts/eval_upernet_univ.py`: future evaluation entry point.

## Example commands

Check a prepared dataset layout:

```bash
python segmentation/scripts/check_seg_dataset.py \
  --root /path/to/MSRS-IR \
  --images images \
  --masks masks \
  --split splits/train.txt
```

Inspect the future training entry point without launching training:

```bash
python segmentation/scripts/train_upernet_univ.py \
  --config segmentation/configs/msrs_ir_upernet_univ.yaml \
  --data-root /path/to/MSRS-IR \
  --work-dir outputs/segmentation/msrs_ir_upernet_univ
```

Inspect the future evaluation entry point:

```bash
python segmentation/scripts/eval_upernet_univ.py \
  --config segmentation/configs/msrs_ir_upernet_univ.yaml \
  --data-root /path/to/MSRS-IR \
  --checkpoint /path/to/checkpoint.pth \
  --split splits/test.txt
```
