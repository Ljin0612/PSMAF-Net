# Detection

This directory defines the object detection downstream-task entry point based on the
original UNIV setting.

## Scope

The original UNIV paper evaluates object detection with **Mask R-CNN / Detectron2**.
This repository therefore treats Mask R-CNN + Detectron2 as the main detection path.
YOLO-style adapters are not the primary direction for this repository.

The current implementation intentionally provides a clear, auditable entry point
instead of unverified training code. The training and evaluation scripts expose
stable command-line interfaces and TODO markers; Detectron2 integration can be
added in a later change once the exact dataset registration and model wiring are
validated.

## Reference configuration

The baseline experiment metadata is stored in
[`configs/m3fd_ir_maskrcnn_univ.yaml`](configs/m3fd_ir_maskrcnn_univ.yaml):

- Dataset: M3FD-IR
- Modality: Infrared
- Head: Mask R-CNN
- Input resolution: 1024 x 1024
- Fine-tuning steps: 85k
- Optimizer: AdamW
- Base learning rate: 8e-5
- Weight decay: 0.1
- Batch size: 4

## Directory layout

```text
detection/
├── adapters/      # Notes for future Detectron2 / dataset adapter code.
├── configs/       # Experiment metadata and future Detectron2 configs.
├── datasets/      # Dataset layout documentation.
└── scripts/       # Dataset checks plus train/eval entry points.
```

## Dataset preflight check

Use the lightweight checker before attempting training or evaluation:

```bash
python detection/scripts/check_m3fd_detection.py \
  --dataset-root /path/to/M3FD-IR \
  --split train
```

By default the checker expects:

```text
/path/to/M3FD-IR/
├── images/
├── annotations/
└── splits/
    └── train.txt
```

The folder names can be overridden with command-line arguments when a local copy
uses a different convention.

## Training and evaluation entry points

The current train/eval scripts are placeholders by design. They validate their
CLI arguments, print the intended experiment metadata, and stop before launching
training or evaluation.

```bash
python detection/scripts/train_m3fd_maskrcnn_univ.py \
  --config detection/configs/m3fd_ir_maskrcnn_univ.yaml \
  --dataset-root /path/to/M3FD-IR \
  --output-dir outputs/detection/m3fd_ir_maskrcnn_univ

python detection/scripts/eval_m3fd_maskrcnn_univ.py \
  --config detection/configs/m3fd_ir_maskrcnn_univ.yaml \
  --dataset-root /path/to/M3FD-IR \
  --checkpoint /path/to/model.pth \
  --output-dir outputs/detection/m3fd_ir_maskrcnn_univ_eval
```

TODO: wire these scripts to Detectron2 dataset registration, Mask R-CNN config
construction, UNIV checkpoint loading, training, and evaluation once those pieces
are verified in this repository.
