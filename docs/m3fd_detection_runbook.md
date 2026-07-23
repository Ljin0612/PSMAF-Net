# M3FD-IR Detectron2 Mask R-CNN runbook

## Goal

The detection path is staged deliberately:

1. Run an M3FD-IR Detectron2 standard Mask R-CNN smoke test.
2. Integrate the original UNIV backbone with Mask R-CNN.
3. Integrate PSMAF-Net RGB-IR pseudo-semantic guided detection.

The current scripts only establish dataset checks, COCO conversion, Detectron2 import checks, runtime config metadata, dry-runs, and a minimal standard Mask R-CNN smoke-run entry point. They do **not** claim that the UNIV backbone or PSMAF fusion has already been connected to Detectron2.

## Dataset layout A: M3FD RGB-T

Expected root:

```text
M3FD/
├── ir/
├── vi/
├── labels/
└── meta/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

The smoke-test pipeline uses infrared images by default and reads image IDs from `meta/<split>.txt`.

## Category table

| ID | Class |
|---:|---|
| 0 | people |
| 1 | car |
| 2 | bus |
| 3 | motorcycle |
| 4 | lamp |
| 5 | truck |

## Checks

Check local Detectron2-related imports without training:

```bash
python detection/scripts/check_detectron2_imports.py
```

Check M3FD RGB-T layout:

```bash
python detection/scripts/check_m3fd_detection.py \
  --dataset-root /path/to/M3FD \
  --layout m3fd-rgbt \
  --split train
```

Check an already prepared M3FD-IR Detectron2-style layout:

```bash
python detection/scripts/check_m3fd_detection.py \
  --dataset-root /path/to/M3FD-IR \
  --layout m3fd-ir-detectron2 \
  --split train
```

## COCO conversion

Generate COCO-style annotations for the infrared training split. Write generated JSON under `outputs/detection/coco/` or another ignored work directory; do not commit generated annotations.

```bash
python detection/scripts/make_m3fd_ir_coco.py \
  --dataset-root /path/to/M3FD \
  --split train \
  --layout m3fd-rgbt \
  --modality ir \
  --output-json outputs/detection/coco/m3fd_ir_train.json
```

Repeat for test or validation splits as needed:

```bash
python detection/scripts/make_m3fd_ir_coco.py \
  --dataset-root /path/to/M3FD \
  --split test \
  --layout m3fd-rgbt \
  --modality ir \
  --output-json outputs/detection/coco/m3fd_ir_test.json
```

## Runtime config metadata

```bash
python detection/scripts/make_m3fd_maskrcnn_config.py \
  --train-json outputs/detection/coco/m3fd_ir_train.json \
  --train-image-root /path/to/M3FD/ir \
  --test-json outputs/detection/coco/m3fd_ir_test.json \
  --test-image-root /path/to/M3FD/ir \
  --output-config outputs/detection/m3fd_ir_maskrcnn_smoke_config.json \
  --output-dir outputs/detection/maskrcnn_smoke \
  --max-iter 100 \
  --ims-per-batch 1 \
  --input-size 1024
```

## Dry-run

Dry-run prints the planned steps and does not generate files or start training:

```bash
python detection/scripts/run_m3fd_maskrcnn_smoke.py \
  --dataset-root /path/to/M3FD \
  --work-dir outputs/detection/maskrcnn_smoke \
  --dry-run
```

## Smoke-run

Non-dry-run checks the dataset, converts train/test COCO JSON, registers Detectron2 datasets, and runs a minimal standard Mask R-CNN R50-FPN smoke training job when Detectron2 is installed:

```bash
python detection/scripts/run_m3fd_maskrcnn_smoke.py \
  --dataset-root /path/to/M3FD \
  --work-dir outputs/detection/maskrcnn_smoke \
  --max-iter 10 \
  --ims-per-batch 1 \
  --input-size 1024
```

## UNIV backbone integration TODO

- Add a Detectron2 backbone adapter for the original UNIV model.
- Load validated UNIV checkpoints without changing detection dataset semantics.
- Compare standard Mask R-CNN R50-FPN smoke metrics against the UNIV-backbone Mask R-CNN setting.

## RGB-IR / pseudo-semantic guided detection TODO

- Extend dataset registration to paired RGB-IR samples instead of IR-only image records.
- Add PSMAF pseudo-semantic guidance modules in a Detectron2-compatible data/model path.
- Report subset analysis for day/night, visible degraded, small object, and crowded-scene examples.
