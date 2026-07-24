# M3FD-IR Detectron2 bbox detection runbook

## Goal

The current M3FD smoke pipeline is **bbox detection-only**. M3FD labels used by this pipeline provide detection boxes, not instance masks, so the smoke baseline deliberately avoids mask loss.

The detection path is staged deliberately:

1. Run an M3FD-IR Detectron2 standard Faster R-CNN R50-FPN bbox detection smoke test.
2. Integrate the original UNIV backbone with a Detectron2 detector after the bbox smoke baseline is stable.
3. Integrate PSMAF-Net RGB-IR pseudo-semantic guided detection.

The current scripts establish dataset checks, bbox-only COCO conversion, Detectron2 import checks, runtime config metadata, dry-runs, a minimal standard Faster R-CNN smoke-run entry point, and bbox AP evaluation on the configured test split. They do **not** claim that the UNIV backbone, PSMAF fusion, or a full paper-facing Mask R-CNN route has already been connected to Detectron2.

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

The generated COCO category IDs are COCO-standard 1-based IDs. Source YOLO label IDs remain 0-based in M3FD label text files, and the converter maps them to the COCO IDs below.

| COCO ID | Class |
|---:|---|
| 1 | people |
| 2 | car |
| 3 | bus |
| 4 | motorcycle |
| 5 | lamp |
| 6 | truck |

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

The generated COCO JSON is **bbox-only** and uses COCO-standard 1-based category IDs: annotations include `bbox`, `area`, `category_id`, `image_id`, and `iscrowd`, but do not include `segmentation` polygons or masks. Do not train mask loss on this JSON unless valid instance masks or documented segmentation annotations are explicitly provided.

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

The smoke config uses Detectron2 `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` and records `mask_on: false` / `annotation_type: bbox_only`.

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

Non-dry-run checks the dataset, converts train/test bbox-only COCO JSON, registers Detectron2 datasets, disables `MODEL.MASK_ON`, runs a minimal standard Faster R-CNN R50-FPN bbox detection smoke training job, and then evaluates bbox AP on the test split with Detectron2's COCO evaluator when Detectron2 is installed:

```bash
python detection/scripts/run_m3fd_maskrcnn_smoke.py \
  --dataset-root /path/to/M3FD \
  --work-dir outputs/detection/maskrcnn_smoke \
  --max-iter 10 \
  --ims-per-batch 1 \
  --input-size 1024
```

Evaluation artifacts are written under the smoke work directory in `eval/`, for example `outputs/detection/maskrcnn_smoke/eval/`. This is still a Faster R-CNN R50-FPN bbox-only baseline; it is not a UNIV detector and it is not PSMAF-Net.

Do **not** use `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` directly with the bbox-only M3FD JSON. If a future experiment uses that instance-segmentation config, it must either provide legal `segmentation` fields or explicitly set `MODEL.MASK_ON = False`.

## UNIV backbone integration TODO

- Add a Detectron2 backbone adapter for the original UNIV model after the bbox-only Faster R-CNN smoke baseline is stable.
- Load validated UNIV checkpoints without changing detection dataset semantics.
- Compare the standard Faster R-CNN R50-FPN bbox smoke metrics against the UNIV-backbone detector setting.
- Treat UNIV-original + Mask R-CNN / Detectron2 integration as the next stage, not the current smoke baseline.

## RGB-IR / pseudo-semantic guided detection TODO

- Extend dataset registration to paired RGB-IR samples instead of IR-only image records.
- Add PSMAF pseudo-semantic guidance modules in a Detectron2-compatible data/model path.
- Report subset analysis for day/night, visible degraded, small object, and crowded-scene examples.
