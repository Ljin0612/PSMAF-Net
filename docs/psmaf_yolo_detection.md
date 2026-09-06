# PSMAF-YOLO detection

## Motivation and relationship to UNIV

PSMAF-Net is the **Pseudo-Semantic Guided Multi-scale Adaptive Fusion
Framework** for RGB-infrared perception. UNIV learns a unified cross-modal
space during pre-training with attention-guided patch anchors and pseudo
patch-level semantic labels. Inspired by that principle, PSMAF moves label-free
pseudo-semantic guidance into downstream P3/P4/P5 fusion: it learns where
object-related signal is likely to occur and how reliable each modality is.
Ground-truth labels are never used to create the guidance map.

UNIV-Faster R-CNN remains preserved as a direct-adaptation comparison baseline.
PSMAF-YOLO is the new main detection branch. A future SegFormer, DeepLab, or
UperNet segmentation branch should reuse the task-independent `core/psmaf`
modules rather than introducing a second fusion implementation.

## Structure

Two YOLO-style backbones independently produce RGB and IR P3/P4/P5 pyramids.
At each matching level, pseudo-semantic guidance predicts spatial attention and
two softmax reliability weights. Adaptive fusion combines the inputs and a
learned residual without changing feature shape. The fused pyramid then enters
the conventional top-down YOLO neck and anchor-free head.

## M3FD layout

```text
M3FD_Detection/
├── vi/                 # visible/RGB images
├── ir/                 # aligned infrared images
├── labels/             # class x_center y_center width height (normalized)
└── meta/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Classes retain the existing order: people, car, bus, motorcycle, lamp, truck.
The loader reads split and label files without modifying them. Override the
portable config's example `path` using `--dataset-root`.

## Train and evaluate

```bash
python detection/scripts/train_psmaf_yolo_m3fd.py \
  --dataset-root /data/M3FD_Detection --epochs 100 --batch 8 \
  --imgsz 640 --device cuda:0 --project runs/detect --name psmaf-yolo

python detection/scripts/eval_psmaf_yolo_m3fd.py \
  --dataset-root /data/M3FD_Detection --weights runs/detect/psmaf-yolo/last.pt \
  --split test --device cuda:0 --project runs/detect --name psmaf-yolo-test
```

Training and evaluation both write `metrics.json`/`metrics.csv`. PSMAF-YOLO
uses a real multi-head detection evaluator: P3, P4, and P5 predictions are
decoded into image coordinates, merged, confidence-filtered, and processed by
class-aware NMS. Precision and recall use class-and-IoU matching; AP50, mAP50,
mAP50:95, and per-class AP are computed from ranked precision-recall curves.
In particular, mAP is not a `precision * recall` proxy. Run artifacts and
weights are git-ignored.

## Debugging zero metrics

A one-epoch end-to-end sanity run (including train-split evaluation) is:

```bash
python detection/scripts/train_psmaf_yolo_m3fd.py --dataset-root /data/M3FD_Detection \
  --epochs 1 --eval-train --name sanity-1epoch
```

To deliberately overfit only the first 20 samples of both train and validation
splits, use `--debug-num-images 20`. This mode is intended solely for debugging
the input, loss, and evaluator pipeline and **must not be used for official
reporting**:

```bash
python detection/scripts/train_psmaf_yolo_m3fd.py --dataset-root /data/M3FD_Detection \
  --epochs 50 --debug-num-images 20 --eval-train --name overfit-20
```

If no prediction passes the default 0.25 confidence threshold, repeat with (for
example) `--conf-thres 0.01`; `--nms-iou 0.45` can also be adjusted to diagnose
suppression. Each run's `train_log.csv` and `train_log.jsonl` show total,
objectness, box, and classification loss alongside learning rate and validation
metrics for every epoch. `eval_diagnostics.json` reports decoded, confidence-
filtered, and post-NMS box counts, GT count, IoU-0.50 TP/FP/FN, and per-class
counts. With `--eval-train`, the latest split results are also written to
`train_metrics.json` and `val_metrics.json`.

Resume the latest run checkpoint with `--resume` or `--resume auto`, or select
an exact checkpoint with `--resume /path/to/checkpoint.pt`. Resume restores the
model, optimizer, AMP scaler, and next epoch; `--weights` only initializes model
weights for a new run. If both are supplied, `--resume` takes precedence with a
warning.

## Ablations

```bash
# Equal-weight addition (no adaptive fusion)
python detection/scripts/train_psmaf_yolo_m3fd.py --dataset-root /data/M3FD_Detection --fusion-mode add
# Concatenation plus 1x1 projection
python detection/scripts/train_psmaf_yolo_m3fd.py --dataset-root /data/M3FD_Detection --fusion-mode concat
# Remove pseudo-semantic guidance or the whole adaptive unit
python detection/scripts/train_psmaf_yolo_m3fd.py --dataset-root /data/M3FD_Detection --no-psg
python detection/scripts/train_psmaf_yolo_m3fd.py --dataset-root /data/M3FD_Detection --no-msaf
```

Expected comparison table: RGB-only YOLO, IR-only YOLO, RGB-IR early fusion,
RGB-IR late fusion, and PSMAF-YOLO, alongside the preserved UNIV-Faster R-CNN
adaptation baseline.
