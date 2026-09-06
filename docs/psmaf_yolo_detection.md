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

Training writes `metrics.json`/`metrics.csv`; evaluation writes
`bbox_metrics.json`/`bbox_metrics.csv`. Both include precision, recall, mAP50,
mAP50:95, and per-class AP fields. Run artifacts and weights are git-ignored.

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
