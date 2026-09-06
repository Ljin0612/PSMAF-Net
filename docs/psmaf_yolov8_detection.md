# PSMAF-YOLOv8 detection

## Motivation and architecture

The compact **PSMAF-YOLO v1** remains useful as a small, auditable debugging
prototype, but its custom backbone and head do not make a controlled comparison
with YOLOv8-s. **PSMAF-YOLOv8** is the official-experiment branch. It mirrors the
YOLOv8-s Conv/C2f/SPPF depth and width, uses independent pretrained RGB and IR
backbones, fuses their stride 8/16/32 (P3/P4/P5) tensors with PSMAF, and sends the
result through the YOLOv8 PAN-FPN and decoupled DFL detection head.

The repository contains no historical runnable RGB-only, IR-only, early-fusion,
or late-fusion YOLOv8 scripts/configs; only the compact branch mentions that
comparison matrix. This branch therefore preserves the established compact
branch conventions: 640-pixel default input, native YOLO labels, class order,
M3FD split files, IoU-ranked AP50 and COCO-style mAP50:95, JSON/CSV metrics, and
`runs/detect/<experiment-name>` outputs. It does not alter any old results.

## Dataset

```text
M3FD_Detection/
  vi/                 # aligned visible images
  ir/                 # aligned infrared images
  labels/             # class x_center y_center width height (normalized)
  meta/{train,val,test}.txt
```

Classes are `people, car, bus, motorcycle, lamp, truck` (IDs 0--5). The config
is `detection/configs/m3fd_psmaf_yolov8.yaml`; neither labels nor splits are
rewritten.

## Pretrained weights

`--weights yolov8s.pt` imports matching official checkpoint tensors into both
backbones and the neck/head. The loader reports loaded, skipped, missing, and
unexpected keys. Downloads are deliberately disabled: provide an existing local
file (an absent path produces an actionable error). A PSMAF training checkpoint
is instead continued with `--resume auto` or `--resume /path/to/last.pt`.

## Commands

One-epoch sanity run:

```bash
python detection/scripts/train_psmaf_yolov8_m3fd.py --dataset-root /data/M3FD_Detection --weights /weights/yolov8s.pt --epochs 1 --batch 2 --workers 0 --name psmaf-yolov8-sanity
```

Twenty-image overfit run:

```bash
python detection/scripts/train_psmaf_yolov8_m3fd.py --dataset-root /data/M3FD_Detection --weights /weights/yolov8s.pt --epochs 50 --debug-num-images 20 --name psmaf-yolov8-overfit20
```

Full training and test evaluation:

```bash
python detection/scripts/train_psmaf_yolov8_m3fd.py --dataset-root /data/M3FD_Detection --weights /weights/yolov8s.pt --epochs 100 --batch 8 --imgsz 640 --device cuda:0 --name psmaf-yolov8s-m3fd
python detection/scripts/eval_psmaf_yolov8_m3fd.py --dataset-root /data/M3FD_Detection --weights runs/detect/psmaf-yolov8s-m3fd/best.pt --split test --device cuda:0 --name psmaf-yolov8s-m3fd-test
```

Ablations keep every other setting fixed:

```bash
python detection/scripts/train_psmaf_yolov8_m3fd.py ... --fusion-mode add --name psmaf-yolov8-add
python detection/scripts/train_psmaf_yolov8_m3fd.py ... --fusion-mode concat --name psmaf-yolov8-concat
python detection/scripts/train_psmaf_yolov8_m3fd.py ... --no-psg --name psmaf-yolov8-no-psg
python detection/scripts/train_psmaf_yolov8_m3fd.py ... --no-msaf --name psmaf-yolov8-no-msaf
```

Training writes `last.pt`, best-validation `best.pt`, optional `epochN.pt`,
`metrics.json/csv`, `train_log.csv/jsonl`, and `eval_diagnostics.json`.
