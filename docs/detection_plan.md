# Detection plan

The detection validation route for PSMAF-Net uses a UNIV-based upstream feature
extractor together with Mask R-CNN implemented through Detectron2.

## Mainline route

- Import and reproduce the original UNIV backbone first.
- Add the PSMAF upstream fusion module on top of the reproduced UNIV baseline.
- Export the fused upstream features to a Detectron2 Mask R-CNN detection head.
- Keep detection as a downstream validation task for the upstream PSMAF-Net
  design rather than as a separate detector-first architecture.

## Non-mainline choices

YOLO-style adapters are not the mainline route for this repository. Historical
or exploratory YOLO-style adapter experiments are not used as the formal
baseline for the PSMAF-Net detection task.
