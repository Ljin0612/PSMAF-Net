# Detection adapters

Detection adapters will bridge repository dataset conventions to the original
UNIV downstream detection path: **Mask R-CNN / Detectron2**.

## Main path

- Primary framework: Detectron2.
- Primary head: Mask R-CNN.
- Target dataset entry: M3FD-IR, infrared modality.
- Target input resolution: 1024 x 1024.

YOLO-style adapters are intentionally not the main line for this repository.
They should not be treated as the default detection implementation.

## TODO

- Add a Detectron2 dataset registration module for M3FD-IR.
- Add a verified mapping from M3FD annotations to Detectron2 dataset dicts.
- Add Mask R-CNN config construction that consumes UNIV backbone/checkpoint
  components.
- Add evaluation adapters after the dataset and training wiring are validated.
