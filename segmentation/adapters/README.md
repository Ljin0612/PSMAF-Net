# Segmentation adapters

Adapters will bridge this repository's UNIV semantic segmentation entry configs
to MMSegmentation.

## Intended boundaries

Future adapter code should handle:

- Translating `segmentation/configs/*.yaml` metadata into MMSegmentation config
  dictionaries or config files.
- Registering MSRS-IR and MFNet-IR dataset classes or dataset aliases.
- Mapping infrared image paths and semantic mask paths to MMSegmentation data
  pipelines.
- Wiring the UNIV backbone/checkpoint into a UPerNet segmentation model.
- Preserving the recorded paper-facing settings: UPerNet, 512 x 512 input,
  AdamW, 0.05 weight decay, Poly LR schedule, and `ignore_index=255`.

## Current status

No adapter is implemented yet. The train/eval scripts are deliberately limited
to argument parsing and explanatory messages so that the repository does not
contain unverified fake training logic.
