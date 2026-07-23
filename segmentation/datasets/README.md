# Segmentation datasets

This directory documents dataset expectations for the semantic segmentation
entry points. Do **not** commit datasets, annotations, generated masks, training
outputs, or model weights.

## Expected layout

The lightweight checker only validates that the configured locations exist. A
recommended layout is:

```text
<dataset-root>/
  images/
    ... image files ...
  masks/
    ... semantic segmentation masks ...
  splits/
    train.txt
    val.txt      # when available
    test.txt
```

The split files should list sample identifiers or relative image paths in the
format expected by the future MMSegmentation dataset adapter. The exact parsing
contract is intentionally left as a TODO until the adapter is implemented.

## Recorded datasets

### MSRS-IR

- Training images: 1083
- Testing images: 361
- Intended task: infrared semantic segmentation
- Config: `../configs/msrs_ir_upernet_univ.yaml`

### MFNet-IR

- Training images: 1569
- Validation images: 392
- Testing images: 393
- Intended task: infrared semantic segmentation
- Config: `../configs/mfnet_ir_upernet_univ.yaml`

## TODO

- Define canonical split-file line format.
- Add dataset-specific metadata validation once the adapters are implemented.
- Add conversion notes for MMSegmentation-compatible annotation structures.
