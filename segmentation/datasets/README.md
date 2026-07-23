# Segmentation datasets

This directory documents dataset expectations for the semantic segmentation
entry points. Do **not** commit datasets, annotations, generated masks, training
outputs, or model weights.

## Expected layouts

The lightweight checker validates configured dataset locations and image /
annotation stem matching. It supports the MSRS directory schema used by the
bundled original UNIV MMSegmentation config and a legacy split-file schema for
other datasets that still require it.

### MSRS-IR

MSRS-IR uses the original UNIV/MMSegmentation directory layout:

```text
MSRS/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

The MSRS config does not use `masks/`, `splits/train.txt`, or
`splits/test.txt`. The `validation` directories are configured for both val and
test to match the bundled UNIV base config.

### Legacy split-file datasets

Other datasets may continue to use dataset-specific split files until their
MMSegmentation adapters are finalized, for example:

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
format expected by the future MMSegmentation dataset adapter. If a dataset's
true layout is uncertain, keep a TODO rather than inventing paths.

## Recorded datasets

### MSRS-IR

- Training images: 1083
- Validation/test images: 361
- Intended task: infrared semantic segmentation
- Config: `../configs/msrs_ir_upernet_univ.yaml`

### MFNet-IR

- Training images: 1569
- Validation images: 392
- Testing images: 393
- Intended task: infrared semantic segmentation
- Config: `../configs/mfnet_ir_upernet_univ.yaml`
- TODO: verify the final MFNet MMSegmentation layout before replacing its
  legacy split-file metadata.

## TODO

- Define canonical split-file line format for legacy datasets.
- Add dataset-specific metadata validation once the adapters are implemented.
- Add conversion notes for MMSegmentation-compatible annotation structures.
