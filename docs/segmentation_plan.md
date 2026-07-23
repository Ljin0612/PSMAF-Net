# Segmentation plan

The segmentation validation route for PSMAF-Net uses a UNIV-based upstream
feature extractor with a UPerNet head implemented through MMSegmentation.

## Mainline route

- Import and reproduce the original UNIV backbone first.
- Add the PSMAF upstream fusion module on top of the reproduced UNIV baseline.
- Export the fused upstream features to UPerNet through MMSegmentation configs
  and training scripts.
- Keep segmentation as a downstream validation task for the upstream PSMAF-Net
  design rather than as a separate segmentation-first architecture.

## Current stage

The repository currently establishes only the segmentation config, lightweight
dataset checker, and command-line entry points. Full MMSegmentation training,
model construction, dataset registration, and checkpoint loading are deferred to
a later integration step.

## MSRS-IR layout

MSRS-IR follows the bundled original UNIV/MMSegmentation directory layout:

```text
images/training
annotations/training
images/validation
annotations/validation
```

The MSRS segmentation config no longer uses the incorrect generic layout with
`images`, `masks`, or `splits/test.txt`. In the original UNIV setup, the
`images/validation` and `annotations/validation` directories are used for both
validation and test.
