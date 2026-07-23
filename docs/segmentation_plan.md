# Segmentation plan

The segmentation baseline target is to align with the original UNIV downstream
route: a UNIV backbone feeding a UPerNet head through MMSegmentation.

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
