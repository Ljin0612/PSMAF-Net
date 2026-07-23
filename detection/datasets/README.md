# Detection datasets

This directory documents dataset expectations for object detection experiments.
Dataset files, annotations, checkpoints, and training outputs must not be
committed to this repository.

## M3FD-IR expected layout

The M3FD infrared detection entry point targets an M3FD-IR dataset root with a
simple, explicit layout:

```text
M3FD-IR/
├── images/
├── annotations/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

The lightweight checker only verifies that the requested dataset root, image
directory, annotation directory, split directory, and split file exist. It does
not validate annotation schema or image/annotation pairing yet.

```bash
python detection/scripts/check_m3fd_detection.py \
  --dataset-root /path/to/M3FD-IR \
  --split train
```

If a local M3FD copy uses different folder names, pass `--images-dir`,
`--annotations-dir`, or `--splits-dir` to the checker and future entry scripts.

## TODO

- Define the canonical annotation format expected by Detectron2 registration.
- Add conversion or registration helpers only after the local M3FD annotation
  convention is verified.
- Keep generated metadata, converted annotations, and dataset symlinks out of
  version control unless they are small documentation fixtures.
