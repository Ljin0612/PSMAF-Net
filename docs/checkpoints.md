# Checkpoint management

## UNIV pretrained checkpoint

The UNIV pretrained checkpoint is required to initialize the UNIV encoder for
downstream detection and, once the segmentation integration is complete,
segmentation. It is an external artifact: download the official **UNIV
Checkpoint** linked from [`third_party/UNIV/README.md`](../third_party/UNIV/README.md)
and do not commit it to this repository. This repository does not include or
automatically download pretrained weights.

Store the downloaded checkpoint at the following conventional local path:

```text
checkpoints/univ/univ_pretrained.pth
```

The entire `checkpoints/` directory and common checkpoint extensions are
ignored by Git. You may use another location, but must pass that path explicitly
to the command-line entry point.

## Detection example

The M3FD detection runner requires the pretrained checkpoint through
`--checkpoint`:

```bash
python detection/scripts/run_m3fd_univ_fasterrcnn.py \
  --dataset-root /path/to/M3FD \
  --checkpoint checkpoints/univ/univ_pretrained.pth \
  --epochs 12 \
  --eval-every-epochs 1 \
  --device cuda \
  --work-dir outputs/detection/m3fd_univ_12ep
```

The runner checks that the checkpoint exists before starting and loads its
compatible encoder weights into the UNIV backbone.

## Segmentation example

The segmentation evaluation entry point accepts a checkpoint at the same local
path:

```bash
python segmentation/scripts/eval_upernet_univ.py \
  --config segmentation/configs/msrs_ir_upernet_univ.yaml \
  --data-root /path/to/MSRS \
  --checkpoint checkpoints/univ/univ_pretrained.pth \
  --split test \
  --output-dir outputs/segmentation/msrs_ir_upernet_univ
```

The segmentation entry point is currently an integration scaffold: it records
the intended UPerNet + UNIV invocation but does not yet construct the model,
load the checkpoint, or run evaluation. Keep the checkpoint external while that
integration is completed.

## Repository policy

Never add pretrained weights, fine-tuned weights, or generated checkpoints to
the repository. Keep them in ignored local directories such as `checkpoints/`
or `pretrained/`, and keep run artifacts in `outputs/`.
