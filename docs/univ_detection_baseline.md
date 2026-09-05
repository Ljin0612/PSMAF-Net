# UNIV-original M3FD Detectron2 detection baseline

## Scope

This baseline integrates the original UNIV ConvMAE encoder with Detectron2 for
M3FD-IR bounding-box detection. It deliberately does **not** implement PSMAF,
pseudo-semantic guidance, RGB-IR fusion, or semantic segmentation. The vendored
files under `third_party/UNIV/` remain unchanged.

M3FD annotations used here contain bounding boxes rather than instance masks, so
the detector uses Faster R-CNN semantics and keeps `MODEL.MASK_ON` disabled.

## Architecture

```text
M3FD infrared image
        |
        v
official UNIV ConvViT encoder
        |
        +-- stage 1, stride 4  -- 1x1 projection --> p2
        +-- stage 2, stride 8  -- 1x1 projection --> p3
        +-- stage 3, stride 16 -- 1x1 projection --> p4
                                                   |
                                                   +-- stride-2 conv --> p5
        |
        v
Detectron2 RPN + Faster R-CNN ROI heads
```

The `UNIVBackbone` feature contract is:

```python
{
    "p2": Tensor[B, 256, H / 4, W / 4],
    "p3": Tensor[B, 256, H / 8, W / 8],
    "p4": Tensor[B, 256, H / 16, W / 16],
    "p5": Tensor[B, 256, H / 32, W / 32],
}
```

The official encoder has three native stages with channels `[256, 384, 768]`.
`detection/models/feature_adapter.py` projects those stages to the common width
required by Detectron2's shared RPN head and derives the fourth pyramid level.
The adapter constructs the official 224-pixel pretraining grid so pretrained
absolute position embeddings load without a shape mismatch, then interpolates
that grid to the padded detector feature resolution during each forward pass.
Infrared inputs use the official UNIV IR normalization (`mean=0.5338`,
`std=0.2519`), expressed on Detectron2's 0-255 pixel scale.

## Checkpoint loading

`UNIVBackbone.load_pretrained()` accepts common checkpoint containers named
`model`, `state_dict`, `student`, `teacher`, or `backbone`. It removes repeated
`module.`, `student.`, `backbone.`, `encoder.`, and PEFT `base_model.model.`
prefixes, loads shape-compatible encoder tensors, and reports loaded, missing,
unexpected, and shape-mismatched keys. For the PEFT checkpoints produced by the
vendored UNIV training path, `base_layer` tensors are mapped back to their native
module names and every complete LoRA A/B pair is merged as
`base + (B @ A) * alpha / rank`. The default alpha is 32, matching the bundled
UNIV configuration. An incomplete LoRA pair is rejected rather than silently
discarded.

Loading also fails when the checkpoint does not contain the first patch-embedding
weight or does not cover at least 80% of encoder parameters by element count.
Element-weighted coverage prevents a collection of small bias tensors from making
an incompatible checkpoint appear sufficiently complete.

The checkpoint initializes only the official encoder. The detection feature
adapter, RPN, and ROI heads are trained for the M3FD task.

## Frozen and fine-tuned modes

Fine-tuning is the default. Add `--freeze-backbone` to disable gradients for the
official encoder while leaving the feature adapter and detection heads trainable.
The frozen encoder remains in evaluation mode even when Detectron2 switches the
whole detector to training mode.

## Training

```bash
python detection/scripts/run_m3fd_univ_fasterrcnn.py \
  --dataset-root /path/to/M3FD \
  --checkpoint /path/to/univ_checkpoint.pth \
  --epochs 12 \
  --eval-every-epochs 1 \
  --device cuda \
  --work-dir outputs/detection/m3fd_univ_12ep
```

Add `--freeze-backbone` for frozen-encoder training. Omit it to fine-tune UNIV.

The runner validates the train/test dataset layouts, creates bbox-only COCO JSON,
registers both datasets, loads the UNIV checkpoint during model construction, and
converts the requested epoch count into Detectron2 iterations.

## Ten-iteration smoke test

Use the same real M3FD data and UNIV checkpoint as the full baseline:

```bash
python detection/scripts/run_m3fd_univ_fasterrcnn.py \
  --dataset-root /path/to/M3FD \
  --checkpoint /path/to/univ_checkpoint.pth \
  --smoke-test \
  --eval-every-epochs 0 \
  --device cuda \
  --work-dir outputs/detection/m3fd_univ_smoke
```

`--smoke-test` forces exactly ten optimizer iterations. Successful completion
therefore verifies checkpoint loading, UNIV forward propagation, Faster R-CNN
loss construction, backward propagation, optimizer updates, and final COCO bbox
evaluation. The normal `--max-iter 10` form is also supported.

## Metrics

The runner uses `COCOEvaluator(tasks=("bbox",))` and writes
`bbox_metrics.json` containing:

- AP;
- AP50;
- AP75;
- per-class AP for people, car, bus, motorcycle, lamp, and truck.

Detectron2 training artifacts and evaluator outputs are stored under the selected
work directory. Generated annotations, checkpoints, logs, and results must not be
committed.

## Compilation and unit checks

```bash
python -m py_compile \
  detection/models/feature_adapter.py \
  detection/models/univ_backbone.py \
  detection/scripts/run_m3fd_univ_fasterrcnn.py

PYTHONPATH=. pytest -q
```
