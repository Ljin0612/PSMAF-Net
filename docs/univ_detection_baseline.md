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
unexpected, and shape-mismatched keys. PEFT `base_layer` tensors are mapped back
to their original module names; users should supply a merged checkpoint when
LoRA adapter deltas must also be retained. Loading fails when the file does not
contain any compatible encoder weight.

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
Add `--amp` to enable Detectron2 mixed-precision training, and add
`--gradient-checkpointing` to recompute the high-level UNIV `blocks3` Transformer
activations during backward. These options can be combined. The runner prints
CUDA allocated, reserved, and peak-allocated memory after model construction,
the first forward, and the first backward.

An RTX 2080 Ti with 11 GB of VRAM may run out of memory at `--input-size 1024`
when fully fine-tuning the encoder, even with batch size 1. Start validation with
a `--input-size 640` smoke test (preferably with `--amp` and
`--gradient-checkpointing`). A frozen baseline retains no encoder autograd graph,
so `--freeze-backbone` is the mode to try first at 1024.

The runner validates the train/test dataset layouts, creates bbox-only COCO JSON,
registers both datasets, loads the UNIV checkpoint during model construction, and
converts the requested epoch count into Detectron2 iterations.

## Debugging low AP

For a quick memorization test, restrict training to a deterministic prefix of the
training set and evaluate that same subset after the final optimizer step:

```bash
python detection/scripts/run_m3fd_univ_fasterrcnn.py \
  --dataset-root /path/to/M3FD \
  --checkpoint /path/to/univ_checkpoint.pth \
  --debug-num-images 16 \
  --eval-train \
  --save-visualizations 8 \
  --epochs 20 \
  --work-dir outputs/detection/m3fd_univ_overfit
```

`--debug-num-images N` changes only the registered training dataset and epoch
schedule; the generated COCO annotations and test split remain unchanged.
`--eval-train` writes `raw_train_eval_results.json` and
`train_bbox_metrics.json`. `--save-visualizations N` runs final inference on the
first N test images and writes side-by-side panels under
`WORK_DIR/debug_visualizations/`. Ground-truth boxes and labels are green;
predicted boxes, class labels, and confidence scores are red.

## Ten-iteration smoke test

Use the same real M3FD data and UNIV checkpoint as the full baseline:

```bash
python detection/scripts/run_m3fd_univ_fasterrcnn.py \
  --dataset-root /path/to/M3FD \
  --checkpoint /path/to/univ_checkpoint.pth \
  --smoke-test \
  --input-size 640 \
  --amp \
  --gradient-checkpointing \
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
