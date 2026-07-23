# Detection plan

The detection route is split into three explicit steps so that dataset handling, the baseline detector, the UNIV backbone, and PSMAF-Net fusion can be validated independently.

## Step 1: M3FD-IR standard Mask R-CNN smoke test

- Validate the local M3FD RGB-T dataset layout (`ir/`, `vi/`, `labels/`, `meta/`).
- Convert M3FD infrared YOLO labels to COCO-style annotations for Detectron2.
- Register the converted COCO datasets with Detectron2.
- Run a small standard Mask R-CNN R50-FPN smoke test to verify the detection stack.
- This step is infrastructure only; it does not claim UNIV or PSMAF integration.

## Step 2: UNIV-original backbone + Mask R-CNN

- Reproduce or load the original UNIV backbone in a Detectron2-compatible wrapper.
- Replace the standard R50-FPN backbone only after the Step 1 smoke test is stable.
- Keep dataset registration and COCO conversion identical to Step 1 for controlled comparison.

## Step 3: PSMAF-Net RGB-IR pseudo-semantic guided detection

- Extend the pipeline from IR-only samples to paired RGB-IR inputs.
- Add PSMAF-Net pseudo-semantic guidance and fusion modules in the Detectron2 model path.
- Compare the PSMAF variant against both the standard Mask R-CNN baseline and the UNIV-original backbone baseline.

## M3FD evaluation role

M3FD is used for complex-environment detection evaluation. After the main smoke and backbone integrations are stable, the evaluation plan should include subset analysis for day/night scenes, degraded visible imagery, small objects, crowded scenes, and other challenging conditions.

## Non-mainline choices

YOLO-style adapters are not the mainline route for this repository. Historical or exploratory YOLO-style adapter experiments are not used as the formal baseline for the PSMAF-Net detection task.
