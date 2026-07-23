#!/usr/bin/env python3
"""Generate runtime metadata for an M3FD-IR Mask R-CNN smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Mask R-CNN smoke-test config metadata.")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--train-image-root", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--test-image-root", required=True)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--ims-per-batch", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=8e-5)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--weights", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "purpose": "M3FD-IR Detectron2 Mask R-CNN smoke test",
        "note": "Standard Mask R-CNN R50-FPN baseline; UNIV backbone is not integrated in this config.",
        "model_zoo_config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "train_json": args.train_json,
        "train_image_root": args.train_image_root,
        "test_json": args.test_json,
        "test_image_root": args.test_image_root,
        "output_dir": args.output_dir,
        "num_classes": args.num_classes,
        "max_iter": args.max_iter,
        "ims_per_batch": args.ims_per_batch,
        "base_lr": args.base_lr,
        "input_size": args.input_size,
        "weights": args.weights,
        "train_dataset_name": "m3fd_ir_train",
        "test_dataset_name": "m3fd_ir_test",
    }


def main() -> int:
    args = parse_args()
    config = build_config(args)
    output_config = args.output_config.expanduser().resolve()
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"wrote config metadata: {output_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
