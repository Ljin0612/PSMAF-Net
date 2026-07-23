#!/usr/bin/env python3
"""Entry point for future M3FD-IR Mask R-CNN + UNIV evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from m3fd_maskrcnn_scaffold import load_experiment


DEFAULT_CONFIG = Path("detection/configs/m3fd_ir_maskrcnn_univ.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the M3FD-IR Mask R-CNN + UNIV detection evaluation entry point. "
            "This script intentionally does not launch unverified Detectron2 evaluation yet."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path, help="Path to experiment metadata YAML.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Path to the local M3FD-IR dataset root.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint path for future Detectron2 evaluation.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for future evaluation outputs.")
    parser.add_argument("--split", default="test", help="Evaluation split name for future Detectron2 registration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = args.config.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not config.is_file():
        print(f"Config file does not exist: {config}")
        return 1
    if not dataset_root.is_dir():
        print(f"Dataset root does not exist or is not a directory: {dataset_root}")
        return 1
    if not checkpoint.is_file():
        print(f"Checkpoint file does not exist: {checkpoint}")
        return 1

    experiment, _ = load_experiment(config)

    print("M3FD-IR Mask R-CNN + UNIV evaluation entry point")
    print(f"  config: {config}")
    print(f"  dataset_root: {dataset_root}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  output_dir: {output_dir}")
    print(f"  split: {args.split}")
    print(f"  expected_test_images: {experiment.test_images}")
    print(f"  head: {experiment.head}")
    print(f"  framework: {experiment.framework}")
    print("TODO: integrate Detectron2 dataset registration, evaluator construction,")
    print("      checkpoint loading, metric computation, and result serialization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
