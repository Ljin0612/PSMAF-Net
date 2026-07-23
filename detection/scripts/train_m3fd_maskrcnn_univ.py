#!/usr/bin/env python3
"""Entry point for future M3FD-IR Mask R-CNN + UNIV fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CONFIG = Path("detection/configs/m3fd_ir_maskrcnn_univ.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the M3FD-IR Mask R-CNN + UNIV detection training entry point. "
            "This script intentionally does not launch unverified training logic yet."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path, help="Path to experiment metadata YAML.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Path to the local M3FD-IR dataset root.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for future training outputs.")
    parser.add_argument("--split", default="train", help="Training split name for future Detectron2 registration.")
    parser.add_argument("--univ-checkpoint", type=Path, help="Optional UNIV checkpoint path for future backbone loading.")
    parser.add_argument("--resume", action="store_true", help="Reserve flag for future Detectron2 resume support.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = args.config.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not config.is_file():
        print(f"Config file does not exist: {config}")
        return 1
    if not dataset_root.is_dir():
        print(f"Dataset root does not exist or is not a directory: {dataset_root}")
        return 1

    print("M3FD-IR Mask R-CNN + UNIV training entry point")
    print(f"  config: {config}")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_dir: {output_dir}")
    print(f"  split: {args.split}")
    print(f"  univ_checkpoint: {args.univ_checkpoint}")
    print(f"  resume: {args.resume}")
    print("TODO: integrate Detectron2 dataset registration, Mask R-CNN config construction,")
    print("      UNIV checkpoint loading, optimizer setup, and the verified training loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
