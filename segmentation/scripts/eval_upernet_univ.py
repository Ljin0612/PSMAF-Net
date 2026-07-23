"""Semantic segmentation evaluation entry point for UPerNet + UNIV.

The original UNIV downstream semantic segmentation setting uses UPerNet through
MMSegmentation. This script currently provides only a clear CLI boundary and
TODOs; it intentionally does not implement unverified fake evaluation logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TODO = """
TODO: connect this entry point to MMSegmentation.
Planned steps:
  1. Load and validate the YAML metadata config.
  2. Convert the metadata into an MMSegmentation UPerNet config.
  3. Register the target infrared segmentation dataset adapter and split.
  4. For evaluation, read the selected split directories from config
     dataset.<split>.img_dir and dataset.<split>.ann_dir instead of using
     legacy split files.
  5. Load the UNIV/UPerNet checkpoint.
  6. Delegate evaluation and metric reporting to the MMSegmentation runner.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a future UPerNet + UNIV semantic segmentation evaluation run."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to segmentation YAML config."
    )
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Dataset root directory."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Checkpoint path for future evaluation.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split name for future directory-based evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for future predictions and metrics.",
    )
    parser.add_argument(
        "--launcher",
        default="none",
        choices=("none", "pytorch", "slurm", "mpi"),
        help="Future MMSegmentation launcher.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("UPerNet + UNIV semantic segmentation evaluation entry point")
    print(f"config: {args.config}")
    print(f"data_root: {args.data_root}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"output_dir: {args.output_dir}")
    print(f"launcher: {args.launcher}")
    print()
    print(TODO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
