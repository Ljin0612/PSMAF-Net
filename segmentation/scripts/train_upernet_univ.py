"""Semantic segmentation training entry point for UPerNet + UNIV.

The original UNIV downstream semantic segmentation setting uses UPerNet through
MMSegmentation. This script currently provides only a clear CLI boundary and
TODOs; it intentionally does not implement unverified fake training logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TODO = """
TODO: connect this entry point to MMSegmentation.
Planned steps:
  1. Load and validate the YAML metadata config.
  2. Convert the metadata into an MMSegmentation UPerNet config.
  3. Register the target infrared segmentation dataset adapter.
  4. Wire the UNIV backbone/checkpoint into UPerNet.
  5. Delegate training to the MMSegmentation runner.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a future UPerNet + UNIV semantic segmentation training run."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to segmentation YAML config."
    )
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Dataset root directory."
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Directory for future training outputs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from once training is implemented.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed to pass to the future runner."
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
    print("UPerNet + UNIV semantic segmentation training entry point")
    print(f"config: {args.config}")
    print(f"data_root: {args.data_root}")
    print(f"work_dir: {args.work_dir}")
    print(f"resume_from: {args.resume_from}")
    print(f"seed: {args.seed}")
    print(f"launcher: {args.launcher}")
    print()
    print(TODO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
