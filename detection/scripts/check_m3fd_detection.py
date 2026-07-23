#!/usr/bin/env python3
"""Lightweight preflight checks for an M3FD-IR detection dataset root."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that the expected M3FD-IR detection dataset root, image "
            "directory, annotation directory, split directory, and split file exist."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Path to the local M3FD-IR dataset root.",
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Image directory name relative to --dataset-root.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="annotations",
        help="Annotation directory name relative to --dataset-root.",
    )
    parser.add_argument(
        "--splits-dir",
        default="splits",
        help="Split directory name relative to --dataset-root.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name to check. The checker expects <split>.txt inside --splits-dir.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str, should_be_dir: bool = True) -> list[str]:
    if not path.exists():
        return [f"Missing {description}: {path}"]
    if should_be_dir and not path.is_dir():
        return [f"Expected {description} to be a directory: {path}"]
    if not should_be_dir and not path.is_file():
        return [f"Expected {description} to be a file: {path}"]
    return []


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    images_dir = dataset_root / args.images_dir
    annotations_dir = dataset_root / args.annotations_dir
    splits_dir = dataset_root / args.splits_dir
    split_file = splits_dir / f"{args.split}.txt"

    errors: list[str] = []
    errors.extend(require_path(dataset_root, "dataset root"))
    errors.extend(require_path(images_dir, "images directory"))
    errors.extend(require_path(annotations_dir, "annotations directory"))
    errors.extend(require_path(splits_dir, "splits directory"))
    errors.extend(require_path(split_file, "split file", should_be_dir=False))

    if errors:
        print("M3FD-IR detection dataset check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("M3FD-IR detection dataset check passed:")
    print(f"  dataset_root: {dataset_root}")
    print(f"  images_dir: {images_dir}")
    print(f"  annotations_dir: {annotations_dir}")
    print(f"  split_file: {split_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
