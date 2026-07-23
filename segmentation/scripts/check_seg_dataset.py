"""Lightweight semantic segmentation dataset layout checker.

This script intentionally checks only path existence. Dataset-specific parsing
and MMSegmentation compatibility checks are TODOs for the future adapter layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that a segmentation dataset root, image directory, mask directory, and split file exist."
    )
    parser.add_argument(
        "--root", required=True, type=Path, help="Dataset root directory."
    )
    parser.add_argument(
        "--images",
        default="images",
        type=Path,
        help="Images directory, relative to root unless absolute.",
    )
    parser.add_argument(
        "--masks",
        default="masks",
        type=Path,
        help="Masks directory, relative to root unless absolute.",
    )
    parser.add_argument(
        "--split",
        required=True,
        type=Path,
        help="Split file path, relative to root unless absolute.",
    )
    return parser


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = build_parser().parse_args()
    root = args.root
    images = resolve_path(root, args.images)
    masks = resolve_path(root, args.masks)
    split = resolve_path(root, args.split)

    checks = [
        ("root", root, root.is_dir),
        ("images", images, images.is_dir),
        ("masks", masks, masks.is_dir),
        ("split", split, split.is_file),
    ]

    missing = []
    for label, path, predicate in checks:
        if predicate():
            print(f"[OK] {label}: {path}")
        else:
            print(f"[MISSING] {label}: {path}")
            missing.append(label)

    if missing:
        print("Dataset layout check failed. Missing: " + ", ".join(missing))
        return 1

    print("Dataset layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
