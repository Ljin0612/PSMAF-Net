#!/usr/bin/env python3
"""Lightweight preflight checks for M3FD detection dataset layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check M3FD detection dataset files without importing Detectron2.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Path to the local dataset root.")
    parser.add_argument(
        "--layout",
        choices=("m3fd-rgbt", "m3fd-ir-detectron2"),
        default="m3fd-rgbt",
        help="Dataset layout to validate.",
    )
    parser.add_argument("--split", default="train", help="Split name; expects <split>.txt in the layout split directory.")
    return parser.parse_args()


def require_path(path: Path, description: str, should_be_dir: bool = True) -> list[str]:
    if not path.exists():
        return [f"missing {description}: {path}"]
    if should_be_dir and not path.is_dir():
        return [f"expected {description} to be a directory: {path}"]
    if not should_be_dir and not path.is_file():
        return [f"expected {description} to be a file: {path}"]
    return []


def read_split(split_file: Path) -> list[str]:
    with split_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def sample_stem(entry: str) -> str:
    return Path(entry).stem


def find_image(directory: Path, entry: str) -> Path | None:
    entry_path = directory / entry
    if entry_path.is_file():
        return entry_path
    stem = sample_stem(entry)
    for ext in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def find_label(directory: Path, entry: str) -> Path | None:
    entry_path = directory / entry
    if entry_path.is_file():
        return entry_path
    candidate = directory / f"{sample_stem(entry)}.txt"
    return candidate if candidate.is_file() else None


def check_m3fd_rgbt(dataset_root: Path, split: str) -> tuple[list[str], list[str], int]:
    ir_dir = dataset_root / "ir"
    vi_dir = dataset_root / "vi"
    labels_dir = dataset_root / "labels"
    split_file = dataset_root / "meta" / f"{split}.txt"
    errors: list[str] = []
    warnings: list[str] = []
    for path, desc, is_dir in (
        (dataset_root, "dataset root", True),
        (ir_dir, "IR image directory", True),
        (vi_dir, "VI image directory", True),
        (labels_dir, "label directory", True),
        (split_file, "split file", False),
    ):
        errors.extend(require_path(path, desc, is_dir))
    if errors:
        return errors, warnings, 0
    entries = read_split(split_file)
    for entry in entries[:5]:
        if find_image(ir_dir, entry) is None:
            errors.append(f"missing IR image for split entry '{entry}' in {ir_dir}")
        if find_image(vi_dir, entry) is None:
            errors.append(f"missing VI image for split entry '{entry}' in {vi_dir}")
        if find_label(labels_dir, entry) is None:
            errors.append(f"missing YOLO label for split entry '{entry}' in {labels_dir}")
    if not entries:
        warnings.append(f"split file is empty: {split_file}")
    return errors, warnings, len(entries)


def check_m3fd_ir_detectron2(dataset_root: Path, split: str) -> tuple[list[str], list[str], int]:
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    split_file = dataset_root / "splits" / f"{split}.txt"
    errors: list[str] = []
    warnings: list[str] = []
    for path, desc, is_dir in (
        (dataset_root, "dataset root", True),
        (images_dir, "image directory", True),
        (annotations_dir, "annotation directory", True),
        (split_file, "split file", False),
    ):
        errors.extend(require_path(path, desc, is_dir))
    count = 0
    if not errors:
        entries = read_split(split_file)
        count = len(entries)
        for entry in entries[:5]:
            if find_image(images_dir, entry) is None:
                errors.append(f"missing image for split entry '{entry}' in {images_dir}")
            if find_label(annotations_dir, entry) is None:
                warnings.append(f"no per-image txt annotation found for sampled entry '{entry}' in {annotations_dir}")
        if not entries:
            warnings.append(f"split file is empty: {split_file}")
    return errors, warnings, count


def check_dataset(dataset_root: Path, layout: str, split: str) -> tuple[list[str], list[str], int]:
    root = dataset_root.expanduser().resolve()
    if layout == "m3fd-rgbt":
        return check_m3fd_rgbt(root, split)
    return check_m3fd_ir_detectron2(root, split)


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    errors, warnings, count = check_dataset(root, args.layout, args.split)
    print("M3FD detection dataset check")
    print(f"  dataset_root: {root}")
    print(f"  layout: {args.layout}")
    print(f"  split: {args.split}")
    print(f"  split_samples: {count}")
    for warning in warnings:
        print(f"  warning: {warning}")
    if errors:
        print("status: failed")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
