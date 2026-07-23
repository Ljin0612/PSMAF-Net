"""Lightweight semantic segmentation dataset layout checker.

The checker validates dataset files only. It does not import MMSegmentation,
UNIV, or any training code.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from collections.abc import Iterable
from pathlib import Path

yaml = importlib.import_module("yaml") if importlib.util.find_spec("yaml") else None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_SUFFIXES = {".png", ".jpg", ".jpeg"}
PLACEHOLDER_ROOTS = {"/path/to/MSRS", "/path/to/MSRS-IR", "/path/to/dataset"}
SPLITS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check a semantic segmentation dataset layout from a YAML config. "
            "Directory schemas with train/val/test img_dir+ann_dir and legacy "
            "images/masks/splits schemas are supported."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Segmentation YAML config.")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override dataset.data_root or dataset.paths.root from the YAML config.",
    )
    return parser


def resolve_path(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ValueError("PyYAML is required to read segmentation YAML configs; install pyyaml or run --help only")
    if not config_path.is_file():
        raise ValueError(f"config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must contain a YAML mapping: {config_path}")
    return loaded


def get_data_root(dataset: dict, override: Path | None) -> Path:
    raw_root = override or dataset.get("data_root") or dataset.get("paths", {}).get("root")
    if raw_root is None:
        raise ValueError("missing dataset data root; set dataset.data_root or pass --data-root")
    root = Path(raw_root)
    if str(root) in PLACEHOLDER_ROOTS:
        raise ValueError(f"data_root is still a placeholder; pass --data-root with the real dataset path: {root}")
    if not root.is_dir():
        raise ValueError(f"data_root does not exist or is not a directory: {root}")
    return root


def collect_files(directory: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in suffix_set
    }


def check_directory_split(root: Path, split_name: str, split_config: dict) -> bool:
    img_dir = resolve_path(root, split_config["img_dir"])
    ann_dir = resolve_path(root, split_config["ann_dir"])
    ok = True

    for label, directory in (("image directory", img_dir), ("annotation directory", ann_dir)):
        if directory.is_dir():
            print(f"[OK] {split_name} {label}: {directory}")
        else:
            print(f"[MISSING] {split_name} {label}: {directory}")
            ok = False

    if not ok:
        return False

    images = collect_files(img_dir, IMAGE_SUFFIXES)
    annotations = collect_files(ann_dir, ANNOTATION_SUFFIXES)
    print(f"[COUNT] {split_name}: images={len(images)} annotations={len(annotations)}")

    if not images:
        print(f"[ERROR] {split_name}: no images found in {img_dir}")
        ok = False
    if not annotations:
        print(f"[ERROR] {split_name}: no annotations found in {ann_dir}")
        ok = False

    missing_annotations = sorted(set(images) - set(annotations))
    extra_annotations = sorted(set(annotations) - set(images))
    if missing_annotations:
        print(
            f"[ERROR] {split_name}: {len(missing_annotations)} images are missing annotations. "
            f"Examples: {', '.join(missing_annotations[:5])}"
        )
        ok = False
    if extra_annotations:
        print(
            f"[WARNING] {split_name}: {len(extra_annotations)} annotations have no matching image. "
            f"Examples: {', '.join(extra_annotations[:5])}"
        )
        ok = False
    if len(images) != len(annotations):
        print(f"[ERROR] {split_name}: image and annotation counts differ.")
        ok = False

    return ok


def has_directory_schema(dataset: dict) -> bool:
    return any(
        isinstance(dataset.get(split), dict)
        and {"img_dir", "ann_dir"}.issubset(dataset[split])
        for split in SPLITS
    )


def check_directory_schema(root: Path, dataset: dict) -> int:
    ok = True
    for split in SPLITS:
        split_config = dataset.get(split)
        if not split_config:
            continue
        if not isinstance(split_config, dict) or not {"img_dir", "ann_dir"}.issubset(split_config):
            print(f"[ERROR] {split}: expected img_dir and ann_dir entries.")
            ok = False
            continue
        ok = check_directory_split(root, split, split_config) and ok
    return 0 if ok else 1


def check_legacy_schema(root: Path, dataset: dict) -> int:
    paths = dataset.get("paths", {})
    images = resolve_path(root, paths.get("images", "images"))
    masks = resolve_path(root, paths.get("masks", "masks"))
    split_keys = [key for key in ("train_split", "val_split", "test_split") if key in paths]
    ok = True

    for label, path, is_valid in (
        ("images", images, images.is_dir()),
        ("masks", masks, masks.is_dir()),
    ):
        if is_valid:
            print(f"[OK] {label}: {path}")
        else:
            print(f"[MISSING] {label}: {path}")
            ok = False

    if not split_keys:
        print("[WARNING] legacy schema has no split files configured.")
    for key in split_keys:
        split = resolve_path(root, paths[key])
        if split.is_file():
            print(f"[OK] {key}: {split}")
        else:
            print(f"[MISSING] {key}: {split}")
            ok = False

    return 0 if ok else 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        dataset = config.get("dataset", {})
        if not isinstance(dataset, dict):
            raise ValueError("config dataset entry must be a mapping")
        root = get_data_root(dataset, args.data_root)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    if has_directory_schema(dataset):
        return check_directory_schema(root, dataset)
    return check_legacy_schema(root, dataset)


if __name__ == "__main__":
    raise SystemExit(main())
