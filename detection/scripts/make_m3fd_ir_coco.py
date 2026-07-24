#!/usr/bin/env python3
"""Convert M3FD RGB-T YOLO annotations to bbox-only COCO-style IR annotations.

This converter emits bbox-only COCO annotations for object detection. It does
not emit segmentation polygons or masks. Downstream Detectron2 configs must
disable mask heads when using this JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from check_m3fd_detection import IMAGE_EXTENSIONS, find_image, find_label, read_split

CATEGORIES = [
    {"id": 0, "name": "people"},
    {"id": 1, "name": "car"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "motorcycle"},
    {"id": 4, "name": "lamp"},
    {"id": 5, "name": "truck"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert M3FD RGB-T YOLO labels to bbox-only COCO JSON for Detectron2.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--layout", default="m3fd-rgbt", choices=("m3fd-rgbt",))
    parser.add_argument("--modality", default="ir", choices=("ir",))
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return image_size_from_header(path)
    with Image.open(path) as image:
        return image.size


def image_size_from_header(path: Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without extra dependencies when Pillow is unavailable."""
    import struct

    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", header[16:24])
        if header.startswith(b"\xff\xd8"):
            handle.seek(2)
            while True:
                marker_start = handle.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    segment = handle.read(7)
                    height, width = struct.unpack(">HH", segment[3:7])
                    return width, height
                length_bytes = handle.read(2)
                if len(length_bytes) != 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                handle.seek(max(length - 2, 0), 1)
    raise RuntimeError(f"could not read image size for {path}; install pillow for broader format support")


def convert_box(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> tuple[list[float], bool]:
    x = (cx - bw / 2.0) * width
    y = (cy - bh / 2.0) * height
    w = bw * width
    h = bh * height
    x2 = x + w
    y2 = y + h
    clipped_x = min(max(x, 0.0), float(width))
    clipped_y = min(max(y, 0.0), float(height))
    clipped_x2 = min(max(x2, 0.0), float(width))
    clipped_y2 = min(max(y2, 0.0), float(height))
    clipped = [clipped_x, clipped_y, max(0.0, clipped_x2 - clipped_x), max(0.0, clipped_y2 - clipped_y)]
    return clipped, clipped != [x, y, w, h]


def convert(dataset_root: Path, split: str, output_json: Path) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve()
    image_dir = root / "ir"
    labels_dir = root / "labels"
    split_file = root / "meta" / f"{split}.txt"
    for path in (image_dir, labels_dir, split_file):
        if not path.exists():
            raise FileNotFoundError(f"required path not found: {path}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    per_class = Counter({category["name"]: 0 for category in CATEGORIES})
    skipped_labels = 0
    clipped_boxes = 0
    category_names = {category["id"]: category["name"] for category in CATEGORIES}
    annotation_id = 1

    for image_id, entry in enumerate(read_split(split_file), start=1):
        image_path = find_image(image_dir, entry)
        if image_path is None:
            skipped_labels += 1
            continue
        width, height = image_size(image_path)
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = find_label(labels_dir, entry)
        if label_path is None:
            continue
        with label_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parts = line.split()
                if len(parts) != 5:
                    skipped_labels += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                    cx, cy, box_w, box_h = (float(value) for value in parts[1:])
                except ValueError:
                    skipped_labels += 1
                    continue
                if class_id not in category_names:
                    skipped_labels += 1
                    continue
                bbox, clipped = convert_box(cx, cy, box_w, box_h, width, height)
                if clipped:
                    clipped_boxes += 1
                if bbox[2] <= 0 or bbox[3] <= 0:
                    skipped_labels += 1
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                per_class[category_names[class_id]] += 1
                annotation_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2)
    return {
        "annotation_type": "bbox_only",
        "has_segmentation": False,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "per_class_counts": dict(per_class),
        "skipped_labels": skipped_labels,
        "clipped_boxes": clipped_boxes,
        "output_json": str(output_json),
    }


def main() -> int:
    args = parse_args()
    stats = convert(args.dataset_root, args.split, args.output_json.expanduser().resolve())
    print("M3FD IR COCO conversion complete")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
