#!/usr/bin/env python3
"""Detectron2 dataset registration helpers for M3FD COCO annotations."""

from __future__ import annotations

THING_CLASSES = ["people", "car", "bus", "motorcycle", "lamp", "truck"]
THING_DATASET_ID_TO_CONTIGUOUS_ID = {dataset_id: contiguous_id for contiguous_id, dataset_id in enumerate(range(1, 7))}


def register_m3fd_coco(name: str, image_root: str, json_file: str) -> None:
    """Register an M3FD COCO-style dataset with Detectron2."""
    try:
        from detectron2.data import MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
    except ImportError as exc:
        raise ImportError(
            "detectron2 is required to register M3FD COCO datasets. "
            "Install a Detectron2 build compatible with your torch/CUDA environment."
        ) from exc

    register_coco_instances(
        name,
        {
            "thing_classes": THING_CLASSES,
            "thing_dataset_id_to_contiguous_id": THING_DATASET_ID_TO_CONTIGUOUS_ID,
        },
        json_file,
        image_root,
    )
    MetadataCatalog.get(name).set(
        thing_classes=THING_CLASSES,
        thing_dataset_id_to_contiguous_id=THING_DATASET_ID_TO_CONTIGUOUS_ID,
    )
