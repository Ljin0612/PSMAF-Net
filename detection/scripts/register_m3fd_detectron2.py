#!/usr/bin/env python3
"""Detectron2 dataset registration helpers for M3FD COCO annotations."""

from __future__ import annotations


def register_m3fd_coco(name: str, image_root: str, json_file: str) -> None:
    """Register an M3FD COCO-style dataset with Detectron2."""
    try:
        from detectron2.data.datasets import register_coco_instances
    except ImportError as exc:
        raise ImportError(
            "detectron2 is required to register M3FD COCO datasets. "
            "Install a Detectron2 build compatible with your torch/CUDA environment."
        ) from exc

    register_coco_instances(name, {}, json_file, image_root)
