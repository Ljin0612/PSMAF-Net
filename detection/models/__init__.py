"""Detectron2 model components for the UNIV-original detection baseline.

Modules are intentionally not imported eagerly so lightweight repository tooling
can inspect this package on hosts where Detectron2 is not installed.
"""
from .psmaf_yolov8 import PSMAFYOLOv8

__all__ = ["PSMAFYOLOv8"]
