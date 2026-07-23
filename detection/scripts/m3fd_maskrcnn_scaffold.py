#!/usr/bin/env python3
"""Shared scaffold utilities for M3FD-IR Mask R-CNN + UNIV experiments."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class M3FDExperiment:
    """Typed subset of the M3FD-IR Mask R-CNN experiment metadata."""

    name: str
    dataset_name: str
    modality: str
    head: str
    framework: str
    image_size: tuple[int, int]
    fine_tuning_steps: int
    optimizer: str
    base_learning_rate: float
    weight_decay: float
    batch_size: int
    train_images: int
    test_images: int


def load_experiment(path: Path) -> tuple[M3FDExperiment, dict[str, Any]]:
    """Load and validate the lightweight experiment metadata YAML."""

    raw = parse_simple_yaml(path)

    dataset = _mapping(raw, "dataset")
    model = _mapping(raw, "model")
    input_cfg = _mapping(raw, "input")
    resolution = _mapping(input_cfg, "resolution")
    training = _mapping(raw, "training")
    splits = _mapping(dataset, "splits")

    experiment = M3FDExperiment(
        name=str(raw["experiment"]),
        dataset_name=str(dataset["name"]),
        modality=str(dataset["modality"]),
        head=str(model["head"]),
        framework=str(model["framework"]),
        image_size=(int(resolution["height"]), int(resolution["width"])),
        fine_tuning_steps=int(training["fine_tuning_steps"]),
        optimizer=str(training["optimizer"]),
        base_learning_rate=float(training["base_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        batch_size=int(training["batch_size"]),
        train_images=int(splits["train_images"]),
        test_images=int(splits["test_images"]),
    )
    return experiment, raw


def detectron2_config_template(experiment: M3FDExperiment, output_dir: Path) -> str:
    """Render a minimal Detectron2 config fragment for downstream integration."""

    height, width = experiment.image_size
    train_name = f"{experiment.name}_train"
    test_name = f"{experiment.name}_test"
    return "\n".join(
        [
            "# Generated scaffold fragment; review before launching Detectron2 training.",
            "MODEL:",
            "  META_ARCHITECTURE: GeneralizedRCNN",
            "  ROI_HEADS:",
            "    NAME: StandardROIHeads",
            "  WEIGHTS: ''  # Set to converted UNIV/Mask R-CNN initialization checkpoint.",
            "DATASETS:",
            f"  TRAIN: ('{train_name}',)",
            f"  TEST: ('{test_name}',)",
            "INPUT:",
            f"  MIN_SIZE_TRAIN: ({height},)",
            f"  MAX_SIZE_TRAIN: {width}",
            f"  MIN_SIZE_TEST: {height}",
            f"  MAX_SIZE_TEST: {width}",
            "SOLVER:",
            f"  MAX_ITER: {experiment.fine_tuning_steps}",
            f"  IMS_PER_BATCH: {experiment.batch_size}",
            f"  BASE_LR: {experiment.base_learning_rate}",
            f"  WEIGHT_DECAY: {experiment.weight_decay}",
            f"  OPTIMIZER: {experiment.optimizer}",
            f"OUTPUT_DIR: {output_dir.as_posix()}",
            "",
        ]
    )


def split_file_count(path: Path) -> int:
    """Count non-empty, non-comment lines in an M3FD split file."""

    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))



def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small repository-owned metadata YAML without external deps.

    This intentionally supports only the subset used by
    detection/configs/m3fd_ir_maskrcnn_univ.yaml: nested mappings by two-space
    indentation, scalar strings, booleans, ints, floats, and folded note blocks.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        elif value == ">-":
            block_lines: list[str] = []
            while index < len(lines):
                next_line = lines[index]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_stripped and next_indent <= indent:
                    break
                index += 1
                if next_stripped:
                    block_lines.append(next_stripped)
            current[key] = " ".join(block_lines)
        else:
            current[key] = parse_scalar(value)
    return root


def parse_scalar(value: str) -> Any:
    """Parse a scalar from the repository-owned metadata YAML subset."""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try_int = value.replace("_", "")
    if try_int.isdigit() or (try_int.startswith("-") and try_int[1:].isdigit()):
        return int(try_int)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value

def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping")
    return value
