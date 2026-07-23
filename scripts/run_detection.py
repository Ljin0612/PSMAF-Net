#!/usr/bin/env python3
"""Detectron2 / Mask R-CNN downstream entry placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/detection/mask_rcnn_psmaf.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / args.config
    print(f"Detection route: Mask R-CNN / Detectron2")
    print(f"Config: {config}")
    if args.dry_run:
        return 0
    print("Detection training is not implemented in this scaffold.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
