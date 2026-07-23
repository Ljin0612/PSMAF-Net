#!/usr/bin/env python3
"""MMSegmentation / UperNet downstream entry placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation/upernet_psmaf.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / args.config
    print("Segmentation route: UperNet / MMSegmentation")
    print(f"Config: {config}")
    if args.dry_run:
        return 0
    print("Segmentation training is not implemented in this scaffold.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
