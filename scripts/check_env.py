#!/usr/bin/env python3
"""Lightweight environment sanity check for the PSMAF-Net scaffold."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


OPTIONAL_PACKAGES = ["torch", "detectron2", "mmseg", "mmcv"]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print(f"Repository: {repo_root}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for package in OPTIONAL_PACKAGES:
        status = "available" if importlib.util.find_spec(package) else "missing"
        print(f"{package}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
