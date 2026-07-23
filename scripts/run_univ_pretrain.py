#!/usr/bin/env python3
"""Entry point wrapper for original UNIV upstream pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved UNIV entry.")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    univ_root = repo_root / "third_party" / "UNIV"
    entry = univ_root / "pretrain_mcmae.py"
    print(f"UNIV root: {univ_root}")
    print(f"Expected pretraining entry: {entry}")
    if not entry.exists():
        print("UNIV source is not available yet; see third_party/UNIV/README.md.")
        return 2
    if args.dry_run:
        return 0
    print("Full UNIV pretraining is intentionally not launched by this scaffold wrapper.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
