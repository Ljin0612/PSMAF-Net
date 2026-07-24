#!/usr/bin/env python3
"""Check Detectron2 runtime imports without starting training."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys


def check_import(module_name: str, attr: str | None = None, label: str | None = None) -> bool:
    display = label or (f"{module_name}.{attr}" if attr else module_name)
    try:
        module = importlib.import_module(module_name)
        if attr is not None:
            getattr(module, attr)
    except Exception as exc:  # noqa: BLE001 - convert dependency failures to concise status lines.
        print(f"missing: {display} ({exc.__class__.__name__}: {exc})")
        return False
    print(f"ok: {display}")
    return True


def main() -> int:
    if importlib.util.find_spec("detectron2") is None:
        print("missing")
        return 1

    print("Detectron2 import check")
    print(f"python: {sys.version.split()[0]} ({platform.python_implementation()})")

    ok = True
    if sys.version_info < (3, 8):
        print("missing: Python >= 3.8")
        ok = False

    if check_import("torch"):
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"torch_cuda_version: {getattr(torch.version, 'cuda', None)}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    else:
        ok = False

    if check_import("torchvision"):
        import torchvision

        print(f"torchvision: {torchvision.__version__}")
    else:
        ok = False

    ok = check_import("detectron2") and ok
    ok = check_import("detectron2.config", "get_cfg", "detectron2.config.get_cfg") and ok
    ok = check_import("detectron2.engine", "DefaultTrainer", "detectron2.engine.DefaultTrainer") and ok
    ok = check_import("detectron2.data", "DatasetCatalog", "detectron2.data.DatasetCatalog") and ok
    ok = check_import("detectron2.data", "MetadataCatalog", "detectron2.data.MetadataCatalog") and ok

    print("status: passed" if ok else "status: missing dependencies")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
