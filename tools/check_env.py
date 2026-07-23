"""Check the PSMAF-Net development environment without installing anything.

The script reports whether the core UNIV, Detectron2, and MMSegmentation
runtime dependencies are importable. Missing packages are reported as
``missing`` and never cause a non-zero exit status.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class PackageCheck:
    """Description for one import/version check."""

    label: str
    import_name: str
    version_attr: str = "__version__"


def _import_optional(import_name: str) -> ModuleType | None:
    """Import a module, returning None for any import-time failure."""

    try:
        return importlib.import_module(import_name)
    except Exception:
        return None


def _module_version(module: ModuleType | None, version_attr: str = "__version__") -> str:
    """Return a module version string or ``unknown`` when unavailable."""

    if module is None:
        return "missing"
    return str(getattr(module, version_attr, "unknown"))


def _print_status(label: str, value: str) -> None:
    """Print one normalized check row."""

    print(f"{label}: {value}")


def check_python() -> None:
    """Report the active Python runtime."""

    version = platform.python_version()
    executable = sys.executable
    _print_status("Python version", f"{version} ({executable})")


def check_torch_and_cuda() -> None:
    """Report PyTorch version and CUDA availability without requiring PyTorch."""

    torch = _import_optional("torch")
    _print_status("PyTorch version", _module_version(torch))

    if torch is None:
        _print_status("CUDA availability", "missing")
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        _print_status("CUDA availability", "unknown")
        return

    available = bool(cuda.is_available())
    if not available:
        _print_status("CUDA availability", "False")
        return

    cuda_version = getattr(torch.version, "cuda", "unknown")
    device_count = cuda.device_count() if hasattr(cuda, "device_count") else "unknown"
    _print_status("CUDA availability", f"True (torch CUDA {cuda_version}, devices: {device_count})")


def check_packages() -> None:
    """Report importability and versions for required framework packages."""

    checks = [
        PackageCheck("torchvision", "torchvision"),
        PackageCheck("timm", "timm"),
        PackageCheck("numpy", "numpy"),
        PackageCheck("opencv-python", "cv2"),
        PackageCheck("peft", "peft"),
        PackageCheck("PyYAML / yaml", "yaml"),
        PackageCheck("packaging", "packaging"),
        PackageCheck("detectron2", "detectron2"),
        PackageCheck("mmcv", "mmcv"),
        PackageCheck("mmengine", "mmengine"),
        PackageCheck("mmsegmentation", "mmseg"),
    ]

    for check in checks:
        module = _import_optional(check.import_name)
        _print_status(check.label, _module_version(module, check.version_attr))


def main() -> int:
    """Run all checks and always return success."""

    check_python()
    check_torch_and_cuda()
    check_packages()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
