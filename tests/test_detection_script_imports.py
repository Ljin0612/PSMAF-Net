import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_runner_can_be_imported_as_package():
    import detection.scripts.run_m3fd_maskrcnn_smoke  # noqa: F401


@pytest.mark.parametrize(
    ("script", "description"),
    [
        (
            "detection/scripts/run_m3fd_maskrcnn_smoke.py",
            "Run an M3FD-IR bbox detection smoke pipeline.",
        ),
        (
            "detection/scripts/train_m3fd_maskrcnn_univ.py",
            "Prepare the M3FD-IR Mask R-CNN + UNIV detection training entry point.",
        ),
        (
            "detection/scripts/eval_m3fd_maskrcnn_univ.py",
            "Prepare the M3FD-IR Mask R-CNN + UNIV detection evaluation entry point.",
        ),
    ],
)
def test_help_supports_direct_execution(script: str, description: str):
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert description in result.stdout
