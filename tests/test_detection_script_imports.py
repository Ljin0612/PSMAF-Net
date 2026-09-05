import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_runner_can_be_imported_as_package():
    import detection.scripts.run_m3fd_maskrcnn_smoke  # noqa: F401


def test_smoke_runner_help_supports_direct_execution():
    result = subprocess.run(
        [
            sys.executable,
            "detection/scripts/run_m3fd_maskrcnn_smoke.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run an M3FD-IR bbox detection smoke pipeline." in result.stdout
