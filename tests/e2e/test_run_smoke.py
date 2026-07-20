import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.gpu]


def test_run_skip_smoke(project_root):
    if not __import__("torch").cuda.is_available():
        pytest.skip("CUDA required for run.py smoke test")
    cmd = [
        sys.executable,
        "run.py",
        "--skip",
        "--batch_size",
        "4",
        "--n_size",
        "50",
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
