from __future__ import annotations

import subprocess
import sys


def test_module_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "form_factors_beads", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--topology" in completed.stdout
    assert "--dry-run" in completed.stdout
