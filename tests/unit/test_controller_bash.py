"""Run the Bash controller test suites (tests/slurm/test_*.sh) under pytest.

The suites use fake ``sbatch``/``python``/``module`` and need only Bash >= 4.3
and GNU coreutils (Git Bash on Windows).  Skipped when no suitable Bash exists.
Set ``MRI_BREAST_LEGACY_REPO`` to also cross-check the grid index mapping
against the reference controller.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.slurm.bashenv import BASH, ROOT

SUITES = sorted((ROOT / "tests" / "slurm").glob("test_*.sh"))


pytestmark = [pytest.mark.bash, pytest.mark.skipif(BASH is None, reason="bash (>= 4.3) not available")]


@pytest.mark.parametrize("suite", SUITES, ids=[s.stem for s in SUITES])
def test_bash_suite(suite: Path) -> None:
    proc = subprocess.run(
        [BASH, suite.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-4000:]
    assert "ok " in proc.stdout and "FAIL" not in output, output[-4000:]


def test_launchers_are_thin_and_lf() -> None:
    launcher = (ROOT / "run_experiment.sh").read_bytes()
    assert len(launcher.splitlines()) <= 5
    for path in [ROOT / "run_experiment.sh", ROOT / "run_experiment.slurm", ROOT / "experiment.conf",
                 *ROOT.joinpath("configs").glob("*.conf"), *ROOT.joinpath("tests", "slurm").rglob("*")]:
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            assert b"\r\n" not in path.read_bytes(), f"CRLF line endings in {path}"


def test_no_cluster_paths_in_package() -> None:
    """Helios /net/... paths may appear only in example configuration comments."""
    offenders = [
        str(p) for p in (ROOT / "src" / "mribreast").rglob("*.py") if "/net/" in p.read_text(encoding="utf-8")
    ]
    assert not offenders
