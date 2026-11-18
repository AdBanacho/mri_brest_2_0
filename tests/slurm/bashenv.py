"""Locate a usable Bash for the controller tests (Git Bash on Windows, never the WSL stub)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def find_bash() -> str | None:
    explicit = os.environ.get("MRIBREAST_BASH")
    if explicit:
        return explicit
    if sys.platform == "win32":
        candidates = []
        git = shutil.which("git")
        if git:
            git_root = Path(git).resolve().parents[1]
            candidates += [git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"]
        candidates.append(Path(r"C:\Program Files\Git\bin\bash.exe"))
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        found = shutil.which("bash")
        return found if found and "system32" not in found.lower() else None
    return shutil.which("bash")


BASH = find_bash()
