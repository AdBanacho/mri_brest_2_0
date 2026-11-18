"""Read-only access to the reference (legacy) repository for parity tests.

* Location: ``$MRI_BREAST_LEGACY_REPO``, else ``C:\\code\\mri_brest`` when it exists; otherwise
  parity tests are skipped.
* Nothing is written into the reference tree: modules are loaded with
  ``sys.dont_write_bytecode = True`` (no ``__pycache__``), and the session
  guard in ``conftest.py`` asserts that the set of files is unchanged.
* :func:`load_file_module` loads one legacy file without importing the
  ``mriBreastDuke`` package (for dependency-free modules such as
  ``checkpoint_selection.py`` or ``threshold_tuning.py``);
  :func:`import_legacy` imports ``mriBreastDuke.<module>`` through the package
  when the module itself uses package-relative imports.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

ENV_VAR = "MRI_BREAST_LEGACY_REPO"
DEFAULT_REPO = Path(r"C:\code\mri_brest")
REFERENCE_SHA = "127a26046db20e5e506f44ce6d4c1946d34c6416"
_PREFIX = "legacy_ref_"


def legacy_repo() -> Path | None:
    configured = os.environ.get(ENV_VAR)
    candidate = Path(configured) if configured else DEFAULT_REPO
    if (candidate / "mriBreastDuke" / "checkpoint_selection.py").is_file():
        return candidate.resolve()
    return None


def legacy_sha(repo: Path) -> str | None:
    """HEAD of the reference checkout (read-only ``git rev-parse``), or None."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=30, check=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@contextlib.contextmanager
def no_bytecode() -> Iterator[None]:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def load_file_module(repo: Path, relative: str) -> ModuleType:
    """Load ``repo/relative`` as an isolated module (no package import, no bytecode)."""
    path = repo / relative
    name = _PREFIX + relative.replace("/", "_").replace("\\", "_").removesuffix(".py")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with no_bytecode():
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return module


def import_legacy(repo: Path, dotted: str) -> ModuleType:
    """Import ``mriBreastDuke.<...>`` from the reference repo (sys.path entry removed afterwards)."""
    root = str(repo)
    with no_bytecode():
        sys.path.insert(0, root)
        try:
            module = importlib.import_module(dotted)
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(root)
    # Compare resolved paths: "C:/code/x" (env var) and "C:\code\x" name the same repository.
    module_file = Path(str(getattr(module, "__file__", "") or "")).resolve()
    if not module_file.is_relative_to(Path(root).resolve()):
        raise ImportError(f"{dotted} resolved outside the reference repo: {module.__file__}")
    return module


def tree_state(repo: Path) -> set[str]:
    """All paths below ``repo`` (excluding ``.git``) - used to prove the reference stays untouched."""
    out = set()
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        rel = os.path.relpath(dirpath, repo)
        out.add(rel)
        out.update(os.path.join(rel, f) for f in filenames)
    return out
