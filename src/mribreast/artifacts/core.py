"""Small, dependency-free helpers for audited, atomic artifacts."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from mribreast import MANIFEST_SCHEMA_VERSION, __version__

COMPLETE_FLAG = "COMPLETE"
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch", "monai",
                    "pytorch-lightning", "torchmetrics", "nibabel", "openpyxl", "joblib", "pillow", "tifffile")


class ArtifactMismatch(RuntimeError):
    """Saved artifacts do not belong to the requested configuration/data."""


def file_fingerprint(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def paths_fingerprint(paths: Mapping[str, str | os.PathLike | None], *, directory_mode: str = "listing") -> dict[str, str]:
    """Fingerprint each configured input.

    Files -> content hash. Directories -> hash of sorted (relative name, size) listing
    (contents of large imaging trees are not hashed; caches are keyed separately).
    Missing/empty -> "absent".
    """
    out: dict[str, str] = {}
    for key, value in paths.items():
        if not value:
            out[key] = "absent"
            continue
        path = Path(value)
        if path.is_file():
            out[key] = file_fingerprint(path)
        elif path.is_dir():
            digest = hashlib.sha256()
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                digest.update(f"{child.relative_to(path).as_posix()}\0{child.stat().st_size}\0".encode())
            out[key] = "dir-listing:" + digest.hexdigest()
        else:
            out[key] = "absent"
    return out


def code_version(repo_dir: str | os.PathLike | None = None) -> dict[str, Any]:
    """Package version + git SHA/dirty flag when available (never fails)."""
    info: dict[str, Any] = {"package": "mribreast", "version": __version__, "git_sha": None, "git_dirty": None}
    repo = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[3]
    with contextlib.suppress(Exception):
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if sha.returncode == 0:
            info["git_sha"] = sha.stdout.strip()
            dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=10)
            info["git_dirty"] = bool(dirty.stdout.strip())
    return info


def environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_text_atomic(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_json_atomic(path: str | os.PathLike, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def is_complete(directory: str | os.PathLike) -> bool:
    return (Path(directory) / COMPLETE_FLAG).is_file()


def mark_complete(directory: str | os.PathLike, payload: Mapping[str, Any] | None = None) -> None:
    body = {"schema_version": MANIFEST_SCHEMA_VERSION, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    body.update(payload or {})
    write_json_atomic(Path(directory) / COMPLETE_FLAG, body)


@contextlib.contextmanager
def atomic_dir(final: str | os.PathLike, *, replace_incomplete: bool = True) -> Iterator[Path]:
    """Build a directory in a sibling temp dir and publish it with one rename.

    * If ``final`` exists and is complete -> FileExistsError (callers check first).
    * If ``final`` exists but is incomplete (interrupted run) -> it is removed
      before publishing when ``replace_incomplete``.
    The caller must call :func:`mark_complete` on the yielded directory before
    the block ends for the result to count as finished.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and is_complete(final):
        raise FileExistsError(f"already complete: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    try:
        yield staging
        if final.exists():
            if not replace_incomplete:
                raise FileExistsError(f"incomplete artifact exists: {final}")
            trash = final.with_name(f".{final.name}.trash-{os.getpid()}-{time.time_ns()}")
            os.replace(final, trash)
            shutil.rmtree(trash, ignore_errors=True)
        os.replace(staging, final)
        _fsync_dir(final.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
