"""Deterministic NIfTI cache for prepared phases and subtraction images.

Key (``cache_key``) = sha256 of canonical JSON::

    {cache_schema_version, kind, sources: [{uid, size, mtime_ns, head_sha256}],
     target_size, normalisation, resize_backend, align_corners, dtype}

so entries built with different preparation options, backends or source files
never share a filename (the legacy key only used ``abspath(image_root)`` + size,
so MONAI and custom preparation shared entries).

Entry layout: ``<root>/<kind>/<key[:2]>/<key>.nii.gz`` (+ ``<key>.json`` with the
key payload for audits, + ``<key>.lock``).  The NIfTI ``descrip`` header stores
``mbc<version>:<sha256 of the float32 voxel bytes>``; a load failure or checksum
mismatch marks the entry corrupt and it is rebuilt from the read-only sources.

Concurrency: readers hold a shared lock while reading; a builder takes the
exclusive lock, re-checks, writes a temp file in the entry directory (same
filesystem) and publishes it with ``os.replace``.  Augmentations are never cached.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import tempfile
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np

from mribreast.imaging.locking import ReadWriteFileLock

CACHE_SCHEMA_VERSION = 1
_DESCRIP_PREFIX = f"mbc{CACHE_SCHEMA_VERSION}:"
_HEAD_BYTES = 1 << 16


class CorruptCacheError(RuntimeError):
    """A cache entry exists but cannot be read or fails its checksum."""


def source_fingerprint(path: str | os.PathLike) -> dict[str, Any]:
    """Cheap, change-sensitive fingerprint: size + mtime_ns + sha256 of the first 64 KiB."""
    path = Path(path)
    stat = path.stat()
    with open(path, "rb") as handle:
        head = handle.read(_HEAD_BYTES)
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
            "head_sha256": hashlib.sha256(head).hexdigest()}


def cache_key_payload(kind: str, sources: Sequence[tuple[str, str | os.PathLike]],
                      params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "kind": kind,
        "sources": [{"uid": uid, **source_fingerprint(path)} for uid, path in sources],
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(params.items())},
    }


def cache_key(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_checksum(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data, dtype=np.float32).tobytes()).hexdigest()


def _is_corruption(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (CorruptCacheError, EOFError, gzip.BadGzipFile, zlib.error,
                                nib.filebasedimages.ImageFileError)):
            return True
        if isinstance(current, OSError) and not isinstance(current, PermissionError):
            message = str(current).lower()
            if "bytes" in message and ("expected" in message or "got" in message):
                return True
        if isinstance(current, ValueError) and "buffer" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def save_entry(path: Path, data: np.ndarray, affine: np.ndarray, payload: Mapping[str, Any] | None = None) -> None:
    """Write NIfTI (+ checksum) to a temp file beside ``path`` and publish atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    image = nib.Nifti1Image(data, np.asarray(affine, dtype=np.float64))
    image.header["descrip"] = (_DESCRIP_PREFIX + array_checksum(data)).encode("ascii")
    image.header.set_xyzt_units("mm", "sec")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".nii.gz", dir=path.parent)
    os.close(fd)
    try:
        nib.save(image, tmp)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
    if payload is not None:
        meta = path.with_name(path.name.replace(".nii.gz", ".json"))
        fd, tmp = tempfile.mkstemp(prefix=f".{meta.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
            os.replace(tmp, meta)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)


def load_entry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Eagerly read an entry and verify its checksum (``CorruptCacheError`` on mismatch)."""
    image = nib.load(str(path))
    descrip = image.header["descrip"].tobytes().split(b"\x00", 1)[0].decode("ascii", "replace")
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if not descrip.startswith(_DESCRIP_PREFIX) or descrip[len(_DESCRIP_PREFIX):] != array_checksum(data):
        raise CorruptCacheError(f"checksum mismatch in {path}")
    return np.ascontiguousarray(data), np.asarray(image.affine, dtype=np.float64)


@dataclass(frozen=True)
class CacheResult:
    data: np.ndarray
    affine: np.ndarray
    key: str
    path: Path | None
    built: bool


class VolumeCache:
    """Get-or-build cache rooted at ``root`` (``None`` -> compute without caching)."""

    def __init__(self, root: str | os.PathLike | None):
        self.root = Path(root) if root else None

    def entry_path(self, kind: str, key: str) -> Path:
        if self.root is None:
            raise ValueError("cache root not configured")
        return self.root / kind / key[:2] / f"{key}.nii.gz"

    def get_or_build(self, kind: str, payload: Mapping[str, Any],
                     build: Callable[[], tuple[np.ndarray, np.ndarray]], *, source: str = "") -> CacheResult:
        key = cache_key(payload)
        if self.root is None:
            data, affine = build()
            return CacheResult(np.asarray(data, np.float32), np.asarray(affine), key, None, True)
        path = self.entry_path(kind, key)
        lock = ReadWriteFileLock(path.with_name(f"{key}.lock"))
        corrupt: BaseException | None = None
        with lock.shared():
            if path.exists():
                try:
                    data, affine = load_entry(path)
                    return CacheResult(data, affine, key, path, False)
                except Exception as error:  # noqa: BLE001 - classified below
                    if not _is_corruption(error):
                        raise RuntimeError(f"cannot read MRI cache {path}") from error
                    corrupt = error
        with lock.exclusive():
            if path.exists():
                try:
                    data, affine = load_entry(path)
                    return CacheResult(data, affine, key, path, False)
                except Exception as error:  # noqa: BLE001
                    if not _is_corruption(error):
                        raise RuntimeError(f"cannot read MRI cache {path}") from error
                    corrupt = error
            if corrupt is not None:
                warnings.warn(f"Rebuilding corrupt MRI cache {path} from {source}: {corrupt}", RuntimeWarning,
                              stacklevel=2)
            try:
                data, affine = build()
            except Exception as error:
                raise RuntimeError(f"cannot create MRI cache {path} from source {source}") from error
            data = np.ascontiguousarray(data, dtype=np.float32)
            save_entry(path, data, affine, payload)
            return CacheResult(data, np.asarray(affine, dtype=np.float64), key, path, True)
