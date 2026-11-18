"""Geometry conventions for NIfTI series (see docs/imaging.md).

* Arrays keep nibabel's on-disk voxel axis order ``(i, j, k)``; no reorientation
  to RAS/LPS is performed (identical to the legacy MONAI ``LoadImaged`` with the
  nibabel reader and ``as_closest_canonical=False``).  For Duke volumes converted
  with SimpleITK this is ``(column, row, slice)``.
* ``affine`` maps voxel indices to world millimetres (nibabel RAS+ convention).
* After an axis-aligned resize with ``align_corners=False`` the affine is updated
  by :func:`resized_affine` (voxel size scaled by ``in/out``, origin moved by half
  a voxel), which equals MONAI's MetaTensor affine.  The legacy cache wrote the
  *pre-resize* affine (stale ``img_meta_dict``); this implementation stores the
  correct one.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from mribreast.contracts import Geometry

NIFTI_SUFFIXES = (".nii.gz", ".nii")
_UID_PATTERN = re.compile(r"^[A-Za-z0-9._\-]+$")


class SeriesNotFoundError(FileNotFoundError):
    """No ``<uid>.nii.gz`` / ``<uid>.nii`` exists for a series UID."""


def parse_series_uids(value: Any) -> tuple[str, ...]:
    """Ordered series UIDs from a tuple/list/array or a ``|``-joined string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    if isinstance(value, str):
        items = value.split("|")
    else:
        items = list(value)
    return tuple(str(item).strip() for item in items if str(item).strip())


def validate_uid(uid: str) -> str:
    uid = str(uid).strip()
    if not uid or uid in (".", "..") or not _UID_PATTERN.match(uid):
        raise ValueError(f"invalid series UID {uid!r} (expected letters, digits, '.', '_' or '-')")
    return uid


def series_path(image_root: str | os.PathLike, uid: str) -> Path:
    """Path of ``<uid>.nii.gz`` (preferred) or ``<uid>.nii`` under ``image_root``."""
    uid = validate_uid(uid)
    root = Path(image_root)
    for suffix in NIFTI_SUFFIXES:
        candidate = root / f"{uid}{suffix}"
        if candidate.is_file():
            return candidate
    raise SeriesNotFoundError(f"series {uid}: neither {uid}.nii.gz nor {uid}.nii found in {root}")


def _as_3d(data: np.ndarray, path: Path) -> np.ndarray:
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {data.shape}")
    return data


def load_series(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read a NIfTI volume as float32 ``[i, j, k]`` plus its 4x4 affine (no reorientation)."""
    path = Path(path)
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    return np.ascontiguousarray(_as_3d(data, path)), np.asarray(image.affine, dtype=np.float64)


def read_header_geometry(path: str | os.PathLike) -> Geometry:
    """Geometry of a NIfTI file from its header only (no voxel data read)."""
    image = nib.load(str(path))
    shape = tuple(int(s) for s in image.shape)
    while len(shape) > 3 and shape[-1] == 1:
        shape = shape[:-1]
    return geometry_from_affine(shape, np.asarray(image.affine, dtype=np.float64))


def geometry_from_affine(shape: Sequence[int], affine: np.ndarray | None) -> Geometry:
    if affine is None:
        return Geometry(shape=tuple(int(s) for s in shape))
    affine = np.asarray(affine, dtype=np.float64)
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(affine))
    orientation = "".join(nib.aff2axcodes(affine))
    return Geometry(shape=tuple(int(s) for s in shape), affine=tuple(tuple(float(v) for v in row) for row in affine),
                    spacing=spacing, orientation=orientation)


def resized_affine(affine: np.ndarray, in_shape: Sequence[int], out_shape: Sequence[int]) -> np.ndarray:
    """Affine of a volume resized from ``in_shape`` to ``out_shape`` (align_corners=False).

    Output voxel ``o`` samples input coordinate ``(o + 0.5) * in/out - 0.5``, so
    ``A_out = A_in @ [[diag(in/out), 0.5*(in/out - 1)], [0, 1]]``.
    """
    affine = np.asarray(affine, dtype=np.float64)
    scale = np.asarray(in_shape, dtype=np.float64) / np.asarray(out_shape, dtype=np.float64)
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = 0.5 * (scale - 1.0)
    return affine @ transform


def cropped_affine(affine: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    """Affine after cropping voxels ``[starts, ...)`` (origin moves to the first kept voxel)."""
    affine = np.asarray(affine, dtype=np.float64)
    transform = np.eye(4)
    transform[: len(starts), 3] = np.asarray(starts, dtype=np.float64)
    return affine @ transform


def slice_affine(affine: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Affine of slice ``index`` along ``axis`` (still 4x4; the sliced axis keeps its column)."""
    starts = [0, 0, 0]
    starts[axis] = int(index)
    return cropped_affine(affine, starts)


def same_grid(shape_a: Sequence[int], affine_a: np.ndarray, shape_b: Sequence[int], affine_b: np.ndarray) -> bool:
    """Legacy geometry agreement test (shape equality, ``np.allclose(rtol=1e-4, atol=1e-3)``)."""
    return tuple(shape_a) == tuple(shape_b) and bool(
        np.allclose(np.asarray(affine_a), np.asarray(affine_b), rtol=1e-4, atol=1e-3))
