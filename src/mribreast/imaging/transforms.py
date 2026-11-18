"""Joint image-mask spatial transforms with shared parameters (2D or 3D).

Images are ``[C, *S]`` and use continuous interpolation (bilinear/trilinear,
``align_corners=False``); masks are ``[*S]`` integer class codes and use
nearest neighbour (``nearest-exact``, the pixel-centre-consistent variant) so
class codes are never blended. Every op appends a JSON-serialisable record.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mribreast.imaging.volume import resize

TransformLog = list[dict[str, Any]]


def _check(image: np.ndarray, mask: np.ndarray | None) -> int:
    spatial = image.ndim - 1
    if mask is not None and tuple(mask.shape) != tuple(image.shape[1:]):
        raise ValueError(f"image grid {image.shape[1:]} and mask grid {mask.shape} differ")
    return spatial


def joint_crop(image: np.ndarray, mask: np.ndarray | None, starts: Sequence[int], ends: Sequence[int],
               log: TransformLog | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    _check(image, mask)
    region = tuple(slice(int(a), int(b)) for a, b in zip(starts, ends))
    image = np.ascontiguousarray(image[(slice(None),) + region])
    mask = None if mask is None else np.ascontiguousarray(mask[region])
    if log is not None:
        log.append({"op": "crop", "starts": [int(a) for a in starts], "ends": [int(b) for b in ends]})
    return image, mask


def joint_flip(image: np.ndarray, mask: np.ndarray | None, axes: Sequence[int],
               log: TransformLog | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Flip spatial ``axes`` (0-based over the spatial dims)."""
    _check(image, mask)
    axes = tuple(int(a) for a in axes)
    if axes:
        image = np.ascontiguousarray(np.flip(image, axis=tuple(a + 1 for a in axes)))
        mask = None if mask is None else np.ascontiguousarray(np.flip(mask, axis=axes))
    if log is not None:
        log.append({"op": "flip", "axes": list(axes)})
    return image, mask


def joint_resize(image: np.ndarray, mask: np.ndarray | None, size: Sequence[int],
                 log: TransformLog | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    spatial = _check(image, mask)
    size = tuple(int(s) for s in size)
    before = list(image.shape[1:])
    image = resize(image, size).astype(np.float32, copy=False)
    mask = None if mask is None else resize(mask, size, mode="nearest")
    if log is not None:
        log.append({"op": "resize", "from": before, "to": list(size),
                    "image_interpolation": {2: "bilinear", 3: "trilinear"}[spatial] + ",align_corners=False",
                    "mask_interpolation": "nearest-exact"})
    return image, mask


def roi_bbox(mask: np.ndarray, roi_classes: Sequence[int], margin: float) -> tuple[list[int], list[int]] | None:
    """Half-open bbox of ROI voxels expanded by ``margin * extent`` per side, clipped to the grid."""
    roi = np.isin(mask, np.asarray(list(roi_classes)))
    if not roi.any():
        return None
    starts, ends = [], []
    for axis in range(roi.ndim):
        other = tuple(a for a in range(roi.ndim) if a != axis)
        present = np.flatnonzero(roi.any(axis=other))
        lo, hi = int(present[0]), int(present[-1]) + 1
        pad = int(np.ceil(margin * (hi - lo)))
        starts.append(max(0, lo - pad))
        ends.append(min(roi.shape[axis], hi + pad))
    return starts, ends
