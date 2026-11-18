"""Deterministic 2D slice selection from 3D volumes.

A slice along ``slice_axis`` keeps the two remaining array axes in increasing
order: ``[C, *S]`` -> ``[C, H, W]`` with ``(H, W) = (S[a], S[b])``, ``a < b``,
``a, b != slice_axis``. ``slice_id`` is the index along ``slice_axis`` in the
ORIGINAL grid of the study's first series (the mask grid).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

SLICE_SELECTIONS = ("center", "uniform", "mask_max_area", "mask_all")


def roi_area_per_slice(mask: np.ndarray, axis: int, roi_classes: Sequence[int]) -> np.ndarray:
    roi = np.isin(np.asarray(mask), np.asarray(list(roi_classes)))
    other = tuple(a for a in range(roi.ndim) if a != axis)
    return roi.sum(axis=other)


def select_slices(n_slices: int, selection: str, count: int, roi_area: np.ndarray | None = None) -> list[int]:
    """Sorted slice indices.

    * ``center``: ``count`` contiguous slices centred on ``n // 2``.
    * ``uniform``: ``int((i + 0.5) * n / count)`` for ``i < count`` (bin centres).
    * ``mask_max_area``: the ``count`` slices with the largest ROI area (>0); ties -> lower index.
    * ``mask_all``: every slice with ROI area > 0.
    Mask-based strategies return ``[]`` when the ROI is empty.
    """
    if n_slices < 1:
        raise ValueError("volume has no slices")
    count = max(1, min(int(count), n_slices))
    if selection == "center":
        start = min(max(0, n_slices // 2 - count // 2), n_slices - count)
        return list(range(start, start + count))
    if selection == "uniform":
        return sorted({min(n_slices - 1, int((i + 0.5) * n_slices / count)) for i in range(count)})
    if selection in ("mask_max_area", "mask_all"):
        if roi_area is None:
            raise ValueError(f"slice_selection={selection} requires a mask")
        area = np.asarray(roi_area)
        if area.shape[0] != n_slices:
            raise ValueError(f"mask has {area.shape[0]} slices, image has {n_slices}")
        positive = np.flatnonzero(area > 0)
        if selection == "mask_all":
            return positive.tolist()
        order = sorted(positive.tolist(), key=lambda i: (-area[i], i))
        return sorted(order[:count])
    raise ValueError(f"unknown slice_selection {selection!r}")


def extract_slice(volume: np.ndarray, axis: int, index: int, *, channel_first: bool = True) -> np.ndarray:
    """``[C, *S]`` -> ``[C, H, W]`` (or ``[*S]`` -> ``[H, W]`` when ``channel_first=False``)."""
    offset = 1 if channel_first else 0
    return np.ascontiguousarray(np.take(np.asarray(volume), int(index), axis=axis + offset))
