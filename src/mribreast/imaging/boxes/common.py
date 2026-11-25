"""Shared box-mode inputs: the per-item context handed to each ``box_mode`` builder and box geometry helpers.

Axis convention (inherited from :mod:`mribreast.imaging.annotations`, OPEN ASSUMPTION): a box is the
half-open interval ``((c0, r0, s0), (c1+1, r1+1, s1+1))`` on the loaded-array axes (column, row, slice)
of the study's NATIVE grid (first series, no reorientation). A wrong convention would crop or mark
the wrong region at training AND inference, so it is a stronger risk for box modes than for the
training-only erasing augmentation (see docs/imaging.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from mribreast.imaging.annotations import Box


@dataclass
class BoxInput:
    """Everything a ``box_mode`` builder needs for one 3D item.

    * ``target``: full volume ``[C, *target_size]`` exactly as the image-only path feeds it
      (after training-only annotation erasing, when enabled).
    * ``native``: zero-argument callable returning the full volume on the NATIVE grid ``[C, *source_shape]``
      (loaded lazily; same per-phase normalisation as ``target``, no resize).
    * ``box`` / ``source_shape``: half-open box on the native grid and that grid's shape;
      both ``None`` for the ``full_image`` fallback of ``box_missing_policy``.
    """

    target: np.ndarray
    native: Callable[[], np.ndarray]
    box: Box | None
    source_shape: tuple[int, ...] | None
    study_id: str = ""

    @property
    def fallback(self) -> bool:
        return self.box is None


def expand_box(box: Box, shape: Sequence[int], margin: float) -> tuple[list[int], list[int]]:
    """Half-open box expanded by ``ceil(margin * extent)`` voxels per side, clipped to ``shape``.

    Same rule as :func:`mribreast.imaging.transforms.roi_bbox` (mask ``roi_crop``), so box and mask crops
    with equal margins cover the same region for the same lesion extent.
    """
    starts, ends = box
    if len(starts) != len(shape) or len(ends) != len(shape):
        raise ValueError(f"box {box} does not match grid {tuple(shape)}")
    out_starts, out_ends = [], []
    for lo, hi, size in zip(starts, ends, shape):
        lo, hi, size = int(lo), int(hi), int(size)
        if not 0 <= lo < hi <= size:
            raise ValueError(f"box {box} is empty or outside grid {tuple(shape)}")
        pad = int(np.ceil(float(margin) * (hi - lo)))
        out_starts.append(max(0, lo - pad))
        out_ends.append(min(size, hi + pad))
    return out_starts, out_ends


__all__ = ["BoxInput", "expand_box"]
