"""Pure geometry for box-prompted 2D segmentation of 3D volumes (no torch, no model).

Conventions (docs/imaging.md §1, §4; docs/segmentation-mri-core.md §3, §4.2):

* A Duke box is half-open on the loaded-array grid: ``((a0, a1, a2), (b0, b1, b2))``.
* Slicing along ``slice_axis`` keeps the other two axes in increasing order: ``[H, W]``.
* A 2D box is half-open ``(h0, h1, w0, w1)`` in slice pixels. SAM's prompt is ``(x0, y0, x1, y1)`` with
  ``x`` along W and ``y`` along H, in continuous pixel-edge coordinates of the model frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

Box3D = tuple[tuple[int, int, int], tuple[int, int, int]]
Box2D = tuple[int, int, int, int]   # h0, h1, w0, w1 (half-open)


def plane_axes(slice_axis: int) -> tuple[int, int]:
    """Array axes that become ``(H, W)`` of a slice taken along ``slice_axis``."""
    if slice_axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1 or 2, got {slice_axis}")
    h, w = (a for a in range(3) if a != slice_axis)
    return h, w


def slice_range(box: Box3D, slice_axis: int, n_slices: int, margin: int = 0) -> range:
    """Slices covered by the box along ``slice_axis`` ± ``margin``, clipped to the volume."""
    if margin < 0:
        raise ValueError("slice margin must be >= 0")
    lo, hi = int(box[0][slice_axis]), int(box[1][slice_axis])
    if not 0 <= lo < hi <= n_slices:
        raise ValueError(f"box {box} is empty or outside {n_slices} slices along axis {slice_axis}")
    return range(max(0, lo - margin), min(n_slices, hi + margin))


def box_in_plane(box: Box3D, slice_axis: int) -> Box2D:
    """Half-open ``(h0, h1, w0, w1)`` of a 3D box on the slice plane."""
    h, w = plane_axes(slice_axis)
    return int(box[0][h]), int(box[1][h]), int(box[0][w]), int(box[1][w])


def box_xyxy(box2d: Box2D) -> tuple[float, float, float, float]:
    """SAM order ``(x0, y0, x1, y1)`` = ``(w0, h0, w1, h1)`` for a half-open box (pixel edges)."""
    h0, h1, w0, w1 = box2d
    return float(w0), float(h0), float(w1), float(h1)


def expand_box2d(box2d: Box2D, shape: Sequence[int], margin: float) -> Box2D:
    """Expand by ``ceil(margin * extent)`` per side and clip (same rule as ``imaging.boxes.expand_box``)."""
    h0, h1, w0, w1 = box2d
    ph = int(math.ceil(float(margin) * (h1 - h0)))
    pw = int(math.ceil(float(margin) * (w1 - w0)))
    return max(0, h0 - ph), min(int(shape[0]), h1 + ph), max(0, w0 - pw), min(int(shape[1]), w1 + pw)


@dataclass(frozen=True)
class Window:
    """Square window ``[h0, h0+side) x [w0, w0+side)`` in slice pixels; may extend past the slice."""

    h0: int
    w0: int
    side: int

    def box_in_window(self, box2d: Box2D) -> Box2D:
        h0, h1, w0, w1 = box2d
        return h0 - self.h0, h1 - self.h0, w0 - self.w0, w1 - self.w0


def context_window(box2d: Box2D, shape: Sequence[int], context: float, min_side: int) -> Window:
    """Square window centred on the box, side ``max(ceil(context * max(h, w)), min_side)``.

    The window is shifted to lie inside the slice where possible. It extends past the slice (the caller pads)
    only when ``side`` exceeds the slice size along that axis.
    """
    if context < 1:
        raise ValueError("context must be >= 1 (the window must contain the box)")
    h0, h1, w0, w1 = box2d
    H, W = int(shape[0]), int(shape[1])
    if not (0 <= h0 < h1 <= H and 0 <= w0 < w1 <= W):
        raise ValueError(f"2D box {box2d} is empty or outside slice {H}x{W}")
    side = max(int(math.ceil(context * max(h1 - h0, w1 - w0))), int(min_side), 1)

    def place(lo: int, hi: int, size: int) -> int:
        start = (lo + hi) // 2 - side // 2
        if side <= size:
            start = min(max(start, 0), size - side)
        else:                                   # window larger than the slice: centre the slice in it
            start = -((side - size) // 2)
        return start

    return Window(place(h0, h1, H), place(w0, w1, W), side)


def extract_window(plane: np.ndarray, window: Window, fill: float) -> np.ndarray:
    """``[side, side]`` crop of ``plane``; pixels outside the slice get ``fill``."""
    H, W = plane.shape
    out = np.full((window.side, window.side), fill, dtype=np.float32)
    sh0, sh1 = max(window.h0, 0), min(window.h0 + window.side, H)
    sw0, sw1 = max(window.w0, 0), min(window.w0 + window.side, W)
    out[sh0 - window.h0: sh1 - window.h0, sw0 - window.w0: sw1 - window.w0] = plane[sh0:sh1, sw0:sw1]
    return out


def paste_window(target: np.ndarray, window_mask: np.ndarray, window: Window) -> None:
    """Write the in-slice part of a ``[side, side]`` window mask into ``target`` (in place, OR-combined)."""
    H, W = target.shape
    sh0, sh1 = max(window.h0, 0), min(window.h0 + window.side, H)
    sw0, sw1 = max(window.w0, 0), min(window.w0 + window.side, W)
    part = window_mask[sh0 - window.h0: sh1 - window.h0, sw0 - window.w0: sw1 - window.w0]
    target[sh0:sh1, sw0:sw1] |= part.astype(target.dtype)


def scale_box(box2d_in_window: Box2D, side: int, frame: int) -> tuple[float, float, float, float]:
    """SAM ``(x0, y0, x1, y1)`` of a window-relative box after resizing the window to ``frame x frame``."""
    s = float(frame) / float(side)
    x0, y0, x1, y1 = box_xyxy(box2d_in_window)
    return x0 * s, y0 * s, x1 * s, y1 * s


def clip_to_box(mask: np.ndarray, box2d: Box2D) -> tuple[np.ndarray, int]:
    """Zero ``mask`` outside the half-open box; return the clipped mask and the number of voxels removed."""
    h0, h1, w0, w1 = box2d
    keep = np.zeros_like(mask, dtype=bool)
    keep[h0:h1, w0:w1] = True
    removed = int(np.count_nonzero(mask & ~keep))
    return (mask & keep), removed


__all__ = ["Box2D", "Box3D", "Window", "box_in_plane", "box_xyxy", "clip_to_box", "context_window",
           "expand_box2d", "extract_window", "paste_window", "plane_axes", "scale_box", "slice_range"]
