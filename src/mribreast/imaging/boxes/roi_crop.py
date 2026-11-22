"""``box_mode=roi_crop``: crop box + ``box_margin`` on the NATIVE grid, resize to ``target_size``; the only image.

The crop region is :func:`~mribreast.imaging.boxes.common.expand_box` (``ceil(box_margin * extent)`` voxels
per side, clipped to the volume), taken on the native array so the box is never resampled before
cropping; the crop is then resized with :func:`~mribreast.imaging.transforms.joint_resize`
(trilinear, align_corners=False), as in the mask ``roi_crop`` analogue.

Fallback (``box_missing_policy=full_image``, ``ctx.box is None``): the full volume on the target grid,
i.e. ``ctx.target`` unchanged (identical to the image-only input; the native grid is not loaded).
"""

from __future__ import annotations

import numpy as np

from mribreast.imaging.boxes.common import BoxInput, expand_box
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import TransformLog, joint_crop, joint_resize


def build_roi_crop(ctx: BoxInput, settings: ImagingSettings, log: TransformLog | None = None) -> np.ndarray:
    """Return ``[C, *settings.target_size]`` float32."""
    if ctx.box is None:
        if log is not None:
            log.append({"op": "box_roi_crop", "fallback": "full_image"})
        return np.ascontiguousarray(ctx.target, dtype=np.float32)
    native = ctx.native()
    spatial = tuple(int(v) for v in native.shape[1:])
    if ctx.source_shape is None or spatial != tuple(int(v) for v in ctx.source_shape):
        raise ValueError(f"study {ctx.study_id}: native volume grid {spatial} != annotation source_shape "
                         f"{ctx.source_shape}; the box would be applied on a different grid")
    margin = float(settings.box_margin)
    starts, ends = expand_box(ctx.box, spatial, margin)
    if log is not None:
        log.append({"op": "box_roi_crop", "box": [list(ctx.box[0]), list(ctx.box[1])], "crop": [starts, ends],
                    "margin": margin, "source_shape": list(spatial)})
    image, _ = joint_crop(native, None, starts, ends, log)
    image, _ = joint_resize(image, None, settings.target_size, log)
    return np.ascontiguousarray(image, dtype=np.float32)


__all__ = ["build_roi_crop"]
