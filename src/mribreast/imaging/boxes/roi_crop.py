"""``box_mode=roi_crop``: crop box + ``box_margin`` on the NATIVE grid, resize to ``target_size``; the only image.

Fallback (``box_missing_policy=full_image``, ``ctx.box is None``): the full volume on the target grid.
"""

from __future__ import annotations

import numpy as np

from mribreast.imaging.boxes.common import BoxInput
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import TransformLog


def build_roi_crop(ctx: BoxInput, settings: ImagingSettings, log: TransformLog | None = None) -> np.ndarray:
    """Return ``[C, *settings.target_size]`` float32."""
    raise NotImplementedError("box_mode=roi_crop is not implemented yet")


__all__ = ["build_roi_crop"]
