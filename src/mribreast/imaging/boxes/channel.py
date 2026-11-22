"""``box_mode=box_channel``: append one binary channel = the box mapped onto the ``target_size`` grid.

The box is mapped with :func:`mribreast.imaging.annotations.resize_box` (start floor, exclusive end ceil:
rounds outward so no lesion voxel is lost). Fallback (``ctx.box is None``): an all-zero channel.
Output channels = base channels + 1.
"""

from __future__ import annotations

import numpy as np

from mribreast.imaging.boxes.common import BoxInput
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import TransformLog


def build_box_channel(ctx: BoxInput, settings: ImagingSettings, log: TransformLog | None = None) -> np.ndarray:
    """Return ``[C + 1, *settings.target_size]`` float32."""
    raise NotImplementedError("box_mode=box_channel is not implemented yet")


__all__ = ["build_box_channel"]
