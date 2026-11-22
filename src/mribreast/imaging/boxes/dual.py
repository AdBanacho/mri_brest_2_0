"""``box_mode=dual_stream``: (global, local) inputs for :class:`mribreast.models.dual_stream.DualStreamNetwork`.

* global = the full volume at ``target_size`` (exactly the image-only input, ``ctx.target``);
* local  = box + ``box_margin`` cropped on the NATIVE grid, resized to ``box_local_size``.

Fallback (``ctx.box is None``): local = the full native volume resized to ``box_local_size``.
"""

from __future__ import annotations

import numpy as np

from mribreast.imaging.boxes.common import BoxInput
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import TransformLog


def build_dual_stream(ctx: BoxInput, settings: ImagingSettings,
                      log: TransformLog | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(global [C, *target_size], local [C, *box_local_size])`` float32."""
    raise NotImplementedError("box_mode=dual_stream is not implemented yet")


__all__ = ["build_dual_stream"]
