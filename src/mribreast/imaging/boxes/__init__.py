"""Duke annotation boxes as a 3D classifier input (``box_mode``): policies/audit, shared context, one module per mode.

* :mod:`.policies`  per-study decisions (``box_missing_policy``) and ``box_audit.csv`` rows;
* :mod:`.common`    :class:`BoxInput` (per-item context) and :func:`expand_box` (margin + clipping);
* :mod:`.roi_crop`, :mod:`.channel`, :mod:`.dual`  one builder per mode, dispatched by :func:`build_box_input`.

Box modes use the box at training AND inference (research setting: a radiologist box, or a future
detector, must exist for every test study).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mribreast.imaging.boxes.common import BoxInput, expand_box
from mribreast.imaging.boxes.policies import (BOX_AUDIT_COLUMNS, BOX_MODES, BOX_POLICIES, BoxDecision, BoxPolicyError,
                                              box_audit_frame, coverage_counts, decide_boxes)


def build_box_input(ctx: BoxInput, settings: Any, log: list[dict[str, Any]] | None = None
                    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Dispatch to the builder of ``settings.box_mode`` (array, or ``(global, local)`` for dual_stream)."""
    mode = settings.box_mode
    if mode == "roi_crop":
        from mribreast.imaging.boxes.roi_crop import build_roi_crop

        return build_roi_crop(ctx, settings, log)
    if mode == "box_channel":
        from mribreast.imaging.boxes.channel import build_box_channel

        return build_box_channel(ctx, settings, log)
    if mode == "dual_stream":
        from mribreast.imaging.boxes.dual import build_dual_stream

        return build_dual_stream(ctx, settings, log)
    raise ValueError(f"unknown box_mode {mode!r}; expected one of {BOX_MODES[1:]}")


__all__ = ["BOX_AUDIT_COLUMNS", "BOX_MODES", "BOX_POLICIES", "BoxDecision", "BoxInput", "BoxPolicyError",
           "box_audit_frame", "build_box_input", "coverage_counts", "decide_boxes", "expand_box"]
