"""``box_mode=dual_stream``: (global, local) inputs for :class:`mribreast.models.dual_stream.DualStreamNetwork`.

* global = the full volume at ``target_size`` (exactly the image-only input, ``ctx.target``);
* local  = box + ``box_margin`` cropped on the NATIVE grid, resized to ``box_local_size``.

Fallback (``ctx.box is None``): local = the full native volume resized to ``box_local_size``.
The local stream is never erased (training-only erasing only touches ``ctx.target``).
"""

from __future__ import annotations

import numpy as np

from mribreast.imaging.boxes.common import BoxInput, expand_box
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import TransformLog, joint_crop, joint_resize


def build_dual_stream(ctx: BoxInput, settings: ImagingSettings,
                      log: TransformLog | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(global [C, *target_size], local [C, *box_local_size])`` float32."""
    global_ = np.ascontiguousarray(ctx.target, dtype=np.float32)
    native = np.asarray(ctx.native())
    spatial = tuple(int(n) for n in native.shape[1:])
    size = tuple(int(v) for v in settings.box_local_size)
    if len(size) != len(spatial):
        raise ValueError(f"box_local_size {size} does not match native grid {spatial}")
    if ctx.box is None:
        if log is not None:
            log.append({"op": "dual_stream_local", "fallback": True, "box": None, "crop": None,
                        "margin": float(settings.box_margin), "native_shape": list(spatial), "to": list(size),
                        "note": "no box (box_missing_policy=full_image): local = full native volume"})
        local, _ = joint_resize(native, None, size, log)
    else:
        if ctx.source_shape is None or tuple(int(v) for v in ctx.source_shape) != spatial:
            raise ValueError(f"study {ctx.study_id!r}: native grid {spatial} differs from the box source shape "
                             f"{ctx.source_shape}")
        starts, ends = expand_box(ctx.box, spatial, settings.box_margin)
        if log is not None:
            log.append({"op": "dual_stream_local", "fallback": False,
                        "box": [[int(v) for v in ctx.box[0]], [int(v) for v in ctx.box[1]]],
                        "crop": [starts, ends], "margin": float(settings.box_margin),
                        "native_shape": list(spatial), "to": list(size)})
        local, _ = joint_crop(native, None, starts, ends, log)
        local, _ = joint_resize(local, None, size, log)
    return global_, np.ascontiguousarray(local, dtype=np.float32)


__all__ = ["build_dual_stream"]
