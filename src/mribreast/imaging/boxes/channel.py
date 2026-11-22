"""``box_mode=box_channel``: append one binary channel = the box mapped onto the ``target_size`` grid.

The box is mapped with :func:`mribreast.imaging.annotations.resize_box` (start floor, exclusive end ceil:
rounds outward so no lesion voxel is lost). Fallback (``ctx.box is None``): an all-zero channel.
Output channels = base channels + 1.
"""

from __future__ import annotations

import numpy as np

from mribreast.imaging.annotations import resize_box
from mribreast.imaging.boxes.common import BoxInput
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import TransformLog


def build_box_channel(ctx: BoxInput, settings: ImagingSettings, log: TransformLog | None = None) -> np.ndarray:
    """Return ``[C + 1, *settings.target_size]`` float32.

    The first ``C`` channels are ``ctx.target`` unchanged (only cast to float32); the last channel is 1
    inside ``resize_box(ctx.box, ctx.source_shape, target_size)`` and 0 elsewhere (all zeros for the
    ``full_image`` fallback). ``box_margin`` does not apply and the native volume is never loaded.
    """
    target_size = tuple(int(v) for v in settings.target_size)
    target = np.asarray(ctx.target)
    if target.ndim != len(target_size) + 1 or tuple(target.shape[1:]) != target_size:
        raise ValueError(f"box_channel: target volume has shape {tuple(target.shape)}, expected "
                         f"[C, {', '.join(map(str, target_size))}]")
    channel = np.zeros((1,) + target_size, dtype=np.float32)
    if ctx.box is None:
        if log is not None:
            log.append({"op": "box_channel", "fallback": "zeros"})
    else:
        if ctx.source_shape is None:
            raise ValueError(f"box_channel: study {ctx.study_id!r} has a box but no source_shape")
        source_shape = tuple(int(v) for v in ctx.source_shape)
        starts, ends = ctx.box
        if not (len(starts) == len(ends) == len(source_shape) == len(target_size)):
            raise ValueError(f"box_channel: box {ctx.box} does not match grid {source_shape}")
        for lo, hi, size in zip(starts, ends, source_shape):
            if not 0 <= int(lo) < int(hi) <= int(size):
                raise ValueError(f"box_channel: box {ctx.box} is empty or outside grid {source_shape} "
                                 f"(study {ctx.study_id!r})")
        t_starts, t_ends = resize_box(ctx.box, source_shape, target_size)
        channel[(0,) + tuple(slice(lo, hi) for lo, hi in zip(t_starts, t_ends))] = 1.0
        if log is not None:
            log.append({"op": "box_channel", "box_native": [list(map(int, starts)), list(map(int, ends))],
                        "source_shape": list(source_shape), "box_target": [list(t_starts), list(t_ends)],
                        "target_size": list(target_size)})
    out = np.concatenate([target.astype(np.float32, copy=False), channel], axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


__all__ = ["build_box_channel"]
