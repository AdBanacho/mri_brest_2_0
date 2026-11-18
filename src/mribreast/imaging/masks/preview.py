"""PNG preview of an image-mask pair plus a JSON record of the applied transforms."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _to_uint8(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float64)
    lo, hi = np.percentile(channel, [1, 99]) if channel.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((channel - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def write_mask_preview(image: np.ndarray, mask: np.ndarray | None, path: str | os.PathLike, *,
                       transforms: Sequence[Mapping[str, Any]] = (), key: Sequence[str] | None = None,
                       metadata: Mapping[str, Any] | None = None, channel: int = 0) -> tuple[Path, Path]:
    """Write ``<path>.png`` (grayscale image | image with red ROI overlay) and ``<path>.json``.

    ``image`` is ``[C, H, W]`` (or ``[H, W]``), ``mask`` ``[H, W]`` on the same grid.
    """
    from PIL import Image

    path = Path(path)
    png = path.with_suffix(".png")
    meta = path.with_suffix(".json")
    png.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    plane = image[channel] if image.ndim == 3 else image
    gray = _to_uint8(plane)
    rgb = np.stack([gray] * 3, axis=-1)
    overlay = rgb.copy()
    if mask is not None:
        if tuple(mask.shape) != tuple(plane.shape):
            raise ValueError(f"mask {mask.shape} and image {plane.shape} grids differ")
        roi = np.asarray(mask) > 0
        overlay[roi] = (0.5 * overlay[roi] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(np.concatenate([rgb, overlay], axis=1)).save(png)
    payload = {"key": list(key) if key is not None else None, "image_shape": list(image.shape),
               "mask_shape": list(mask.shape) if mask is not None else None,
               "mask_voxels": int((np.asarray(mask) > 0).sum()) if mask is not None else 0,
               "transforms": [dict(t) for t in transforms], **dict(metadata or {})}
    meta.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return png, meta
