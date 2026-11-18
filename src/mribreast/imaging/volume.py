"""Deterministic volume operations (pure functions; no I/O except ``prepare_*`` reading sources).

Legacy order of operations (``NiftiDataset``):

* ``subtraction_mode=none`` (``_monai_preprocess_and_save``): load -> z-score of the
  whole volume (MONAI ``NormalizeIntensityd(nonzero=False, channel_wise=True)``:
  population std, **no epsilon**, std==0 -> divide by 1) -> trilinear resize
  (``align_corners=False``).
* subtraction (``_load_or_create_subtraction``): both raw phases must share the
  original shape and affine -> resize each raw phase -> ``minuend - subtrahend``
  -> z-score with population std and ``+1e-5``.
* Studies with fewer phases than channels get zero channels appended at the end
  (``pad_collate``); spatial zero padding is appended at the high end of each axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mribreast.imaging.geometry import load_series, resized_affine, same_grid

SUBTRACTION_MODES = ("none", "post_minus_pre", "consecutive")
SUBTRACTION_EPS = 1e-5

NORMALISATION_PHASE = "zscore_population_std_zero_to_one"      # MONAI NormalizeIntensity semantics
NORMALISATION_DIFFERENCE = "zscore_population_std_eps_1e-05"    # legacy subtraction z-score
RESIZE_BACKEND = "torch_trilinear"


def input_channels(subtraction_mode: str, phase_count: int) -> int:
    """Legacy ``get_input_channels``: ``phase_count`` for none, ``phase_count - 1`` for subtraction."""
    if subtraction_mode not in SUBTRACTION_MODES:
        raise ValueError(f"unknown subtraction mode {subtraction_mode!r}")
    if subtraction_mode == "none":
        return int(phase_count)
    if phase_count < 2:
        raise ValueError("subtraction requires at least two series")
    return int(phase_count) - 1


def subtraction_pairs(series_uids: Sequence[str], mode: str) -> list[tuple[str, str]]:
    """Ordered ``(minuend, subtrahend)`` pairs (legacy ``build_subtraction_pairs``)."""
    if mode not in SUBTRACTION_MODES:
        raise ValueError(f"unknown subtraction mode {mode!r}")
    uids = list(series_uids)
    if mode == "none":
        return []
    if len(uids) < 2:
        raise ValueError(f"subtraction mode {mode!r} requires at least two series; received {len(uids)}")
    if mode == "post_minus_pre":
        return [(post, uids[0]) for post in uids[1:]]
    return list(zip(uids[1:], uids[:-1]))


def zscore_phase(volume: np.ndarray) -> np.ndarray:
    """MONAI ``NormalizeIntensity`` on one channel: ``(x - mean) / std`` (ddof=0), std==0 -> 1."""
    tensor = torch.as_tensor(np.asarray(volume, dtype=np.float32))
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    if float(std) == 0.0:
        std = torch.tensor(1.0)
    return ((tensor - mean) / std).numpy()


def zscore_difference(volume: np.ndarray, eps: float = SUBTRACTION_EPS) -> np.ndarray:
    """Legacy subtraction z-score: ``(x - mean) / (std_ddof0 + eps)``."""
    tensor = torch.as_tensor(np.asarray(volume, dtype=np.float32))
    return ((tensor - tensor.mean()) / (tensor.std(unbiased=False) + eps)).numpy()


def resize(volume: np.ndarray, size: Sequence[int] | None, *, mode: str = "linear") -> np.ndarray:
    """Resize ``[*S]`` or ``[C, *S]`` (2D/3D spatial) with ``align_corners=False``.

    ``mode="linear"`` -> bilinear/trilinear; ``mode="nearest"`` -> ``nearest-exact`` (masks).
    A same-size resize returns the input unchanged.
    """
    array = np.asarray(volume)
    if size is None:
        return array
    size = tuple(int(s) for s in size)
    spatial_dims = len(size)
    has_channel = array.ndim == spatial_dims + 1
    if array.ndim not in (spatial_dims, spatial_dims + 1):
        raise ValueError(f"cannot resize array of shape {array.shape} to {size}")
    if tuple(array.shape[-spatial_dims:]) == size:
        return array
    tensor = torch.as_tensor(np.ascontiguousarray(array, dtype=np.float32))
    tensor = tensor[None] if has_channel else tensor[None, None]
    if mode == "nearest":
        out = F.interpolate(tensor, size=size, mode="nearest-exact")
    else:
        interp = {2: "bilinear", 3: "trilinear"}[spatial_dims]
        out = F.interpolate(tensor, size=size, mode=interp, align_corners=False)
    out = out[0] if has_channel else out[0, 0]
    result = out.numpy()
    if mode == "nearest" and np.issubdtype(array.dtype, np.integer):
        result = np.rint(result).astype(array.dtype)
    return result


def pad_high(volume: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Zero-pad the trailing spatial axes at their high end (legacy ``pad_collate``)."""
    array = np.asarray(volume)
    spatial = len(shape)
    current = array.shape[-spatial:]
    if any(c > s for c, s in zip(current, shape)):
        raise ValueError(f"cannot pad {current} down to {tuple(shape)}")
    widths = [(0, 0)] * (array.ndim - spatial) + [(0, int(s) - int(c)) for c, s in zip(current, shape)]
    return np.pad(array, widths, mode="constant", constant_values=0)


def pad_channels(volume: np.ndarray, channels: int) -> np.ndarray:
    """Append zero channels up to ``channels`` (legacy ``pad_collate`` series padding)."""
    if volume.shape[0] > channels:
        raise ValueError(f"volume contains {volume.shape[0]} channels, more than {channels}")
    if volume.shape[0] == channels:
        return volume
    pad = np.zeros((channels - volume.shape[0],) + volume.shape[1:], dtype=volume.dtype)
    return np.concatenate([volume, pad], axis=0)


@dataclass(frozen=True)
class PreparedVolume:
    data: np.ndarray        # float32 [*S]
    affine: np.ndarray      # 4x4, consistent with data's grid
    source_shape: tuple[int, ...]
    source_affine: np.ndarray


def prepare_phase(path: str, target_size: Sequence[int] | None) -> PreparedVolume:
    """Legacy ``none`` path for one series: z-score BEFORE resize."""
    data, affine = load_series(path)
    normalised = zscore_phase(data)
    out = resize(normalised, target_size)
    new_affine = resized_affine(affine, data.shape, out.shape) if target_size is not None else affine
    return PreparedVolume(np.ascontiguousarray(out, dtype=np.float32), new_affine, tuple(data.shape), affine)


class GeometryMismatchError(ValueError):
    """Phases cannot be subtracted because their original grids differ."""


def prepare_difference(minuend_path: str, subtrahend_path: str, target_size: Sequence[int] | None,
                       *, minuend_uid: str = "", subtrahend_uid: str = "") -> PreparedVolume:
    """Legacy subtraction: geometry check -> resize raw phases -> difference -> z-score(+eps)."""
    minuend, minuend_affine = load_series(minuend_path)
    subtrahend, subtrahend_affine = load_series(subtrahend_path)
    if not same_grid(minuend.shape, minuend_affine, subtrahend.shape, subtrahend_affine):
        raise GeometryMismatchError(
            f"cannot subtract MRI series with different geometry: {minuend_uid or minuend_path} has shape "
            f"{minuend.shape}, {subtrahend_uid or subtrahend_path} has shape {subtrahend.shape}; "
            f"affines_match={np.allclose(minuend_affine, subtrahend_affine, rtol=1e-4, atol=1e-3)}")
    a = resize(minuend, target_size)
    b = resize(subtrahend, target_size)
    difference = zscore_difference(a - b)
    new_affine = resized_affine(minuend_affine, minuend.shape, a.shape) if target_size is not None else minuend_affine
    return PreparedVolume(np.ascontiguousarray(difference, dtype=np.float32), new_affine,
                          tuple(minuend.shape), minuend_affine)
