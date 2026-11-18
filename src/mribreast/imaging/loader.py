"""Study-level volume assembly: series UIDs -> channel-first array ``[C, *target]``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mribreast.imaging.cache import CACHE_SCHEMA_VERSION, VolumeCache, cache_key_payload
from mribreast.imaging.geometry import read_header_geometry, resized_affine, series_path
from mribreast.imaging.volume import (NORMALISATION_DIFFERENCE, NORMALISATION_PHASE, RESIZE_BACKEND, pad_channels,
                                      prepare_difference, prepare_phase, subtraction_pairs)


@dataclass(frozen=True)
class StudyVolume:
    data: np.ndarray                    # float32 [C, *S]
    affine: np.ndarray                  # grid of the study's first series (reference/mask grid)
    reference_uid: str                  # series whose grid ``affine`` describes
    channel_sources: tuple[str, ...]    # e.g. ("uid0", "uid1") or ("uid1-uid0", ...)
    cache_keys: tuple[str, ...] = field(default=())


class StudyVolumeLoader:
    """Loads one study's phases (or subtraction images) through the deterministic cache.

    ``target_size=None`` keeps the original grid of the first series (used for 2D
    slicing); phases with another grid are then resized to it (after their z-score).
    """

    def __init__(self, image_root: str | os.PathLike, *, subtraction_mode: str, channels: int,
                 target_size: Sequence[int] | None, prepared_root: str | os.PathLike | None = None,
                 subtraction_root: str | os.PathLike | None = None):
        self.image_root = Path(image_root)
        self.subtraction_mode = subtraction_mode
        self.channels = int(channels)
        self.target_size = tuple(int(s) for s in target_size) if target_size is not None else None
        self.phase_cache = VolumeCache(prepared_root)
        self.difference_cache = VolumeCache(subtraction_root)

    def describe(self) -> dict[str, Any]:
        return {"cache_schema_version": CACHE_SCHEMA_VERSION, "subtraction_mode": self.subtraction_mode,
                "target_size": list(self.target_size) if self.target_size else None,
                "normalisation": NORMALISATION_PHASE if self.subtraction_mode == "none" else NORMALISATION_DIFFERENCE,
                "resize_backend": RESIZE_BACKEND, "align_corners": False, "channels": self.channels,
                "channel_padding": "zeros appended (legacy pad_collate)",
                "axis_order": "nibabel voxel axes (i, j, k), no reorientation"}

    def _target_for(self, series_uids: Sequence[str]) -> tuple[int, ...]:
        if self.target_size is not None:
            return self.target_size
        return tuple(read_header_geometry(series_path(self.image_root, series_uids[0])).shape)

    def load(self, series_uids: Sequence[str]) -> StudyVolume:
        uids = [str(u) for u in series_uids]
        if not uids:
            raise ValueError("study has no series UIDs")
        target = self._target_for(uids)
        params = {"target_size": target, "resize_backend": RESIZE_BACKEND, "align_corners": False,
                  "dtype": "float32"}
        arrays: list[np.ndarray] = []
        keys: list[str] = []
        affine = None
        if self.subtraction_mode == "none":
            sources = tuple(uids)
            for uid in uids:
                path = series_path(self.image_root, uid)
                payload = cache_key_payload("phase", [(uid, path)], {**params, "normalisation": NORMALISATION_PHASE})

                def build(path: Path = path) -> tuple[np.ndarray, np.ndarray]:
                    prepared = prepare_phase(str(path), target)
                    return prepared.data, prepared.affine

                result = self.phase_cache.get_or_build("phase", payload, build, source=str(path))
                arrays.append(result.data)
                keys.append(result.key)
                affine = result.affine if affine is None else affine
        else:
            pairs = subtraction_pairs(uids, self.subtraction_mode)
            sources = tuple(f"{a}-{b}" for a, b in pairs)
            for minuend, subtrahend in pairs:
                path_a = series_path(self.image_root, minuend)
                path_b = series_path(self.image_root, subtrahend)
                payload = cache_key_payload("difference", [(minuend, path_a), (subtrahend, path_b)],
                                            {**params, "normalisation": NORMALISATION_DIFFERENCE,
                                             "subtraction": "minuend_minus_subtrahend"})

                def build(a: Path = path_a, b: Path = path_b, ua: str = minuend,
                          ub: str = subtrahend) -> tuple[np.ndarray, np.ndarray]:
                    prepared = prepare_difference(str(a), str(b), target, minuend_uid=ua, subtrahend_uid=ub)
                    return prepared.data, prepared.affine

                result = self.difference_cache.get_or_build("difference", payload, build,
                                                            source=f"{path_a} - {path_b}")
                arrays.append(result.data)
                keys.append(result.key)
                affine = result.affine if affine is None else affine
        data = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        if self.subtraction_mode != "none":
            # Reference grid of a study = its FIRST series (the mask grid). Subtraction phases equal it within the
            # legacy same_grid tolerance; report the first series' grid, not the minuend's, for consistency.
            reference = read_header_geometry(series_path(self.image_root, uids[0]))
            affine = resized_affine(np.asarray(reference.affine), reference.shape, data.shape[1:])
        data = pad_channels(data, self.channels)
        return StudyVolume(data, np.asarray(affine), uids[0], sources, tuple(keys))
