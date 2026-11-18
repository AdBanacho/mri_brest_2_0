"""Deterministic synthetic mask provider (tests / CPU acceptance; no real data)."""

from __future__ import annotations

import hashlib
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mribreast.contracts import Geometry, ImageKey, MaskRecord
from mribreast.imaging.geometry import (SeriesNotFoundError, geometry_from_affine, read_header_geometry,
                                        series_path, slice_affine)
from mribreast.imaging.manifest import read_image_file, read_image_manifest
from mribreast.imaging.slices import extract_slice


def ellipsoid_mask(shape: tuple[int, ...], seed_parts: list[int], fraction: float = 0.25) -> np.ndarray:
    """Deterministic non-empty ellipsoid (label 1) inside ``shape``."""
    rng = np.random.default_rng(seed_parts)
    radii = [max(1.0, s * fraction / 2.0) for s in shape]
    centre = [rng.uniform(r, max(r, s - r)) for r, s in zip(radii, shape)]
    grids = np.meshgrid(*[np.arange(s) + 0.5 for s in shape], indexing="ij")
    distance = sum(((g - c) / r) ** 2 for g, c, r in zip(grids, centre, radii))
    mask = (distance <= 1.0).astype(np.int64)
    if not mask.any():
        mask[tuple(min(int(c), s - 1) for c, s in zip(centre, shape))] = 1
    return mask


class SyntheticMaskProvider:
    """Generates ellipsoid masks on the grid of the referenced image.

    Grid lookup: a NIfTI series ``<image_root>/<series_id>.nii[.gz]`` (whole volume
    for ``slice_id == ""``, otherwise the slice along ``slice_axis`` of the volume
    mask) or the ready 2D image for the key in ``image_manifest_file``.
    """

    name = "synthetic"

    def __init__(self, seed: int = 0, *, image_root: str | os.PathLike | None = None,
                 image_manifest_file: str | os.PathLike | None = None, slice_axis: int = 2,
                 fraction: float = 0.25, missing_keys: frozenset[tuple[str, str, str, str]] = frozenset(),
                 provenance: str = "frozen_external"):
        self.seed = int(seed)
        self.image_root = Path(image_root) if image_root else None
        self.image_manifest_file = Path(image_manifest_file) if image_manifest_file else None
        self.slice_axis = int(slice_axis)
        self.fraction = float(fraction)
        self.missing_keys = frozenset(missing_keys)
        self.provenance = provenance
        self._manifest: pd.DataFrame | None = None

    def with_image_sources(self, *, image_root: Any = None, image_manifest_file: Any = None,
                           slice_axis: int | None = None) -> "SyntheticMaskProvider":
        """Copy bound to image sources (used by ``ImageBranch`` when the CLI passed only a seed)."""
        return SyntheticMaskProvider(
            self.seed, image_root=image_root or self.image_root,
            image_manifest_file=image_manifest_file or self.image_manifest_file,
            slice_axis=self.slice_axis if slice_axis is None else slice_axis, fraction=self.fraction,
            missing_keys=self.missing_keys, provenance=self.provenance)

    def _seed_parts(self, key: ImageKey) -> list[int]:
        return [self.seed, zlib.crc32("|".join(key.as_tuple()[:3]).encode())]

    def get(self, key: ImageKey, fold: int | None = None) -> MaskRecord | None:
        if key.as_tuple() in self.missing_keys:
            return None
        volume_key = ImageKey(key.patient_id, key.study_id, key.series_id, "")
        if self.image_root is not None:
            try:
                header = read_header_geometry(series_path(self.image_root, key.series_id))
            except (SeriesNotFoundError, ValueError):
                header = None
            if header is not None:
                volume = ellipsoid_mask(tuple(header.shape), self._seed_parts(volume_key), self.fraction)
                affine = np.asarray(header.affine)
                if key.slice_id == "":
                    return self._record(key, volume, geometry_from_affine(volume.shape, affine))
                index = int(key.slice_id)
                array = extract_slice(volume, self.slice_axis, index, channel_first=False)
                return self._record(key, array, geometry_from_affine(array.shape, slice_affine(affine, self.slice_axis,
                                                                                             index)))
        if self.image_manifest_file is not None:
            if self._manifest is None:
                self._manifest = read_image_manifest(self.image_manifest_file)
            frame = self._manifest
            rows = frame[(frame["patient_id"] == key.patient_id) & (frame["study_id"] == key.study_id)
                         & (frame["series_id"] == key.series_id) & (frame["slice_id"] == key.slice_id)]
            if len(rows):
                shape = read_image_file(rows.iloc[0]["path"]).shape[1:]
                return self._record(key, ellipsoid_mask(tuple(shape), self._seed_parts(key), self.fraction),
                                    Geometry(shape=tuple(shape)))
        return None

    def _record(self, key: ImageKey, array: np.ndarray, geometry: Geometry) -> MaskRecord:
        digest = hashlib.sha256(np.ascontiguousarray(array, dtype=np.int64).tobytes()).hexdigest()
        return MaskRecord(key=key, array=array, geometry=geometry, class_codes={0: "background", 1: "lesion"},
                          source="synthetic", model_id="synthetic-ellipsoid", model_version="1",
                          fingerprint="sha256:" + digest, grid_source=key.series_id,
                          training_provenance=self.provenance, known_overlap="none (synthetic)")

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "seed": self.seed, "fraction": self.fraction, "slice_axis": self.slice_axis,
                "model_id": "synthetic-ellipsoid", "model_version": "1", "training_provenance": self.provenance,
                "known_overlap": "none (synthetic)", "unknown_provenance": False,
                "n_missing_keys": len(self.missing_keys), "limitations": []}


__all__ = ["SyntheticMaskProvider", "ellipsoid_mask"]
