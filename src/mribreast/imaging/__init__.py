"""mribreast.imaging: geometry, NIfTI loading, normalisation/resize/subtraction, deterministic cache,
annotation boxes, 2D slicing/manifests, joint image-mask transforms and datasets.

Public names are exported lazily (PEP 562) so importing a light submodule such as
``mribreast.imaging.cache`` does not import torch. Importing performs no I/O.
See docs/imaging.md for conventions.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "annotation_coverage": "annotations", "boxes_from_frame": "annotations", "erase_background_box": "annotations",
    "load_annotation_boxes": "annotations", "resize_box": "annotations", "write_annotation_coverage": "annotations",
    "CACHE_SCHEMA_VERSION": "cache", "VolumeCache": "cache", "cache_key": "cache", "cache_key_payload": "cache",
    "load_series": "geometry", "parse_series_uids": "geometry", "read_header_geometry": "geometry",
    "resized_affine": "geometry", "series_path": "geometry",
    "StudyVolume": "loader", "StudyVolumeLoader": "loader",
    "read_image_file": "manifest", "read_image_manifest": "manifest",
    "ImagingSettings": "settings", "settings_from_cfg": "settings",
    "extract_slice": "slices", "select_slices": "slices",
    "joint_crop": "transforms", "joint_flip": "transforms", "joint_resize": "transforms",
    "input_channels": "volume", "prepare_difference": "volume", "prepare_phase": "volume", "resize": "volume",
    "subtraction_pairs": "volume", "zscore_difference": "volume", "zscore_phase": "volume",
    "ImageDataset": "datasets", "ItemBuilder": "datasets", "aggregate_study_probs": "datasets",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'mribreast.imaging' has no attribute {name!r}")
    return getattr(importlib.import_module(f"mribreast.imaging.{module}"), name)
