"""Ready 2D images: manifest reader and PNG/TIFF loading as ``[C, H, W]`` float32."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from mribreast.contracts import IMAGE_KEY_COLUMNS, PATIENT_ID, SERIES_ID, SLICE_ID, STUDY_ID

MANIFEST_REQUIRED = (*IMAGE_KEY_COLUMNS, "path")
CHANNEL_COLUMN = "channel"
IMAGE_SUFFIXES = (".png", ".tif", ".tiff")


def _normalise_ids(frame: pd.DataFrame) -> pd.DataFrame:
    for column in IMAGE_KEY_COLUMNS:
        frame[column] = frame[column].map(lambda v: "" if pd.isna(v) else str(v).strip())
    return frame


def read_image_manifest(path: str | os.PathLike) -> pd.DataFrame:
    """Read a 2D image manifest CSV.

    Required columns: ``patient_id, study_id, series_id, slice_id, path``;
    optional ``channel`` (int, rows sharing a key are stacked in channel order;
    alias ``phase``). Relative paths resolve against the manifest directory.
    Key columns are strings; the returned frame keeps any extra columns.
    """
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "phase" in frame.columns and CHANNEL_COLUMN not in frame.columns:
        frame = frame.rename(columns={"phase": CHANNEL_COLUMN})
    missing = [c for c in MANIFEST_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: image manifest lacks columns {missing}")
    frame = _normalise_ids(frame)
    if (frame[[PATIENT_ID, STUDY_ID, SERIES_ID, SLICE_ID]] == "").any().any():
        raise ValueError(f"{path}: empty key values in image manifest")
    if CHANNEL_COLUMN in frame.columns:
        frame[CHANNEL_COLUMN] = frame[CHANNEL_COLUMN].replace("", "0").astype(int)
    else:
        frame[CHANNEL_COLUMN] = 0
    duplicated = frame.duplicated(list(IMAGE_KEY_COLUMNS) + [CHANNEL_COLUMN])
    if duplicated.any():
        raise ValueError(f"{path}: duplicate image keys {frame.loc[duplicated, list(IMAGE_KEY_COLUMNS)].values[:3]}")
    base = path.parent
    frame["path"] = [str(p if Path(p).is_absolute() else (base / p)) for p in frame["path"]]
    bad = [p for p in frame["path"] if Path(p).suffix.lower() not in IMAGE_SUFFIXES]
    if bad:
        raise ValueError(f"{path}: unsupported image formats {bad[:3]} (PNG/TIFF only)")
    patients = frame.groupby(STUDY_ID)[PATIENT_ID].nunique()
    if (patients > 1).any():
        raise ValueError(f"{path}: studies mapped to several patients: {list(patients[patients > 1].index[:3])}")
    return frame.reset_index(drop=True)


def read_image_file(path: str | os.PathLike) -> np.ndarray:
    """Read PNG/TIFF into ``[C, H, W]`` float32 (grayscale -> C=1; RGB(A) -> 3/4 channels)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image)
    elif suffix in (".tif", ".tiff"):
        import tifffile

        array = np.asarray(tifffile.imread(str(path)))
    else:
        raise ValueError(f"unsupported image format: {path}")
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3 and suffix == ".png":
        array = np.moveaxis(array, -1, 0)                       # PIL is always H, W, C
    elif array.ndim == 3 and array.shape[-1] in (3, 4) and array.shape[0] not in (3, 4):
        array = np.moveaxis(array, -1, 0)                       # TIFF: H, W, C unless already C, H, W
    elif array.ndim != 3:
        raise ValueError(f"{path}: expected 2D image, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def load_manifest_image(rows: pd.DataFrame) -> np.ndarray:
    """Stack the files of one image key (sorted by ``channel``) into ``[C, H, W]``."""
    parts = [read_image_file(p) for p in rows.sort_values(CHANNEL_COLUMN)["path"]]
    shapes = {p.shape[1:] for p in parts}
    if len(shapes) != 1:
        raise ValueError(f"channels of one image have different sizes: {shapes}")
    return np.concatenate(parts, axis=0)


def zscore_channels(image: np.ndarray) -> np.ndarray:
    """Per-channel z-score (population std; std==0 -> 1), as for 3D phases."""
    image = np.asarray(image, dtype=np.float32)
    mean = image.reshape(image.shape[0], -1).mean(axis=1)
    std = image.reshape(image.shape[0], -1).std(axis=1)
    std[std == 0] = 1.0
    shape = (-1,) + (1,) * (image.ndim - 1)
    return ((image - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32)
