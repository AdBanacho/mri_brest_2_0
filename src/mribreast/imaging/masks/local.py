"""Local-file mask provider: 2D PNG/TIFF masks and NIfTI masks (whole volume or extracted slices)."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd

from mribreast.contracts import IMAGE_KEY_COLUMNS, SERIES_ID, ImageKey, MaskRecord
from mribreast.imaging.geometry import geometry_from_affine, slice_affine
from mribreast.imaging.manifest import read_image_file
from mribreast.imaging.slices import extract_slice

MASK_MANIFEST_REQUIRED = (*IMAGE_KEY_COLUMNS, "path")
MASK_MANIFEST_OPTIONAL = {
    "class_codes": "", "source": "manual", "model_id": "", "model_version": "", "grid_source": "",
    "training_provenance": "unknown", "known_overlap": "", "fold": "",
}
DEFAULT_CLASS_CODES = {0: "background", 1: "lesion"}
NIFTI_MASK_SUFFIXES = (".nii", ".nii.gz")


def _is_nifti(path: str) -> bool:
    return str(path).lower().endswith(NIFTI_MASK_SUFFIXES)


def parse_class_codes(text: str) -> dict[int, str]:
    if not str(text).strip():
        return dict(DEFAULT_CLASS_CODES)
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"class_codes must be a JSON object, got {text!r}")
    return {int(k): str(v) for k, v in raw.items()}


def read_mask_manifest(path: str | os.PathLike) -> pd.DataFrame:
    """Mask manifest: key columns + ``path`` (+ optional provenance columns, see module constants).

    ``slice_id`` empty = whole-volume (NIfTI) mask; ``fold`` empty = fold-agnostic
    (frozen/manual) mask, otherwise a fold-trained segmenter output for that outer fold.
    """
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in MASK_MANIFEST_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: mask manifest lacks columns {missing}")
    for column, default in MASK_MANIFEST_OPTIONAL.items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].map(lambda v: str(v).strip())
    for column in IMAGE_KEY_COLUMNS:
        frame[column] = frame[column].map(lambda v: str(v).strip())
    frame.loc[frame["source"] == "", "source"] = "manual"
    frame.loc[frame["training_provenance"] == "", "training_provenance"] = "unknown"
    dup = frame.duplicated(list(IMAGE_KEY_COLUMNS) + ["fold"])
    if dup.any():
        raise ValueError(f"{path}: duplicate mask keys (per fold): {frame.loc[dup, list(IMAGE_KEY_COLUMNS)].values[:3]}")
    base = path.parent
    frame["path"] = [str(p if Path(p).is_absolute() else (base / p)) for p in frame["path"]]
    for text in frame["class_codes"].unique():
        parse_class_codes(text)
    return frame.reset_index(drop=True)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


class LocalFileMaskProvider:
    """``MaskProvider`` over a CSV manifest of local mask files (no network access)."""

    name = "local_files"

    def __init__(self, manifest_path: str | os.PathLike, *, slice_axis: int = 2, source: str | None = None):
        self.manifest_path = Path(manifest_path)
        self.slice_axis = int(slice_axis)
        frame = read_mask_manifest(self.manifest_path)
        if source is not None:
            frame = frame[frame["source"] == source].reset_index(drop=True)
        self.frame = frame
        self._index: dict[tuple[str, str, str, str], list[int]] = {}
        for position, key in enumerate(frame[list(IMAGE_KEY_COLUMNS)].itertuples(index=False, name=None)):
            self._index.setdefault(tuple(key), []).append(position)

    # ------------------------------------------------------------------ lookup
    def _row_for(self, key_tuple: tuple[str, str, str, str], fold: int | None) -> pd.Series | None:
        positions = self._index.get(key_tuple, [])
        if not positions:
            return None
        rows = self.frame.iloc[positions]
        if fold is not None:
            fold_rows = rows[rows["fold"] == str(fold)]
            if len(fold_rows):
                return fold_rows.iloc[0]
        agnostic = rows[rows["fold"] == ""]
        return agnostic.iloc[0] if len(agnostic) else None

    def get(self, key: ImageKey, fold: int | None = None) -> MaskRecord | None:
        key_tuple = tuple(str(v) for v in key.as_tuple())
        row = self._row_for(key_tuple, fold)  # type: ignore[arg-type]
        extract_index: int | None = None
        if row is None and key.slice_id != "":
            row = self._row_for((key.patient_id, key.study_id, key.series_id, ""), fold)
            if row is None or not _is_nifti(row["path"]):
                return None
            extract_index = int(key.slice_id)
        if row is None or not Path(row["path"]).is_file():
            return None
        return self._record(key, row, extract_index)

    def _record(self, key: ImageKey, row: pd.Series, extract_index: int | None) -> MaskRecord:
        path = row["path"]
        if _is_nifti(path):
            image = nib.load(path)
            array = np.rint(np.asarray(image.get_fdata(dtype=np.float32)))
            while array.ndim > 3 and array.shape[-1] == 1:
                array = array[..., 0]
            affine = np.asarray(image.affine, dtype=np.float64)
            if extract_index is not None:
                array = extract_slice(array, self.slice_axis, extract_index, channel_first=False)
                affine = slice_affine(affine, self.slice_axis, extract_index)
            geometry = geometry_from_affine(array.shape, affine)
        else:
            array = read_image_file(path)
            if array.shape[0] != 1:
                raise ValueError(f"{path}: mask must be single-channel, got {array.shape[0]} channels")
            array = np.rint(array[0])
            geometry = geometry_from_affine(array.shape, None)
        provenance = row["training_provenance"]
        if row["fold"] and provenance == "unknown":
            provenance = f"fold_trained:{row['fold']}"
        return MaskRecord(
            key=key, array=array.astype(np.int64), geometry=geometry,
            class_codes=parse_class_codes(row["class_codes"]), source=row["source"],
            model_id=row["model_id"], model_version=row["model_version"], fingerprint=_file_sha256(path),
            grid_source=row["grid_source"] or row[SERIES_ID], training_provenance=provenance,
            known_overlap=row["known_overlap"])

    def describe(self) -> dict[str, Any]:
        frame = self.frame
        unknown = int((frame["training_provenance"] == "unknown").sum() - ((frame["training_provenance"] == "unknown")
                                                                          & (frame["fold"] != "")).sum())
        limitations = []
        if unknown:
            limitations.append(f"{unknown} masks have unknown segmenter training provenance; results using them "
                               "carry an unquantified leakage risk")
        if (frame["class_codes"] == "").any():
            limitations.append("some masks declare no class_codes; default {0: background, 1: lesion} assumed")
        return {
            "provider": self.name, "manifest": str(self.manifest_path),
            "manifest_fingerprint": _file_sha256(str(self.manifest_path)), "n_masks": int(len(frame)),
            "sources": dict(Counter(frame["source"])), "model_ids": sorted(set(frame["model_id"]) - {""}),
            "model_versions": sorted(set(frame["model_version"]) - {""}),
            "training_provenance": dict(Counter(frame["training_provenance"])),
            "known_overlap": sorted(set(frame["known_overlap"]) - {""}),
            "folds": sorted(set(frame["fold"]) - {""}), "slice_axis": self.slice_axis,
            "unknown_provenance": bool(unknown), "limitations": limitations,
        }


def mask_manifest_frame(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
    """Helper for writers: complete rows with the optional columns in canonical order."""
    columns = list(MASK_MANIFEST_REQUIRED) + list(MASK_MANIFEST_OPTIONAL)
    frame = pd.DataFrame(rows)
    for column, default in MASK_MANIFEST_OPTIONAL.items():
        if column not in frame.columns:
            frame[column] = default
    return frame.loc[:, columns + [c for c in frame.columns if c not in columns]]


__all__ = ["LocalFileMaskProvider", "read_mask_manifest", "parse_class_codes", "mask_manifest_frame",
           "DEFAULT_CLASS_CODES"]
