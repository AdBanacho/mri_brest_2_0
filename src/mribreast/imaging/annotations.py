"""Duke annotation boxes: loading, matching audit, resize mapping and training-only erasing.

Convention (verified from legacy ``dataLoaders/annotation_boxes.py`` and its tests):

* Workbook ``Start/End Row|Column|Slice`` are **zero-based and inclusive**.
* Volume axes of the loaded array are ``(column, row, slice)`` (SimpleITK-written
  NIfTI read by nibabel without reorientation), so a box is stored as the
  half-open interval ``starts=(c0, r0, s0)``, ``ends=(c1+1, r1+1, s1+1)``.
* Resize maps starts with ``floor(a * dst / src)`` and exclusive ends with
  ``ceil(b * dst / src)``.
* OPEN ASSUMPTION: the workbook names no series UID, origin or orientation; the
  mapping must be checked visually against real overlays (docs/imaging.md).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.contracts import PATIENT_ID, SERIES_UIDS, STUDY_ID
from mribreast.imaging.geometry import SeriesNotFoundError, parse_series_uids, read_header_geometry, series_path

COLUMNS = ("Patient ID", "Start Row", "End Row", "Start Column", "End Column", "Start Slice", "End Slice")
COVERAGE_COLUMNS = (PATIENT_ID, STUDY_ID, "status", "detail")
Box = tuple[tuple[int, int, int], tuple[int, int, int]]


def boxes_from_frame(data: pd.DataFrame) -> dict[str, Box]:
    """Validate a Duke-format sheet and return ``patient -> ((c0, r0, s0), (c1+1, r1+1, s1+1))``."""
    if tuple(data.columns) != COLUMNS:
        raise ValueError(f"unexpected annotation columns: {list(data.columns)}; expected {list(COLUMNS)}")
    boxes: dict[str, Box] = {}
    for row_number, row in enumerate(data.itertuples(index=False, name=None), start=2):
        patient = row[0]
        if not isinstance(patient, str) or not patient.strip():
            raise ValueError(f"annotation row {row_number}: missing Patient ID")
        patient = patient.strip()
        if patient in boxes:
            raise ValueError(f"duplicate annotation for {patient} (row {row_number})")
        values = row[1:]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
               or not np.isfinite(v) or v < 0 or int(v) != v for v in values):
            raise ValueError(f"annotation row {row_number}: nonnegative integer coordinates required")
        r0, r1, c0, c1, s0, s1 = (int(v) for v in values)
        if r0 > r1 or c0 > c1 or s0 > s1:
            raise ValueError(f"annotation row {row_number}: reversed box bounds")
        boxes[patient] = ((c0, r0, s0), (c1 + 1, r1 + 1, s1 + 1))
    return boxes


def load_annotation_boxes(path: str | os.PathLike) -> dict[str, Box]:
    """Read ``Annotation_Boxes.xlsx`` (sheet ``Sheet1``) or a CSV export with the same columns."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        data = pd.read_excel(path, sheet_name="Sheet1")
    return boxes_from_frame(data)


def annotation_coverage(studies: pd.DataFrame, boxes: Mapping[str, Box],
                        image_root: str | os.PathLike) -> tuple[dict[str, tuple[Box, tuple[int, ...]]], pd.DataFrame]:
    """Audit all studies; return ``study_id -> (box, source_shape)`` for eligible studies + report.

    Statuses (legacy): ``eligible``, ``missing_box``, ``ambiguous_patient_studies``
    (patient has != 1 study in ``studies``), ``missing_series``,
    ``incompatible_series_geometry`` (non-3D, shape or affine disagreement between
    series), ``out_of_volume_box`` (any series), ``unmatched_annotation``.
    New: ``missing_series_file`` (legacy raised while loading).
    """
    missing = {PATIENT_ID, STUDY_ID, SERIES_UIDS}.difference(studies.columns)
    if missing:
        raise ValueError(f"annotation matching requires columns {sorted(missing)}")
    if studies[STUDY_ID].duplicated().any():
        raise ValueError("duplicate study_id rows make annotation matching ambiguous")
    patients = studies[PATIENT_ID].astype(str).str.strip()
    counts = Counter(patients)
    matched: dict[str, tuple[Box, tuple[int, ...]]] = {}
    report = []
    for patient, row in zip(patients, studies.itertuples(index=False)):
        study = str(getattr(row, STUDY_ID))
        status, detail = "eligible", ""
        box = boxes.get(patient)
        uids = parse_series_uids(getattr(row, SERIES_UIDS))
        reference = None
        if box is None:
            status = "missing_box"
        elif counts[patient] != 1:
            status = "ambiguous_patient_studies"
        elif not uids:
            status = "missing_series"
        else:
            for uid in uids:
                try:
                    geometry = read_header_geometry(series_path(image_root, uid))
                except (SeriesNotFoundError, ValueError) as error:
                    status, detail = "missing_series_file", str(error)
                    break
                shape = tuple(geometry.shape)
                if len(shape) != 3:
                    status = "incompatible_series_geometry"
                    break
                if any(end > size for end, size in zip(box[1], shape)):
                    status = "out_of_volume_box"
                    detail = f"Annotation for {patient} exceeds {uid} image shape {shape}: {box}"
                    break
                affine = np.asarray(geometry.affine)
                if reference is not None and (shape != reference[0] or not np.allclose(
                        affine, reference[1], rtol=1e-4, atol=1e-3)):
                    status = "incompatible_series_geometry"
                    break
                reference = (shape, affine)
            if status == "eligible" and reference is not None:
                matched[study] = (box, reference[0])
        report.append({PATIENT_ID: patient, STUDY_ID: study, "status": status, "detail": detail})
    unused = sorted(set(boxes).difference(patients))
    report.extend({PATIENT_ID: p, STUDY_ID: "", "status": "unmatched_annotation", "detail": ""} for p in unused)
    return matched, pd.DataFrame(report, columns=list(COVERAGE_COLUMNS))


def write_annotation_coverage(report: pd.DataFrame, path: str | os.PathLike) -> pd.DataFrame:
    """Write ``annotation_coverage.csv`` and return the frame written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = report.loc[:, list(COVERAGE_COLUMNS)].copy()
    frame.to_csv(path, index=False)
    return frame


def resize_box(box: Box, source_shape: Sequence[int], target_shape: Sequence[int]) -> Box:
    """Map half-open voxel edges through an axis-aligned resize (floor start, ceil exclusive end)."""
    starts, ends = box
    new_starts = tuple(int(np.floor(a * dst / src)) for a, src, dst in zip(starts, source_shape, target_shape))
    new_ends = tuple(int(np.ceil(b * dst / src)) for b, src, dst in zip(ends, source_shape, target_shape))
    return new_starts, new_ends  # type: ignore[return-value]


def erase_background_box(volume: np.ndarray, lesion: Box, rng: np.random.Generator,
                         attempts: int = 8) -> tuple[np.ndarray, tuple[int, ...] | None]:
    """Zero one background cuboid (sides ``max(1, dim // 4)``) shared across channels.

    Up to ``attempts`` uniform draws; a draw intersecting the lesion box on all
    three axes is rejected. If none fits the volume is returned unchanged.
    Returns ``(volume, origin or None)``; the input is never modified in place.
    """
    spatial = volume.shape[-3:]
    patch = tuple(max(1, size // 4) for size in spatial)
    starts, ends = lesion
    for _ in range(int(attempts)):
        origin = tuple(int(rng.integers(0, size - width + 1)) for size, width in zip(spatial, patch))
        if all(origin[a] + patch[a] > starts[a] and origin[a] < ends[a] for a in range(3)):
            continue
        out = np.array(volume, copy=True)
        d, h, w = origin
        pd_, ph, pw = patch
        out[..., d:d + pd_, h:h + ph, w:w + pw] = 0
        return out, origin
    return volume, None
