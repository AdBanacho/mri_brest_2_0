"""Deterministic synthetic imaging data (NIfTI series, 2D PNG/TIFF images, masks). No real data.

* Every study has a lesion (box geometry independent of the label); only its
  contrast enhancement in post-contrast phases depends on the label, so a model
  can learn the task and masks/boxes never leak the label by their presence.
* Volumes use a non-identity affine (LPS-like, anisotropic spacing) and an
  asymmetric intensity gradient along array axis 0.
* Duke box convention (verified, see docs/imaging.md): zero-based inclusive
  ``Start/End Column`` -> array axis 0, ``Row`` -> axis 1, ``Slice`` -> axis 2.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from mribreast.contracts import LABEL, PATIENT_ID, SERIES_UIDS, STUDY_ID
from mribreast.imaging.annotations import COLUMNS as BOX_COLUMNS
from mribreast.imaging.annotations import boxes_from_frame
from mribreast.imaging.geometry import parse_series_uids

SYNTHETIC_AFFINE = np.array([[-0.7, 0.0, 0.0, 90.0],
                             [0.0, -0.75, 0.0, 110.0],
                             [0.0, 0.0, 2.0, -40.0],
                             [0.0, 0.0, 0.0, 1.0]])
CLASS_CODES_JSON = json.dumps({"0": "background", "1": "lesion"})


def _crc(*parts: Any) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _region_for(patient: str, study: str, shape: Sequence[int], boxes: dict[str, Any] | None,
                seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Half-open lesion region in array axes; from the Duke box when given (and inside the volume)."""
    if boxes is not None and patient in boxes:
        starts, ends = boxes[patient]
        if all(e <= s for e, s in zip(ends, shape)):
            return tuple(starts), tuple(ends)
    rng = np.random.default_rng([seed, _crc("lesion", patient, study)])
    sizes = [max(1, s // 4) for s in shape]
    starts = tuple(int(rng.integers(0, s - w + 1)) for s, w in zip(shape, sizes))
    return starts, tuple(a + w for a, w in zip(starts, sizes))


def make_annotation_boxes(studies: pd.DataFrame, shape: Sequence[int], seed: int = 0) -> pd.DataFrame:
    """Duke-format box sheet (one row per patient), label-independent, inside ``shape``."""
    rows = []
    for patient in dict.fromkeys(studies[PATIENT_ID].astype(str)):
        rng = np.random.default_rng([seed, _crc("box", patient)])
        sizes = [max(1, int(s) // 4) for s in shape]
        starts = [int(rng.integers(0, int(s) - w + 1)) for s, w in zip(shape, sizes)]
        c0, r0, s0 = starts
        c1, r1, s1 = (a + w - 1 for a, w in zip(starts, sizes))   # inclusive ends
        rows.append((patient, r0, r1, c0, c1, s0, s1))
    return pd.DataFrame(rows, columns=list(BOX_COLUMNS))


def _boxes_map(boxes: pd.DataFrame | None) -> dict[str, Any] | None:
    if boxes is None:
        return None
    return boxes_from_frame(boxes.loc[:, list(BOX_COLUMNS)])


def write_nifti_series(image_root: str | os.PathLike, studies: pd.DataFrame, shape: Sequence[int] = (24, 24, 12),
                       boxes: pd.DataFrame | None = None, seed: int = 0) -> None:
    """Write ``<uid>.nii.gz`` (int16) for every series of every study (phase 0 = pre-contrast)."""
    root = Path(image_root)
    root.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(s) for s in shape)
    box_map = _boxes_map(boxes)
    grad = np.linspace(0.0, 60.0, shape[0], dtype=np.float32)[:, None, None]
    for row in studies.itertuples(index=False):
        patient, study = str(getattr(row, PATIENT_ID)), str(getattr(row, STUDY_ID))
        label = int(getattr(row, LABEL))
        uids = parse_series_uids(getattr(row, SERIES_UIDS))
        starts, ends = _region_for(patient, study, shape, box_map, seed)
        region = tuple(slice(a, b) for a, b in zip(starts, ends))
        amplitude = 30.0 + 120.0 * label
        for phase, uid in enumerate(uids):
            rng = np.random.default_rng([seed, _crc("phase", study, phase)])
            volume = 200.0 + grad + rng.normal(0.0, 10.0, size=shape).astype(np.float32)
            enhancement = 20.0 + (amplitude * (1.0 - np.exp(-phase)) if phase > 0 else 0.0)
            volume[region] += enhancement
            image = nib.Nifti1Image(np.clip(np.rint(volume), 0, 32767).astype(np.int16), SYNTHETIC_AFFINE)
            image.header.set_xyzt_units("mm", "sec")
            nib.save(image, str(root / f"{uid}.nii.gz"))


def write_2d_dataset(root: str | os.PathLike, studies: pd.DataFrame, size: Sequence[int] = (32, 32),
                     n_images_per_study: int = 3, seed: int = 0, *, vary_count: bool = False) -> Path:
    """Write PNG (even study rows) / TIFF (odd rows) images + ``manifest.csv``; returns the manifest path.

    ``vary_count=True`` gives study ``k`` ``1 + k % n_images_per_study`` images.
    Manifest columns: patient_id, study_id, series_id, slice_id, path, lesion_box (JSON ``[r0, r1, c0, c1]``).
    """
    import tifffile
    from PIL import Image

    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    height, width = (int(v) for v in size)
    rows = []
    for k, row in enumerate(studies.itertuples(index=False)):
        patient, study = str(getattr(row, PATIENT_ID)), str(getattr(row, STUDY_ID))
        label = int(getattr(row, LABEL))
        uids = parse_series_uids(getattr(row, SERIES_UIDS, "")) if hasattr(row, SERIES_UIDS) else ()
        series = uids[0] if uids else f"{study}-2d"
        count = 1 + k % n_images_per_study if vary_count else n_images_per_study
        for index in range(count):
            rng = np.random.default_rng([seed, _crc("img2d", study, index)])
            image = 80.0 + np.linspace(0, 30, width)[None, :] + rng.normal(0, 8, size=(height, width))
            bh, bw = max(1, height // 4), max(1, width // 4)
            r0, c0 = int(rng.integers(0, height - bh + 1)), int(rng.integers(0, width - bw + 1))
            image[r0:r0 + bh, c0:c0 + bw] += 25.0 + 90.0 * label
            if k % 2 == 0:
                rel = f"images/{_safe(study)}_{index}.png"
                Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)).save(root / rel)
            else:
                rel = f"images/{_safe(study)}_{index}.tif"
                tifffile.imwrite(str(root / rel), np.clip(np.rint(image * 16), 0, 65535).astype(np.uint16))
            rows.append({PATIENT_ID: patient, STUDY_ID: study, "series_id": series, "slice_id": str(index),
                         "path": rel, "lesion_box": json.dumps([r0, r0 + bh, c0, c0 + bw])})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _pick(n: int, n_missing: int, n_empty: int, seed: int) -> tuple[set[int], set[int]]:
    order = np.random.default_rng([seed, 7]).permutation(n).tolist()
    missing = set(order[:max(0, n_missing)])
    empty = set(order[max(0, n_missing):max(0, n_missing) + max(0, n_empty)])
    return missing, empty


def write_masks(root: str | os.PathLike, studies_or_manifest: pd.DataFrame | str | os.PathLike, *, seed: int = 0,
                n_missing: int = 1, n_empty: int = 1, image_root: str | os.PathLike | None = None,
                boxes: pd.DataFrame | None = None, folds: Sequence[int] | None = None,
                source: str = "predicted", provenance: str = "frozen_external") -> Path:
    """Write masks + ``manifest.csv`` (LocalFileMaskProvider format); returns the manifest path.

    * ``studies_or_manifest`` = path of a 2D image manifest -> one PNG/TIFF mask per image
      (lesion box from the manifest's ``lesion_box`` column).
    * ``studies_or_manifest`` = study DataFrame (+ ``image_root``) -> one NIfTI volume mask
      (``slice_id`` empty) per study on the grid of its first series; lesion region as in
      :func:`write_nifti_series` (pass the same ``boxes``/``seed``).
    ``n_missing`` items get no manifest row, ``n_empty`` items get an all-background mask.
    ``folds`` -> one mask set per fold (``fold`` column, ``training_provenance=fold_trained:<k>``,
    region shifted by one voxel per fold to mimic different segmenters).
    """
    import tifffile
    from PIL import Image

    from mribreast.imaging.manifest import read_image_file

    root = Path(root)
    (root / "files").mkdir(parents=True, exist_ok=True)
    fold_list: list[int | None] = list(folds) if folds else [None]
    rows: list[dict[str, Any]] = []
    if isinstance(studies_or_manifest, (str, os.PathLike)):
        manifest_path = Path(studies_or_manifest)
        images = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        missing, empty = _pick(len(images), n_missing, n_empty, seed)
        for position, image_row in enumerate(images.itertuples(index=False)):
            if position in missing:
                continue
            path = Path(image_row.path)
            path = path if path.is_absolute() else manifest_path.parent / path
            height, width = read_image_file(path).shape[1:]
            box = json.loads(image_row.lesion_box) if getattr(image_row, "lesion_box", "") else None
            for fold in fold_list:
                mask = np.zeros((height, width), dtype=np.uint8)
                if position not in empty:
                    if box is None:
                        r0, r1, c0, c1 = height // 4, height // 2, width // 4, width // 2
                    else:
                        r0, r1, c0, c1 = box
                    shift = 0 if fold is None else int(fold) % 2
                    mask[min(r0 + shift, height - 1):r1, c0:c1] = 1
                suffix = Path(path).suffix.lower()
                tag = "" if fold is None else f"_f{fold}"
                rel = f"files/{_safe(image_row.study_id)}_{_safe(image_row.slice_id)}{tag}_mask{suffix}"
                if suffix == ".png":
                    Image.fromarray(mask).save(root / rel)
                else:
                    tifffile.imwrite(str(root / rel), mask)
                rows.append(_mask_row(image_row.patient_id, image_row.study_id, image_row.series_id,
                                      image_row.slice_id, rel, fold, source, provenance))
    else:
        if image_root is None:
            raise ValueError("NIfTI masks need image_root (the grid of each study's first series)")
        from mribreast.imaging.geometry import series_path

        studies = studies_or_manifest.reset_index(drop=True)
        box_map = _boxes_map(boxes)
        missing, empty = _pick(len(studies), n_missing, n_empty, seed)
        for position, row in enumerate(studies.itertuples(index=False)):
            if position in missing:
                continue
            patient, study = str(getattr(row, PATIENT_ID)), str(getattr(row, STUDY_ID))
            uid = parse_series_uids(getattr(row, SERIES_UIDS))[0]
            reference = nib.load(str(series_path(image_root, uid)))
            shape = tuple(int(s) for s in reference.shape[:3])
            starts, ends = _region_for(patient, study, shape, box_map, seed)
            for fold in fold_list:
                mask = np.zeros(shape, dtype=np.uint8)
                if position not in empty:
                    shift = 0 if fold is None else int(fold) % 2
                    region = tuple(slice(min(a + (shift if axis == 0 else 0), s - 1), b)
                                   for axis, (a, b, s) in enumerate(zip(starts, ends, shape)))
                    mask[region] = 1
                tag = "" if fold is None else f"_f{fold}"
                rel = f"files/{_safe(uid)}{tag}_mask.nii.gz"
                nib.save(nib.Nifti1Image(mask, reference.affine), str(root / rel))
                rows.append(_mask_row(patient, study, uid, "", rel, fold, source, provenance))
    manifest = root / "manifest.csv"
    from mribreast.imaging.masks.local import mask_manifest_frame

    mask_manifest_frame(rows).to_csv(manifest, index=False)
    return manifest


def _mask_row(patient: str, study: str, series: str, slice_id: str, rel: str, fold: int | None, source: str,
              provenance: str) -> dict[str, Any]:
    return {PATIENT_ID: patient, STUDY_ID: study, "series_id": series, "slice_id": slice_id, "path": rel,
            "class_codes": CLASS_CODES_JSON, "source": source, "model_id": "synthetic-segmenter",
            "model_version": "1.0", "grid_source": series,
            "training_provenance": provenance if fold is None else f"fold_trained:{fold}",
            "known_overlap": "none (synthetic data)", "fold": "" if fold is None else str(fold)}


def synthetic_studies(n_patients: int = 8, *, phases: int = 3, seed: int = 0, studies_per_patient: int = 1,
                      num_classes: int = 2) -> pd.DataFrame:
    """Small cohort frame (patient_id, study_id, label, series_uids) with alternating labels."""
    rows = []
    for p in range(n_patients):
        for s in range(studies_per_patient):
            study = f"1.2.826.0.1.{seed}.{p}.{s}"
            uids = tuple(f"{study}.{k + 1}" for k in range(phases))
            rows.append({PATIENT_ID: f"Breast_MRI_{p:03d}", STUDY_ID: study, LABEL: p % num_classes,
                         SERIES_UIDS: uids})
    return pd.DataFrame(rows)
