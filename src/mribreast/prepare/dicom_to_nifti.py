"""DICOM series -> ``<SeriesInstanceUID>.nii.gz`` with UID and geometry verification.

Replaces legacy ``mriBreastDuke/dataLoaders/transfer_dcm_to_nifti.py`` which
trusted the download directory name as the series UID, paired folders and
names with ``zip`` (misaligned when non-directories exist), wrote
``NIFTI_PATH + serie + '.nii.gz'`` without a path separator and checked no
geometry.  Here a series is written only after:

1. every DICOM file in the directory has ``SeriesInstanceUID == expected_uid``
   (and one ``StudyInstanceUID``, equal to ``expected_study_uid`` when given);
2. rows/columns, ``PixelSpacing`` and ``ImageOrientationPatient`` agree on all
   slices;
3. slice positions (``ImagePositionPatient`` projected on the slice normal) are
   unique and uniformly spaced, and all slices lie on one line along the
   normal (no gantry tilt / mixed stacks).

Voxel order and affine follow SimpleITK's ``ImageSeriesReader`` + ``WriteImage``
(the legacy converter): array axes ``(column, row, slice)`` with slices sorted
by position along the normal, NIfTI affine = LPS->RAS of the DICOM geometry.
The ``nibabel`` backend (default) needs only ``pydicom`` + ``nibabel``; the
``sitk`` backend uses SimpleITK for pixel reading (legacy-identical) after the
same checks.  ``pydicom``/SimpleITK are imported lazily (``pip install .[prepare]``).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BACKENDS = ("nibabel", "sitk")
SIDE_CAR_DIR = "_conversion"


class ConversionError(ValueError):
    """The series cannot be converted safely."""


class SeriesUidMismatch(ConversionError):
    """DICOM headers do not carry the expected SeriesInstanceUID."""


class GeometryError(ConversionError):
    """Inconsistent or ambiguous slice geometry."""


@dataclass
class SeriesGeometry:
    series_uid: str
    study_uid: str
    patient_id: str
    shape: tuple[int, int, int]                 # (columns, rows, slices) = NIfTI voxel axes
    spacing: tuple[float, float, float]         # (column spacing, row spacing, slice spacing) mm
    affine: list[list[float]]                   # voxel -> RAS mm
    orientation: str                            # nibabel axis codes, e.g. "LPS"
    files: list[str] = field(default_factory=list)   # sorted by slice position
    rescale: bool = False
    dtype: str = ""


def _require_pydicom():
    try:
        import pydicom  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError("DICOM conversion needs pydicom: pip install 'mribreast[prepare]'") from error
    return pydicom


def _read_headers(series_dir: Path) -> list[tuple[Path, Any]]:
    pydicom = _require_pydicom()
    from pydicom.errors import InvalidDicomError

    headers = []
    for path in sorted(p for p in series_dir.rglob("*") if p.is_file() and not p.name.startswith(".")):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, IsADirectoryError):
            continue
        if "PixelData" not in ds and getattr(ds, "Rows", None) is None:
            continue
        headers.append((path, ds))
    return headers


def _floats(value: Any, n: int, name: str, path: Path) -> np.ndarray:
    try:
        arr = np.asarray([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GeometryError(f"{path.name}: unreadable {name}") from error
    if arr.shape != (n,):
        raise GeometryError(f"{path.name}: {name} must have {n} values")
    return arr


def inspect_series(series_dir: str | os.PathLike, expected_uid: str, *, expected_study_uid: str | None = None,
                   orientation_atol: float = 1e-4, spacing_rtol: float = 1e-2, position_atol: float = 1e-2) -> SeriesGeometry:
    """Verify UID + geometry from headers only (no pixel data) and return the sorted geometry."""
    series_dir = Path(series_dir)
    headers = _read_headers(series_dir)
    if not headers:
        raise ConversionError(f"{series_dir}: no DICOM image files")
    uids = {str(ds.SeriesInstanceUID) for _, ds in headers}
    if uids != {str(expected_uid)}:
        raise SeriesUidMismatch(f"{series_dir}: expected SeriesInstanceUID {expected_uid}, headers carry {sorted(uids)}")
    studies = {str(getattr(ds, "StudyInstanceUID", "")) for _, ds in headers}
    if len(studies) != 1:
        raise ConversionError(f"{series_dir}: several StudyInstanceUIDs {sorted(studies)}")
    study_uid = studies.pop()
    if expected_study_uid is not None and study_uid != str(expected_study_uid):
        raise ConversionError(f"{series_dir}: StudyInstanceUID {study_uid} != expected {expected_study_uid}")
    first_path, first = headers[0]
    rows, cols = int(first.Rows), int(first.Columns)
    pixel_spacing = _floats(first.PixelSpacing, 2, "PixelSpacing", first_path)
    iop = _floats(first.ImageOrientationPatient, 6, "ImageOrientationPatient", first_path)
    row_dir, col_dir = iop[:3], iop[3:]        # row_dir: direction of increasing column index
    normal = np.cross(row_dir, col_dir)
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-3):
        raise GeometryError(f"{first_path.name}: ImageOrientationPatient is not orthonormal")
    positions = []
    for path, ds in headers:
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise GeometryError(f"{path.name}: matrix {ds.Rows}x{ds.Columns} differs from {rows}x{cols}")
        if not np.allclose(_floats(ds.PixelSpacing, 2, "PixelSpacing", path), pixel_spacing, rtol=1e-4, atol=1e-5):
            raise GeometryError(f"{path.name}: PixelSpacing differs within the series")
        if not np.allclose(_floats(ds.ImageOrientationPatient, 6, "ImageOrientationPatient", path), iop,
                           atol=orientation_atol):
            raise GeometryError(f"{path.name}: ImageOrientationPatient differs within the series")
        positions.append(_floats(ds.ImagePositionPatient, 3, "ImagePositionPatient", path))
    positions_arr = np.vstack(positions)
    distance = positions_arr @ normal
    order = np.argsort(distance, kind="stable")
    distance, positions_arr = distance[order], positions_arr[order]
    files = [str(headers[i][0]) for i in order]
    if len(files) > 1:
        steps = np.diff(distance)
        if np.any(steps <= position_atol):
            raise GeometryError(f"{series_dir}: duplicate slice positions (multi-phase or repeated slices)")
        step = float(np.median(steps))
        if np.max(np.abs(steps - step)) > max(position_atol, spacing_rtol * step):
            raise GeometryError(f"{series_dir}: non-uniform slice spacing {steps.min():.4f}..{steps.max():.4f} mm")
        in_plane = positions_arr - np.outer(distance - distance[0], normal) - positions_arr[0]
        if np.max(np.linalg.norm(in_plane, axis=1)) > max(position_atol, 0.05 * step):
            raise GeometryError(f"{series_dir}: slices are not stacked along the normal (tilt / mixed stacks)")
    else:
        step = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    lps = np.eye(4)
    lps[:3, 0] = row_dir * pixel_spacing[1]
    lps[:3, 1] = col_dir * pixel_spacing[0]
    lps[:3, 2] = normal * step
    lps[:3, 3] = positions_arr[0]
    ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps
    import nibabel as nib

    slopes = {float(getattr(ds, "RescaleSlope", 1) or 1) for _, ds in headers}
    intercepts = {float(getattr(ds, "RescaleIntercept", 0) or 0) for _, ds in headers}
    return SeriesGeometry(
        series_uid=str(expected_uid), study_uid=study_uid, patient_id=str(getattr(first, "PatientID", "")),
        shape=(cols, rows, len(files)), spacing=(float(pixel_spacing[1]), float(pixel_spacing[0]), step),
        affine=ras.tolist(), orientation="".join(nib.aff2axcodes(ras)), files=files,
        rescale=slopes != {1.0} or intercepts != {0.0})


def _volume_nibabel(geometry: SeriesGeometry) -> np.ndarray:
    pydicom = _require_pydicom()
    slices = []
    for path in geometry.files:
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array
        if geometry.rescale:
            pixels = pixels.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) \
                + float(getattr(ds, "RescaleIntercept", 0) or 0)
        slices.append(np.asarray(pixels).T)          # (rows, cols) -> (cols, rows)
    return np.stack(slices, axis=2)


def _volume_sitk(series_dir: Path, geometry: SeriesGeometry) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover
        raise ImportError("backend='sitk' needs SimpleITK: pip install 'mribreast[prepare]'") from error
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(series_dir), geometry.series_uid, recursive=True)
    if len(names) != len(geometry.files):
        raise GeometryError(f"{series_dir}: GDCM found {len(names)} files, headers {len(geometry.files)}")
    reader.SetFileNames(names)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)            # (slices, rows, cols)
    return np.transpose(array, (2, 1, 0))


def convert_series(series_dir: str | os.PathLike, expected_uid: str, output_root: str | os.PathLike, *,
                   expected_study_uid: str | None = None, backend: str = "nibabel", overwrite: bool = False,
                   write_sidecar: bool = True) -> dict[str, Any]:
    """Verify and convert one series directory. Returns a report row (status ``converted``/``skipped_existing``)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    series_dir, output_root = Path(series_dir), Path(output_root)
    target = output_root / f"{expected_uid}.nii.gz"
    if target.exists() and not overwrite:
        return {"series_uid": expected_uid, "status": "skipped_existing", "output": str(target), "error": ""}
    geometry = inspect_series(series_dir, expected_uid, expected_study_uid=expected_study_uid)
    volume = _volume_nibabel(geometry) if backend == "nibabel" else _volume_sitk(series_dir, geometry)
    if tuple(volume.shape) != geometry.shape:
        raise GeometryError(f"{series_dir}: pixel volume {volume.shape} != header geometry {geometry.shape}")
    import nibabel as nib

    image = nib.Nifti1Image(volume, np.asarray(geometry.affine))
    image.header.set_xyzt_units("mm")
    geometry.dtype = str(volume.dtype)
    output_root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{expected_uid}.", suffix=".nii.gz", dir=output_root)
    os.close(fd)
    try:
        nib.save(image, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if write_sidecar:
        sidecar = output_root / SIDE_CAR_DIR / f"{expected_uid}.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        info = asdict(geometry)
        info["files"] = [Path(f).name for f in info["files"]]
        info["backend"] = backend
        sidecar.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"series_uid": expected_uid, "status": "converted", "output": str(target), "error": "",
            "shape": " ".join(map(str, geometry.shape)), "spacing": " ".join(f"{v:.6g}" for v in geometry.spacing),
            "orientation": geometry.orientation, "study_uid": geometry.study_uid}


def convert_tree(download_root: str | os.PathLike, output_root: str | os.PathLike, *,
                 expected: Iterable[tuple[str, str]] | None = None, backend: str = "nibabel",
                 overwrite: bool = False) -> list[dict[str, Any]]:
    """Convert every series directory below ``download_root``.

    ``expected`` yields ``(directory name, SeriesInstanceUID)``; by default the
    directory name is the expected UID (layout written by :mod:`.nbia` and by
    ``tcia_utils``).  Failures are reported per series and never published.
    """
    download_root = Path(download_root)
    pairs: Sequence[tuple[str, str]] = list(expected) if expected is not None else [
        (p.name, p.name) for p in sorted(download_root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    report = []
    for directory, uid in pairs:
        try:
            report.append(convert_series(download_root / directory, uid, output_root, backend=backend,
                                         overwrite=overwrite))
        except SeriesUidMismatch as error:
            report.append({"series_uid": uid, "status": "uid_mismatch", "output": "", "error": str(error)})
        except GeometryError as error:
            report.append({"series_uid": uid, "status": "geometry_error", "output": "", "error": str(error)})
        except ConversionError as error:
            report.append({"series_uid": uid, "status": "failed", "output": "", "error": str(error)})
    return report
