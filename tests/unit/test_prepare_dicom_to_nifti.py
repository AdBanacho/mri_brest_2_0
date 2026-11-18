"""DICOM -> NIfTI with UID/geometry verification on tiny synthetic DICOM series (pydicom)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom", reason="pydicom not installed (pip install .[prepare])")
nib = pytest.importorskip("nibabel")

from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from mribreast.prepare import dicom_to_nifti as d2n  # noqa: E402

ROWS, COLS, SLICES = 4, 3, 5
PIXEL_SPACING = (0.7, 0.8)          # (row spacing, column spacing)
SLICE_STEP = 2.5


def _write_series(directory, series_uid, study_uid, *, iop=(1, 0, 0, 0, 1, 0), positions=None, uids=None,
                  orientations=None, seed=0):
    directory.mkdir(parents=True, exist_ok=True)
    row_dir, col_dir = np.array(iop[:3], float), np.array(iop[3:], float)
    normal = np.cross(row_dir, col_dir)
    origin = np.array([-10.0, 20.0, 5.0])
    if positions is None:
        positions = [origin + normal * SLICE_STEP * k for k in range(SLICES)]
    order = list(range(len(positions)))
    random.Random(seed).shuffle(order)            # files are written in random order with random names
    for k in order:
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.PatientID = "Breast_MRI_900"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = (uids or {}).get(k, series_uid)
        ds.InstanceNumber = k + 1
        ds.Rows, ds.Columns = ROWS, COLS
        ds.PixelSpacing = list(PIXEL_SPACING)
        ds.SliceThickness = SLICE_STEP
        ds.ImageOrientationPatient = list((orientations or {}).get(k, iop))
        ds.ImagePositionPatient = [float(v) for v in positions[k]]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        r, c = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
        ds.PixelData = (100 * k + 10 * r + c).astype(np.int16).tobytes()   # recognizable (k, row, col)
        ds.save_as(directory / f"{generate_uid()[-12:]}.dcm", enforce_file_format=True)
    return directory


def test_convert_orientation_affine_and_sidecar(tmp_path):
    series, study = generate_uid(), generate_uid()
    src = _write_series(tmp_path / "dl" / series, series, study)
    row = d2n.convert_series(src, series, tmp_path / "nifti", expected_study_uid=study)
    assert row["status"] == "converted" and row["orientation"] == "LPS"
    image = nib.load(str(tmp_path / "nifti" / f"{series}.nii.gz"))
    data = np.asarray(image.dataobj)
    assert data.shape == (COLS, ROWS, SLICES)
    k, r, c = 3, 2, 1
    assert data[c, r, k] == 100 * k + 10 * r + c      # axis 0 = column, axis 1 = row, axis 2 = slice (ascending)
    expected = np.array([[-0.8, 0, 0, 10.0], [0, -0.7, 0, -20.0], [0, 0, SLICE_STEP, 5.0], [0, 0, 0, 1]])
    assert np.allclose(image.affine, expected, atol=1e-6)
    assert (tmp_path / "nifti" / d2n.SIDE_CAR_DIR / f"{series}.json").is_file()
    assert d2n.convert_series(src, series, tmp_path / "nifti")["status"] == "skipped_existing"


def test_uid_mismatch_is_never_published(tmp_path):
    series, study = generate_uid(), generate_uid()
    src = _write_series(tmp_path / series, series, study)
    with pytest.raises(d2n.SeriesUidMismatch):
        d2n.convert_series(src, generate_uid(), tmp_path / "out")
    mixed = _write_series(tmp_path / "mixed", series, study, uids={2: generate_uid()})
    with pytest.raises(d2n.SeriesUidMismatch):
        d2n.convert_series(mixed, series, tmp_path / "out")
    with pytest.raises(d2n.ConversionError):
        d2n.convert_series(src, series, tmp_path / "out", expected_study_uid=generate_uid())
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.nii.gz"))


@pytest.mark.parametrize("case", ["nonuniform", "duplicate", "orientation", "tilt"])
def test_geometry_errors(tmp_path, case):
    series, study = generate_uid(), generate_uid()
    base = [np.array([0.0, 0.0, SLICE_STEP * k]) for k in range(SLICES)]
    kwargs = {}
    if case == "nonuniform":
        base[4] = base[4] + np.array([0, 0, 1.0])
        kwargs["positions"] = base
    elif case == "duplicate":
        base[4] = base[3]
        kwargs["positions"] = base
    elif case == "orientation":
        kwargs["orientations"] = {1: (1, 0, 0, 0, 0.9, math.sqrt(1 - 0.81))}
    else:
        kwargs["positions"] = [p + np.array([0.5 * k, 0, 0]) for k, p in enumerate(base)]
    src = _write_series(tmp_path / series, series, study, **kwargs)
    with pytest.raises(d2n.GeometryError):
        d2n.convert_series(src, series, tmp_path / "out")


def test_convert_tree_report(tmp_path):
    good, other = generate_uid(), generate_uid()
    _write_series(tmp_path / "dl" / good, good, generate_uid())
    _write_series(tmp_path / "dl" / "wrong-name", other, generate_uid())
    report = {r["series_uid"]: r["status"] for r in d2n.convert_tree(tmp_path / "dl", tmp_path / "nifti")}
    assert report == {good: "converted", "wrong-name": "uid_mismatch"}


def test_matches_legacy_simpleitk_conversion(tmp_path):
    """Legacy transfer_dcm_to_nifti: sitk ImageSeriesReader + WriteImage. Same voxels/affine, oblique series."""
    sitk = pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
    a = math.radians(30)
    iop = (math.cos(a), math.sin(a), 0.0, -math.sin(a), math.cos(a), 0.0)
    series = generate_uid()
    src = _write_series(tmp_path / series, series, generate_uid(), iop=iop, seed=3)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(src)))
    legacy_path = tmp_path / "legacy.nii.gz"
    sitk.WriteImage(reader.Execute(), str(legacy_path))
    legacy = nib.load(str(legacy_path))
    for backend in ("nibabel", "sitk"):
        d2n.convert_series(src, series, tmp_path / backend, backend=backend)
        new = nib.load(str(tmp_path / backend / f"{series}.nii.gz"))
        assert np.array_equal(np.asarray(new.dataobj), np.asarray(legacy.dataobj)), backend
        assert np.allclose(new.affine, legacy.affine, atol=1e-4), backend
