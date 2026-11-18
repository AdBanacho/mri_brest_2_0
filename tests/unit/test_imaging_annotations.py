"""Duke annotation boxes: convention, audit statuses, resize mapping and training-only erasing."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mribreast.imaging.annotations import (COLUMNS, annotation_coverage, boxes_from_frame, erase_background_box,
                                           load_annotation_boxes, resize_box, write_annotation_coverage)
from mribreast.imaging.datasets import ImageDataset, ItemBuilder
from mribreast.imaging.settings import ImagingSettings
from mribreast.synthetic.imaging import make_annotation_boxes, synthetic_studies, write_nifti_series


def _save(root, uid, shape, affine=None):
    nib.save(nib.Nifti1Image(np.zeros(shape, np.uint8), np.eye(4) if affine is None else affine),
             str(root / f"{uid}.nii.gz"))


def _studies(rows):
    return pd.DataFrame(rows, columns=["patient_id", "study_id", "series_uids"])


def test_workbook_convention_zero_based_inclusive_column_row_slice(tmp_path):
    sheet = pd.DataFrame([("P1", 10, 12, 3, 4, 0, 1)], columns=list(COLUMNS))
    sheet.to_csv(tmp_path / "boxes.csv", index=False)
    boxes = load_annotation_boxes(tmp_path / "boxes.csv")
    # rows 10..12 / columns 3..4 / slices 0..1 inclusive -> half-open (col, row, slice)
    assert boxes == {"P1": ((3, 10, 0), (5, 13, 2))}
    # legacy test value (Breast_MRI_207: rows 374-386, columns 358-364, slices 34-39)
    legacy = boxes_from_frame(pd.DataFrame([("Breast_MRI_207", 374, 386, 358, 364, 34, 39)], columns=list(COLUMNS)))
    assert legacy["Breast_MRI_207"] == ((358, 374, 34), (365, 387, 40))


@pytest.mark.parametrize("coords", [(-1, 2, 0, 2, 0, 2), (3, 2, 0, 2, 0, 2), (0.5, 2, 0, 2, 0, 2)])
def test_malformed_coordinates_raise(coords):
    with pytest.raises(ValueError):
        boxes_from_frame(pd.DataFrame([("p", *coords)], columns=list(COLUMNS)))


def test_duplicates_and_bad_columns_raise():
    with pytest.raises(ValueError, match="duplicate"):
        boxes_from_frame(pd.DataFrame([("p", 0, 1, 0, 1, 0, 1)] * 2, columns=list(COLUMNS)))
    with pytest.raises(ValueError, match="columns"):
        boxes_from_frame(pd.DataFrame([("p", 0, 1)], columns=["Patient ID", "a", "b"]))


def test_coverage_statuses(tmp_path):
    _save(tmp_path, "big", (16, 16, 8))
    _save(tmp_path, "small", (8, 8, 8))
    shifted = np.eye(4)
    shifted[0, 3] = 10
    _save(tmp_path, "moved", (16, 16, 8), shifted)
    boxes = {"ok": ((0, 0, 0), (16, 16, 8)), "oov": ((0, 0, 0), (9, 8, 8)), "multi": ((1, 1, 1), (2, 2, 2)),
             "geom": ((1, 1, 1), (2, 2, 2)), "missingfile": ((1, 1, 1), (2, 2, 2)), "unused": ((0, 0, 0), (1, 1, 1))}
    frame = _studies([
        ("ok", "s1", ("big", "big")),            # exclusive end exactly at the volume edge -> eligible
        ("oov", "s2", ("big", "small")),         # exceeds the second series on axis 0
        ("multi", "s3", ("big",)), ("multi", "s4", ("big",)),
        ("geom", "s5", "big|moved"),
        ("missingfile", "s6", ("nope",)),
        ("nobox", "s7", ("big",)),
    ])
    matched, report = annotation_coverage(frame, boxes, tmp_path)
    status = dict(zip(report["study_id"].where(report["study_id"] != "", report["patient_id"]), report["status"]))
    assert status == {"s1": "eligible", "s2": "out_of_volume_box", "s3": "ambiguous_patient_studies",
                      "s4": "ambiguous_patient_studies", "s5": "incompatible_series_geometry",
                      "s6": "missing_series_file", "s7": "missing_box", "unused": "unmatched_annotation"}
    assert matched == {"s1": (boxes["ok"], (16, 16, 8))}
    assert "small" in report.loc[report.study_id == "s2", "detail"].iloc[0]
    written = write_annotation_coverage(report, tmp_path / "out" / "annotation_coverage.csv")
    assert list(pd.read_csv(tmp_path / "out" / "annotation_coverage.csv").columns) == list(written.columns)
    with pytest.raises(ValueError, match="duplicate"):
        annotation_coverage(pd.concat([frame, frame.iloc[:1]]), boxes, tmp_path)


def test_resize_box_floor_start_ceil_end():
    box = ((3, 5, 1), (7, 10, 3))
    assert resize_box(box, (16, 20, 6), (8, 10, 4)) == ((1, 2, 0), (4, 5, 2))
    assert resize_box(box, (16, 20, 6), (16, 20, 6)) == box
    assert resize_box(((0, 0, 0), (16, 20, 6)), (16, 20, 6), (5, 7, 3)) == ((0, 0, 0), (5, 7, 3))


def test_erase_never_touches_lesion_and_is_shared_across_channels():
    volume = np.ones((3, 16, 16, 8), np.float32)
    lesion = ((4, 4, 2), (12, 12, 6))
    for seed in range(40):
        out, origin = erase_background_box(volume, lesion, np.random.default_rng(seed), attempts=8)
        assert np.all(out[:, 4:12, 4:12, 2:6] == 1)                       # lesion preserved
        if origin is not None:
            zeroed = out == 0
            assert zeroed.sum() == 3 * 4 * 4 * 2                           # max(1, dim // 4) sides
            assert np.array_equal(zeroed[0], zeroed[1]) and np.array_equal(zeroed[1], zeroed[2])
    assert volume.min() == 1                                               # input untouched
    full, origin = erase_background_box(volume, ((0, 0, 0), (16, 16, 8)), np.random.default_rng(0))
    assert origin is None and full is volume                               # nothing fits -> unchanged


def test_synthetic_boxes_are_label_independent_and_inside_volume():
    studies = synthetic_studies(12)
    flipped = studies.assign(label=1 - studies["label"])
    a = make_annotation_boxes(studies, (16, 16, 8), seed=3)
    b = make_annotation_boxes(flipped, (16, 16, 8), seed=3)
    pd.testing.assert_frame_equal(a, b)
    for starts, ends in boxes_from_frame(a).values():
        assert all(0 <= s < e <= n for s, e, n in zip(starts, ends, (16, 16, 8)))


def test_erasing_only_in_training_dataset_and_lesion_box_matches_synthetic_lesion(tmp_path):
    studies = synthetic_studies(2, phases=2)
    boxes = make_annotation_boxes(studies, (16, 16, 8), seed=0)
    write_nifti_series(tmp_path, studies, (16, 16, 8), boxes=boxes, seed=0)
    matched, _ = annotation_coverage(studies, boxes_from_frame(boxes), tmp_path)
    # the synthetic lesion (post - pre enhancement) lies exactly inside the box on axes (column, row, slice)
    assert studies.iloc[1]["label"] == 1                                  # strong enhancement: 5+ noise sigmas
    row = boxes.iloc[1]
    uids = studies.iloc[1]["series_uids"]
    pre = nib.load(str(tmp_path / f"{uids[0]}.nii.gz")).get_fdata()
    post = nib.load(str(tmp_path / f"{uids[1]}.nii.gz")).get_fdata()
    enhanced = np.argwhere(post - pre > 70)
    assert enhanced[:, 0].min() == row["Start Column"] and enhanced[:, 0].max() == row["End Column"]
    assert enhanced[:, 1].min() == row["Start Row"] and enhanced[:, 1].max() == row["End Row"]
    assert enhanced[:, 2].min() == row["Start Slice"] and enhanced[:, 2].max() == row["End Slice"]
    settings = ImagingSettings(image_root=str(tmp_path), phase_count=2, target_size=(8, 8, 4),
                               use_annotation_boxes=True)
    items = ItemBuilder(settings, None, None).build(studies, None).items
    train = ImageDataset(items, settings, channels=2, training=True, annotation_boxes=matched, seed=0)
    evaluation = ImageDataset(items, settings, channels=2, training=False, annotation_boxes=matched, seed=0)
    x_train, x_eval = train.load_item(0), evaluation.load_item(0)
    assert train.erase_log and train.erase_log[0][1] is not None
    assert (x_train == 0).sum() > (x_eval == 0).sum()
    assert evaluation.annotation_boxes == {}
