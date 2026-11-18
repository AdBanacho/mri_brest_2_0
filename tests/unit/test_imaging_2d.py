"""2D: image manifest (PNG/TIFF), deterministic slice selection, [C,H,W] layout, study aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from mribreast.imaging.datasets import ImageDataset, ItemBuilder, aggregate_study_probs, expected_channels
from mribreast.imaging.manifest import load_manifest_image, read_image_file, read_image_manifest
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.slices import extract_slice, roi_area_per_slice, select_slices
from mribreast.synthetic.imaging import synthetic_studies, write_2d_dataset, write_nifti_series


def test_manifest_png_tiff_and_channel_stacking(tmp_path):
    Image.fromarray(np.arange(12, dtype=np.uint8).reshape(3, 4)).save(tmp_path / "a.png")
    tifffile.imwrite(str(tmp_path / "b.tif"), (np.arange(12, dtype=np.uint16).reshape(3, 4) * 10))
    Image.fromarray(np.zeros((3, 4, 3), np.uint8)).save(tmp_path / "rgb.png")
    pd.DataFrame({"patient_id": ["p", "p", "q"], "study_id": ["s", "s", "t"], "series_id": ["x", "x", "y"],
                  "slice_id": [0, 0, 1], "path": ["b.tif", "a.png", "rgb.png"], "phase": [1, 0, 0]}
                 ).to_csv(tmp_path / "m.csv", index=False)
    manifest = read_image_manifest(tmp_path / "m.csv")
    assert manifest["slice_id"].tolist() == ["0", "0", "1"] and manifest["channel"].tolist() == [1, 0, 0]
    stacked = load_manifest_image(manifest[manifest.study_id == "s"])
    assert stacked.shape == (2, 3, 4)
    np.testing.assert_array_equal(stacked[0], np.arange(12).reshape(3, 4))          # channel 0 = PNG
    np.testing.assert_array_equal(stacked[1], np.arange(12).reshape(3, 4) * 10)     # channel 1 = TIFF
    assert read_image_file(tmp_path / "rgb.png").shape == (3, 3, 4)


def test_manifest_validation(tmp_path):
    pd.DataFrame({"patient_id": ["p"], "study_id": ["s"], "series_id": ["x"], "path": ["a.png"]}).to_csv(
        tmp_path / "missing.csv", index=False)
    with pytest.raises(ValueError, match="slice_id"):
        read_image_manifest(tmp_path / "missing.csv")
    pd.DataFrame({"patient_id": ["p", "p"], "study_id": ["s", "s"], "series_id": ["x", "x"], "slice_id": [0, 0],
                  "path": ["a.png", "b.png"]}).to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_image_manifest(tmp_path / "dup.csv")
    pd.DataFrame({"patient_id": ["p"], "study_id": ["s"], "series_id": ["x"], "slice_id": [0],
                  "path": ["a.jpg"]}).to_csv(tmp_path / "jpg.csv", index=False)
    with pytest.raises(ValueError, match="PNG/TIFF"):
        read_image_manifest(tmp_path / "jpg.csv")


def test_slice_selection_rules():
    assert select_slices(9, "center", 1) == [4]
    assert select_slices(10, "center", 3) == [4, 5, 6]
    assert select_slices(10, "uniform", 4) == [1, 3, 6, 8]
    assert select_slices(3, "uniform", 10) == [0, 1, 2]
    area = np.array([0, 5, 9, 9, 2, 0])
    assert select_slices(6, "mask_max_area", 2, area) == [2, 3]
    assert select_slices(6, "mask_max_area", 1, area) == [2]                      # tie -> lower index
    assert select_slices(6, "mask_all", 1, area) == [1, 2, 3, 4]
    assert select_slices(6, "mask_all", 1, np.zeros(6)) == []
    with pytest.raises(ValueError):
        select_slices(6, "mask_all", 1, None)


def test_extract_slice_axis_semantics():
    volume = np.arange(2 * 4 * 5 * 6).reshape(2, 4, 5, 6)
    assert extract_slice(volume, 2, 3).shape == (2, 4, 5)
    np.testing.assert_array_equal(extract_slice(volume, 0, 1), volume[:, 1, :, :])
    np.testing.assert_array_equal(extract_slice(volume, 1, 2), volume[:, :, 2, :])
    mask = np.zeros((4, 5, 6), int)
    mask[1, 2, 3] = 1
    mask[1, 2, 4] = 1
    np.testing.assert_array_equal(roi_area_per_slice(mask, 2, [1]), [0, 0, 0, 1, 1, 0])


def test_nifti_slices_items_have_cxhxw_and_follow_original_grid(tmp_path):
    studies = synthetic_studies(3, phases=3)
    write_nifti_series(tmp_path, studies, (12, 10, 6), seed=0)
    settings = ImagingSettings(image_dim="2d", image_root=str(tmp_path), phase_count=3, slice_axis=2,
                               slice_selection="uniform", slices_per_study=3, target_size_2d=(8, 8))
    table = ItemBuilder(settings, None, None).build(studies, None)
    assert len(table.items) == 9 and table.items["slice_id"].tolist()[:3] == ["1", "3", "5"]
    dataset = ImageDataset(table.items, settings, channels=expected_channels(settings))
    x, y, study = dataset[0]
    assert tuple(x.shape) == (3, 8, 8) and int(study) == 0
    axis0 = ImagingSettings(**{**settings.__dict__, "slice_axis": 0, "slices_per_study": 1,
                               "slice_selection": "center"})
    items = ItemBuilder(axis0, None, None).build(studies, None).items
    assert items["slice_id"].tolist() == ["6", "6", "6"]
    log: list = []
    out = ImageDataset(items, axis0, channels=3).load_item(0, log)
    assert out.shape == (3, 8, 8) and log[0]["op"] == "slice" and log[0]["axis"] == 0


def test_manifest_items_varying_counts_and_aggregation(tmp_path):
    studies = synthetic_studies(5)
    manifest_path = write_2d_dataset(tmp_path, studies, (16, 12), 3, seed=1, vary_count=True)
    manifest = read_image_manifest(manifest_path)
    assert {p.split(".")[-1] for p in manifest["path"]} == {"png", "tif"}
    settings = ImagingSettings(image_dim="2d", image_source="manifest_2d", image_manifest_file=str(manifest_path),
                               target_size_2d=(8, 8))
    extra = pd.concat([studies, pd.DataFrame([{"patient_id": "nobody", "study_id": "no.images", "label": 0,
                                               "series_uids": ()}])], ignore_index=True)
    table = ItemBuilder(settings, None, manifest).build(extra, None)
    counts = table.items.groupby("study_index").size().tolist()
    assert counts == [1, 2, 3, 1, 2] and table.notes[-1]["note"] == "no_images_in_manifest"
    assert expected_channels(settings, manifest) == 1
    x, _, _ = ImageDataset(table.items, settings, channels=1, manifest=manifest)[2]
    assert tuple(x.shape) == (1, 8, 8) and abs(float(x.mean())) < 0.5
    probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
    index = np.array([0, 1, 1, 1])
    mean = aggregate_study_probs(probs, index, 3, "mean_prob")
    np.testing.assert_allclose(mean[0], [0.8, 0.2])
    np.testing.assert_allclose(mean[1], [0.4, 0.6])
    assert np.isnan(mean[2]).all()                                                 # study without images
    peak = aggregate_study_probs(probs, index, 3, "max_prob")
    np.testing.assert_allclose(peak[1], [0.1, 0.9])
    three = aggregate_study_probs(np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]]), np.array([0, 0]), 1, "max_prob")
    np.testing.assert_allclose(three[0], np.array([0.6, 0.3, 0.6]) / 1.5)
    single = aggregate_study_probs(probs[:1], np.array([0]), 1, "mean_prob")
    np.testing.assert_allclose(single, probs[:1])                                  # 1 image == same path
