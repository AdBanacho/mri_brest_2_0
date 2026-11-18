"""Mask providers, policies, joint transforms (alignment), audit and preview."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mribreast.contracts import Geometry, ImageKey, MaskProvider
from mribreast.imaging.datasets import ImageDataset, ItemBuilder, expected_channels
from mribreast.imaging.geometry import read_header_geometry
from mribreast.imaging.manifest import read_image_manifest
from mribreast.imaging.masks import (LocalFileMaskProvider, MaskPolicyError, SyntheticMaskProvider, apply_mask_mode,
                                     audit_frame, resolve_mask, summarize_mask_audit, write_mask_preview)
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import joint_crop, joint_flip, joint_resize, roi_bbox
from mribreast.synthetic.imaging import synthetic_studies, write_2d_dataset, write_masks, write_nifti_series


@pytest.fixture()
def data2d(tmp_path):
    studies = synthetic_studies(6)
    manifest = write_2d_dataset(tmp_path / "img", studies, (20, 16), 2, seed=0)
    masks = write_masks(tmp_path / "masks", manifest, seed=0, n_missing=1, n_empty=1)
    return studies, manifest, masks


def test_local_provider_2d_records_provenance(data2d):
    studies, manifest_path, mask_path = data2d
    provider = LocalFileMaskProvider(mask_path)
    assert isinstance(provider, MaskProvider)
    rows = pd.read_csv(mask_path, dtype=str, keep_default_na=False)
    first = rows.iloc[0]
    key = ImageKey(first.patient_id, first.study_id, first.series_id, first.slice_id)
    record = provider.get(key)
    assert record is not None and record.key == key and record.array.shape == (20, 16)
    assert record.geometry == Geometry(shape=(20, 16))
    assert record.class_codes == {0: "background", 1: "lesion"} and record.model_id == "synthetic-segmenter"
    assert record.fingerprint.startswith("sha256:") and record.training_provenance == "frozen_external"
    assert provider.get(ImageKey("nobody", "x", "y", "0")) is None
    described = provider.describe()
    assert described["n_masks"] == len(rows) and described["unknown_provenance"] is False
    images = read_image_manifest(manifest_path)
    assert len(rows) == len(images) - 1                                   # one deliberately missing


def test_unknown_provenance_and_fold_selection(tmp_path):
    mask = np.zeros((4, 4), np.uint8)
    mask[1:3, 1:3] = 1
    Image.fromarray(mask).save(tmp_path / "m.png")
    Image.fromarray(mask * 0).save(tmp_path / "m_f1.png")
    pd.DataFrame([
        {"patient_id": "p", "study_id": "s", "series_id": "x", "slice_id": "0", "path": "m.png"},
        {"patient_id": "p", "study_id": "s", "series_id": "x", "slice_id": "0", "path": "m_f1.png", "fold": "1",
         "source": "predicted"},
    ]).to_csv(tmp_path / "manifest.csv", index=False)
    provider = LocalFileMaskProvider(tmp_path / "manifest.csv")
    key = ImageKey("p", "s", "x", "0")
    assert provider.get(key, fold=1).training_provenance == "fold_trained:1"
    assert provider.get(key, fold=1).array.sum() == 0
    assert provider.get(key, fold=2).array.sum() == 4                     # other folds use the frozen mask
    assert provider.get(key).training_provenance == "unknown"
    info = provider.describe()
    assert info["unknown_provenance"] and info["limitations"] and info["folds"] == ["1"]
    assert LocalFileMaskProvider(tmp_path / "manifest.csv", source="predicted").get(key) is None


def test_nifti_masks_whole_volume_and_slice_extraction(tmp_path):
    studies = synthetic_studies(3, phases=2)
    write_nifti_series(tmp_path / "nifti", studies, (12, 10, 6), seed=0)
    path = write_masks(tmp_path / "masks", studies, image_root=tmp_path / "nifti", seed=0, n_missing=0, n_empty=0)
    provider = LocalFileMaskProvider(path, slice_axis=2)
    row = studies.iloc[0]
    uid = row.series_uids[0]
    volume = provider.get(ImageKey(row.patient_id, row.study_id, uid, ""))
    expected = read_header_geometry(tmp_path / "nifti" / f"{uid}.nii.gz")
    assert volume.geometry.matches(expected) and volume.geometry.orientation == expected.orientation
    k = int(np.argmax(volume.array.sum(axis=(0, 1))))
    sliced = provider.get(ImageKey(row.patient_id, row.study_id, uid, str(k)))
    np.testing.assert_array_equal(sliced.array, volume.array[:, :, k])
    assert sliced.geometry.shape == (12, 10)


def test_synthetic_provider_is_deterministic(tmp_path):
    studies = synthetic_studies(2, phases=2)
    write_nifti_series(tmp_path, studies, (12, 10, 6), seed=0)
    row = studies.iloc[0]
    key = ImageKey(row.patient_id, row.study_id, row.series_uids[0], "")
    a = SyntheticMaskProvider(5, image_root=tmp_path).get(key)
    b = SyntheticMaskProvider(5).with_image_sources(image_root=tmp_path).get(key)
    np.testing.assert_array_equal(a.array, b.array)
    assert a.array.any() and a.fingerprint == b.fingerprint
    c = SyntheticMaskProvider(6, image_root=tmp_path).get(key)
    assert not np.array_equal(a.array, c.array)
    sliced = SyntheticMaskProvider(5, image_root=tmp_path).get(ImageKey(*key.as_tuple()[:3], "2"))
    np.testing.assert_array_equal(sliced.array, a.array[:, :, 2])


def test_policies_missing_empty_invalid(data2d):
    studies, manifest_path, mask_path = data2d
    provider = LocalFileMaskProvider(mask_path)
    rows = pd.read_csv(mask_path, dtype=str, keep_default_na=False)
    images = read_image_manifest(manifest_path)
    present = set(map(tuple, rows[["study_id", "slice_id"]].values))
    missing_row = next(r for r in images.itertuples() if (r.study_id, r.slice_id) not in present)
    missing_key = ImageKey(missing_row.patient_id, missing_row.study_id, missing_row.series_id, missing_row.slice_id)
    common = dict(fold=None, expected=Geometry(shape=(20, 16)), roi_classes=(1,))
    with pytest.raises(MaskPolicyError):
        resolve_mask(provider, missing_key, missing_policy="error", empty_policy="full_image", **common)
    assert resolve_mask(provider, missing_key, missing_policy="drop", empty_policy="full_image",
                        **common).action == "drop"
    empty_key = None
    for r in rows.itertuples():
        key = ImageKey(r.patient_id, r.study_id, r.series_id, r.slice_id)
        if provider.get(key).array.sum() == 0:
            empty_key = key
    decision = resolve_mask(provider, empty_key, missing_policy="error", empty_policy="full_image", **common)
    assert (decision.status, decision.action) == ("empty", "full_image")
    ok_key = ImageKey(rows.iloc[0].patient_id, rows.iloc[0].study_id, rows.iloc[0].series_id, rows.iloc[0].slice_id)
    with pytest.raises(MaskPolicyError, match="invalid_geometry"):
        resolve_mask(provider, ok_key, fold=None, expected=Geometry(shape=(16, 20)), roi_classes=(1,),
                     missing_policy="error", empty_policy="error")
    with pytest.raises(MaskPolicyError, match="source_mismatch"):
        resolve_mask(provider, ok_key, missing_policy="error", empty_policy="error", mask_kind="manual", **common)


def test_builder_audit_cohort_change_with_drop(data2d):
    studies, manifest_path, mask_path = data2d
    manifest = read_image_manifest(manifest_path)
    settings = ImagingSettings(image_dim="2d", image_source="manifest_2d", image_manifest_file=str(manifest_path),
                               target_size_2d=(8, 8), mask_mode="foreground_only", mask_missing_policy="drop",
                               mask_empty_policy="drop", mask_kind="predicted")
    table = ItemBuilder(settings, LocalFileMaskProvider(mask_path), manifest).build(studies, None)
    audit = audit_frame(table.decisions)
    assert len(audit) == len(manifest) and (audit["action"] == "drop").sum() == 2
    assert len(table.items) == len(manifest) - 2
    summary = summarize_mask_audit(audit, len(studies), table.studies_with_items, len(manifest), len(table.items))
    assert int(summary.loc[0, "n_images_dropped"]) == 2 and int(summary.loc[0, "status_missing"]) == 1
    with pytest.raises(ValueError):
        ItemBuilder(settings, None, manifest)


def test_joint_transforms_keep_image_and_mask_aligned():
    image = np.zeros((2, 20, 30), np.float32)
    mask = np.zeros((20, 30), np.int64)
    image[:, 4:10, 6:18] = 1.0
    mask[4:10, 6:18] = 1
    log: list = []
    img, msk = joint_crop(image, mask, (2, 3), (16, 25), log)
    img, msk = joint_flip(img, msk, (1,), log)
    img, msk = joint_resize(img, msk, (28, 44), log)
    assert img.shape == (2, 28, 44) and msk.shape == (28, 44)
    assert set(np.unique(msk)) == {0, 1}                                  # nearest: codes never blended
    inside = msk == 1
    assert img[0][inside].mean() > 0.9 and img[0][~inside].mean() < 0.1
    assert np.mean((img[0] > 0.5) == inside) > 0.97
    assert [entry["op"] for entry in log] == ["crop", "flip", "resize"]
    assert log[2]["mask_interpolation"] == "nearest-exact"
    with pytest.raises(ValueError):
        joint_resize(image, mask[:-1], (4, 4))
    assert roi_bbox(mask, [1], 0.5) == ([1, 0], [13, 24])
    assert roi_bbox(mask * 0, [1], 0.5) is None


def test_mask_modes_shapes_and_full_image_fallback():
    image = np.random.default_rng(0).normal(size=(2, 16, 16)).astype(np.float32) + 5
    mask = np.zeros((16, 16), int)
    mask[4:8, 4:12] = 1
    assert apply_mask_mode(image, mask, "none", roi_classes=[1], margin=0.1, out_size=(8, 8)).shape == (2, 8, 8)
    channel = apply_mask_mode(image, mask, "mask_channel", roi_classes=[1], margin=0.1, out_size=(8, 8))
    assert channel.shape == (3, 8, 8) and set(np.unique(channel[2])) == {0.0, 1.0}
    fg = apply_mask_mode(image, mask, "foreground_only", roi_classes=[1], margin=0.1, out_size=(16, 16))
    assert np.all(fg[:, mask == 0] == 0) and np.all(fg[:, mask == 1] != 0)
    log: list = []
    crop = apply_mask_mode(image, mask, "roi_crop", roi_classes=[1], margin=0.25, out_size=(8, 8), log=log)
    assert crop.shape == (2, 8, 8) and log[0] == {"op": "crop", "starts": [3, 2], "ends": [9, 14]}
    fallback = apply_mask_mode(image, None, "mask_channel", roi_classes=[1], margin=0.1, out_size=(8, 8))
    assert fallback.shape == (3, 8, 8) and np.all(fallback[2] == 1)      # channel count preserved
    volume = np.ones((2, 8, 8, 4), np.float32)
    mask3 = np.zeros((8, 8, 4), int)
    mask3[2:4, 2:6, 1:3] = 1
    assert apply_mask_mode(volume, mask3, "roi_crop", roi_classes=[1], margin=0.0,
                           out_size=(8, 8, 4)).shape == (2, 8, 8, 4)


def test_dataset_applies_mask_channel_and_preview(tmp_path, data2d):
    studies, manifest_path, mask_path = data2d
    manifest = read_image_manifest(manifest_path)
    settings = ImagingSettings(image_dim="2d", image_source="manifest_2d", image_manifest_file=str(manifest_path),
                               target_size_2d=(10, 8), mask_mode="mask_channel", mask_missing_policy="full_image",
                               mask_kind="predicted")
    provider = LocalFileMaskProvider(mask_path)
    table = ItemBuilder(settings, provider, manifest).build(studies, None)
    channels = expected_channels(settings, manifest)
    assert channels == 2
    dataset = ImageDataset(table.items, settings, channels=channels, provider=provider, manifest=manifest)
    position = int(np.flatnonzero(table.items["mask_action"] == "use")[0])
    log: list = []
    x = dataset.load_item(position, log)
    assert x.shape == (2, 10, 8) and x[1].sum() > 0
    item = table.items.iloc[position]
    record = provider.get(ImageKey(item.patient_id, item.study_id, item.series_id, item.slice_id))
    png, meta = write_mask_preview(x[:1], x[1].astype(int), tmp_path / "preview" / "pair", transforms=log,
                                   key=record.key.as_tuple(), metadata={"mask_fingerprint": record.fingerprint})
    assert png.exists() and Image.open(png).size == (16, 10)
    payload = json.loads(meta.read_text())
    assert payload["transforms"][0]["op"] == "resize" and payload["mask_fingerprint"] == record.fingerprint


def test_nifti_mask_geometry_mismatch_is_invalid(tmp_path):
    studies = synthetic_studies(1, phases=2)
    write_nifti_series(tmp_path / "nifti", studies, (12, 10, 6), seed=0)
    row = studies.iloc[0]
    nib.save(nib.Nifti1Image(np.ones((12, 10, 6), np.uint8), np.eye(4)), str(tmp_path / "m.nii.gz"))
    pd.DataFrame([{"patient_id": row.patient_id, "study_id": row.study_id, "series_id": row.series_uids[0],
                   "slice_id": "", "path": "m.nii.gz"}]).to_csv(tmp_path / "masks.csv", index=False)
    settings = ImagingSettings(image_dim="3d", image_root=str(tmp_path / "nifti"), phase_count=2,
                               target_size=(6, 5, 3), mask_mode="mask_channel")
    with pytest.raises(MaskPolicyError, match="invalid_geometry"):
        ItemBuilder(settings, LocalFileMaskProvider(tmp_path / "masks.csv"), None).build(studies, None)
