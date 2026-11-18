"""Geometry, operation order, subtraction math and channel layout of 3D volumes."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mribreast.imaging.geometry import (SeriesNotFoundError, load_series, parse_series_uids, resized_affine,
                                        series_path)
from mribreast.imaging.loader import StudyVolumeLoader
from mribreast.imaging.volume import (GeometryMismatchError, input_channels, pad_channels, pad_high,
                                      prepare_difference, prepare_phase, resize, subtraction_pairs,
                                      zscore_difference, zscore_phase)

AFFINE = np.array([[0.0, -0.8, 0.0, 10.0], [1.2, 0.0, 0.0, -5.0], [0.0, 0.0, 2.5, 3.0], [0, 0, 0, 1]])


def asymmetric_volume(shape=(10, 7, 5)) -> np.ndarray:
    """Recognisable pattern: value = 100*i + 10*j + k (+ a bright corner voxel)."""
    i, j, k = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    vol = (100.0 * i + 10.0 * j + k).astype(np.float32)
    vol[0, 0, 0] = 5000.0
    return vol


def save(path, data, affine=AFFINE):
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))
    return path


def test_series_keys_order_and_channels():
    assert parse_series_uids("a|b|c") == ("a", "b", "c")
    assert parse_series_uids(("a", "b")) == ("a", "b")
    assert parse_series_uids(np.array(["x", "y"])) == ("x", "y")
    assert subtraction_pairs(["pre", "p1", "p2", "p3"], "post_minus_pre") == [("p1", "pre"), ("p2", "pre"),
                                                                               ("p3", "pre")]
    assert subtraction_pairs(["pre", "p1", "p2"], "consecutive") == [("p1", "pre"), ("p2", "p1")]
    assert subtraction_pairs(["pre", "p1"], "none") == []
    with pytest.raises(ValueError):
        subtraction_pairs(["only"], "consecutive")
    assert input_channels("none", 5) == 5 and input_channels("post_minus_pre", 5) == 4
    assert input_channels("consecutive", 5) == 4


def test_series_path_and_invalid_uid(tmp_path):
    save(tmp_path / "1.2.3.nii.gz", np.zeros((2, 2, 2), np.float32))
    save(tmp_path / "4.5.nii", np.zeros((2, 2, 2), np.float32))
    assert series_path(tmp_path, "1.2.3").name == "1.2.3.nii.gz"
    assert series_path(tmp_path, "4.5").name == "4.5.nii"
    with pytest.raises(SeriesNotFoundError, match="9.9"):
        series_path(tmp_path, "9.9")
    for bad in ("", "..", "a/b", "a\\b", "x y"):
        with pytest.raises(ValueError):
            series_path(tmp_path, bad)


def test_axis_order_is_not_reoriented(tmp_path):
    vol = asymmetric_volume()
    data, affine = load_series(save(tmp_path / "s.nii.gz", vol))
    np.testing.assert_array_equal(data, vol)          # (i, j, k) kept, no flips/transposes
    np.testing.assert_allclose(affine, AFFINE)
    assert data.shape == (10, 7, 5)


def test_none_path_normalises_before_resize_and_matches_legacy_monai(tmp_path):
    monai_transforms = pytest.importorskip("monai.transforms")
    vol = asymmetric_volume()
    path = save(tmp_path / "s.nii.gz", vol)
    target = (4, 6, 3)
    prepared = prepare_phase(str(path), target)
    z = (vol - vol.mean()) / vol.std()
    expected = F.interpolate(torch.from_numpy(z)[None, None], size=target, mode="trilinear",
                             align_corners=False)[0, 0].numpy()
    np.testing.assert_allclose(prepared.data, expected, atol=1e-5)
    # resize-then-normalise gives a different result -> the order is observable
    r = F.interpolate(torch.from_numpy(vol)[None, None], size=target, mode="trilinear", align_corners=False)[0, 0]
    other = ((r - r.mean()) / r.std(unbiased=False)).numpy()
    assert np.abs(other - prepared.data).max() > 1e-2
    # legacy MONAI pipeline (NiftiDataset._monai_preprocess_and_save)
    out = monai_transforms.Compose([
        monai_transforms.LoadImaged(keys="img", image_only=False),
        monai_transforms.EnsureChannelFirstd(keys="img"),
        monai_transforms.NormalizeIntensityd(keys="img", nonzero=False, channel_wise=True),
        monai_transforms.ResizeD(keys="img", spatial_size=target, mode="trilinear", align_corners=False),
    ])({"img": str(path)})
    np.testing.assert_allclose(prepared.data, out["img"][0].numpy(), atol=1e-5)
    # our affine equals MONAI's MetaTensor affine (legacy saved the stale pre-resize affine)
    np.testing.assert_allclose(prepared.affine, np.asarray(out["img"].affine), atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["img_meta_dict"]["affine"]), AFFINE, atol=1e-6)


def test_resized_affine_keeps_world_position_of_volume_centre():
    affine = resized_affine(AFFINE, (10, 7, 5), (5, 14, 5))
    centre_in = AFFINE @ np.array([4.5, 3.0, 2.0, 1.0])
    centre_out = affine @ np.array([2.0, 6.5, 2.0, 1.0])
    np.testing.assert_allclose(centre_in, centre_out)


def test_zscore_semantics():
    x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    np.testing.assert_allclose(zscore_phase(x), (x - x.mean()) / x.std(), rtol=1e-6)      # ddof=0, no eps
    np.testing.assert_allclose(zscore_difference(x), (x - x.mean()) / (x.std() + 1e-5), rtol=1e-6)
    np.testing.assert_array_equal(zscore_phase(np.full(4, 3.0)), np.zeros(4))           # std==0 -> /1


def test_subtraction_math_and_geometry_check(tmp_path):
    rng = np.random.default_rng(0)
    pre = rng.normal(100, 10, (8, 6, 4)).astype(np.float32)
    post = pre + asymmetric_volume((8, 6, 4)) / 10
    a = save(tmp_path / "post.nii.gz", post)
    b = save(tmp_path / "pre.nii.gz", pre)
    out = prepare_difference(str(a), str(b), (4, 3, 2))
    ra = F.interpolate(torch.from_numpy(post)[None, None], size=(4, 3, 2), mode="trilinear", align_corners=False)
    rb = F.interpolate(torch.from_numpy(pre)[None, None], size=(4, 3, 2), mode="trilinear", align_corners=False)
    diff = (ra - rb)[0, 0]
    expected = ((diff - diff.mean()) / (diff.std(unbiased=False) + 1e-5)).numpy()
    np.testing.assert_allclose(out.data, expected, atol=1e-5)
    assert out.data.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.data.std() == pytest.approx(1.0, abs=1e-3)
    shifted = AFFINE.copy()
    shifted[0, 3] += 10
    c = save(tmp_path / "moved.nii.gz", pre, shifted)
    with pytest.raises(GeometryMismatchError):
        prepare_difference(str(a), str(c), (4, 3, 2))
    d = save(tmp_path / "small.nii.gz", pre[:-1])
    with pytest.raises(GeometryMismatchError):
        prepare_difference(str(a), str(d), None)


def test_loader_channels_padding_and_order(tmp_path):
    phases = [asymmetric_volume((6, 5, 4)) * (p + 1) + p for p in range(2)]
    for p, data in enumerate(phases):
        save(tmp_path / f"u{p}.nii.gz", data)
    loader = StudyVolumeLoader(tmp_path, subtraction_mode="none", channels=3, target_size=(3, 5, 4))
    study = loader.load(["u0", "u1"])
    assert study.data.shape == (3, 3, 5, 4)
    assert np.all(study.data[2] == 0)                       # missing phase -> zero channel appended at the end
    np.testing.assert_allclose(study.data[0], prepare_phase(str(tmp_path / "u0.nii.gz"), (3, 5, 4)).data)
    sub = StudyVolumeLoader(tmp_path, subtraction_mode="post_minus_pre", channels=2, target_size=None)
    s = sub.load(["u0", "u1"])
    assert s.data.shape == (2, 6, 5, 4) and s.channel_sources == ("u1-u0",)
    with pytest.raises(SeriesNotFoundError):
        loader.load(["u0", "does.not.exist"])


def test_pad_helpers_pad_high_side():
    x = np.ones((2, 2, 3, 1), np.float32)
    y = pad_high(x, (4, 3, 2))
    assert y.shape == (2, 4, 3, 2)
    assert y[:, :2, :3, :1].sum() == x.sum() and y.sum() == x.sum()
    assert pad_channels(np.ones((1, 2, 2)), 3)[1:].sum() == 0
    with pytest.raises(ValueError):
        pad_channels(np.ones((4, 2, 2)), 3)


def test_resize_nearest_keeps_integer_codes():
    mask = np.zeros((4, 4), np.int64)
    mask[1:3, 1:3] = 2
    out = resize(mask, (8, 8), mode="nearest")
    assert out.dtype == np.int64 and set(np.unique(out)) == {0, 2}
    assert out[2:6, 2:6].min() == 2 and out.sum() == 2 * 16
