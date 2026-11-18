"""Independent review: image/mask geometry (keys, joint transforms, slicing, 3D mask resize, boxes, affine)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mribreast.contracts import ImageKey
from mribreast.imaging.annotations import erase_background_box, resize_box
from mribreast.imaging.datasets import ImageDataset, ItemBuilder
from mribreast.imaging.geometry import resized_affine
from mribreast.imaging.masks.local import LocalFileMaskProvider
from mribreast.imaging.masks.policies import apply_mask_mode
from mribreast.imaging.settings import ImagingSettings
from mribreast.imaging.transforms import joint_resize, roi_bbox
from tests.fakes import import_legacy

AFFINE = np.diag([0.7, 0.9, 2.0, 1.0])


def _write(tmp_path, name, array, affine=AFFINE):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(array, affine), str(path))
    return path


def _study(series=("S0",)):
    return pd.DataFrame({"patient_id": ["P1"], "study_id": ["ST1"], "label": [1], "series_uids": ["|".join(series)]})


def _mask_manifest(tmp_path, rows):
    frame = pd.DataFrame(rows)
    path = tmp_path / "masks.csv"
    frame.to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------------- 3D mask resize before ROI use
@pytest.mark.parametrize("mode", ["foreground_only", "roi_crop", "mask_channel"])
def test_3d_small_lesion_survives_mask_mode(tmp_path, mode):
    """A lesion accepted as non-empty ('use') on the native grid must not vanish from the network input.

    datasets.py:316 resizes the mask to target_size with nearest-exact BEFORE roi_bbox / roi_array, so a
    lesion smaller than the downsampling stride disappears: foreground_only -> all-zero input, roi_crop ->
    silent full-image (no crop), mask_channel -> all-zero channel - while the audit says status=ok/use.
    """
    shape = (32, 24, 16)                                   # non-cubic
    image = np.random.default_rng(0).normal(100, 10, shape).astype(np.float32)
    image[1, 1, 1] = 1000.0
    _write(tmp_path, "S0.nii.gz", image)
    mask = np.zeros(shape, np.uint8)
    mask[1, 1, 1] = 1                                      # single lesion voxel near the low corner
    _write(tmp_path, "m.nii.gz", mask)
    manifest = _mask_manifest(tmp_path, [{"patient_id": "P1", "study_id": "ST1", "series_id": "S0", "slice_id": "",
                                          "path": "m.nii.gz"}])
    s = ImagingSettings(image_root=str(tmp_path), phase_count=1, target_size=(8, 6, 4), mask_mode=mode,
                        mask_empty_policy="error", roi_margin=0.0)
    provider = LocalFileMaskProvider(manifest)
    table = ItemBuilder(s, provider, None).build(_study(), fold=1)
    assert table.items["mask_action"].tolist() == ["use"]
    channels = 2 if mode == "mask_channel" else 1
    ds = ImageDataset(table.items, s, channels=channels, provider=provider, fold=1)
    log: list = []
    out = ds.load_item(0, log)
    if mode == "foreground_only":
        assert np.count_nonzero(out) > 0, "foreground_only produced an all-zero input for a 'use' mask"
    elif mode == "mask_channel":
        assert out[-1].sum() > 0, "mask channel is empty although the mask was accepted as non-empty"
    else:
        assert any(op.get("op") == "crop" for op in log), f"roi_crop silently skipped the crop: {log}"


# ----------------------------------------------------------------------------- 2D slices from NIfTI
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_2d_nifti_slice_image_and_mask_aligned_per_axis(tmp_path, axis):
    shape = (20, 28, 12)
    image = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.uint8)
    block = (slice(15, 19), slice(2, 6), slice(7, 10))     # asymmetric location
    image[block] = 500.0
    mask[block] = 1
    _write(tmp_path, "S0.nii.gz", image)
    _write(tmp_path, "m.nii.gz", mask)
    manifest = _mask_manifest(tmp_path, [{"patient_id": "P1", "study_id": "ST1", "series_id": "S0", "slice_id": "",
                                          "path": "m.nii.gz"}])
    s = ImagingSettings(image_dim="2d", image_root=str(tmp_path), phase_count=1, target_size_2d=(16, 10),
                        slice_axis=axis, slice_selection="mask_max_area", slices_per_study=1,
                        mask_mode="mask_channel")
    provider = LocalFileMaskProvider(manifest, slice_axis=axis)
    table = ItemBuilder(s, provider, None).build(_study(), fold=None)
    idx = int(table.items["slice_id"].iloc[0])
    assert block[axis].start <= idx < block[axis].stop
    out = ImageDataset(table.items, s, channels=2, provider=provider).load_item(0)
    img, roi = out[0], out[1]
    assert roi.sum() > 0
    peak = np.unravel_index(np.argmax(img), img.shape)
    assert roi[peak] == 1, f"image peak {peak} outside mask ROI on axis {axis}"
    # mask ROI equals thresholded image (same nearest grid) up to interpolation blur
    assert (img[roi == 1] > img[roi == 0].max(initial=0) * 0.5).mean() > 0.5


# ----------------------------------------------------------------------------- joint transforms, bbox edges
def test_roi_bbox_edges_margin_clipping_hw_asymmetric():
    H, W = 7, 13
    mask = np.zeros((H, W), np.int64)
    mask[H - 1, W - 1] = 1                                 # bottom-right corner pixel
    mask[H - 2, W - 3] = 1
    starts, ends = roi_bbox(mask, (1,), margin=0.5)
    assert starts == [H - 2 - 1, W - 3 - 2] and ends == [H, W]   # ceil(0.5*2)=1, ceil(0.5*3)=2, clipped high
    image = np.zeros((1, H, W), np.float32)
    image[0, H - 1, W - 1] = 9
    out = apply_mask_mode(image, mask, "roi_crop", roi_classes=(1,), margin=0.5, out_size=(3, 5))
    assert out.shape == (1, 3, 5)
    assert out[0, -1, -1] == out.max() > 0                 # corner pixel stays at the corner (no H/W swap)
    starts, ends = roi_bbox(np.pad(np.ones((1, 1)), ((0, 4), (0, 6))), (1,), 0.0)
    assert (starts, ends) == ([0, 0], [1, 1])              # exclusive end, no off-by-one


def test_joint_resize_hw_order_and_nearest_mask():
    image = np.zeros((2, 6, 10), np.float32)
    mask = np.zeros((6, 10), np.int64)
    image[:, 0:2, 7:10] = 1.0
    mask[0:2, 7:10] = 3
    img, msk = joint_resize(image, mask, (12, 5))
    assert img.shape == (2, 12, 5) and msk.shape == (12, 5)
    assert set(np.unique(msk)) <= {0, 3}                   # nearest: no blended codes
    assert msk[0, -1] == 3 and msk[-1, 0] == 0
    assert img[0][msk == 3].min() > img[0][msk == 0].mean()


# ----------------------------------------------------------------------------- mask key matching / folds
def test_mask_from_other_series_or_fold_never_used(tmp_path):
    shape = (8, 9, 4)
    for uid in ("A", "B"):
        _write(tmp_path, f"{uid}.nii.gz", np.ones(shape, np.float32))
    m = np.zeros(shape, np.uint8)
    m[2, 2, 2] = 1
    _write(tmp_path, "mB.nii.gz", m)
    _write(tmp_path, "mA2.nii.gz", m)
    manifest = _mask_manifest(tmp_path, [
        {"patient_id": "P1", "study_id": "ST1", "series_id": "B", "slice_id": "", "path": "mB.nii.gz", "fold": ""},
        {"patient_id": "P1", "study_id": "ST1", "series_id": "A", "slice_id": "", "path": "mA2.nii.gz", "fold": "2"},
        {"patient_id": "P1", "study_id": "OTHER", "series_id": "A", "slice_id": "", "path": "mB.nii.gz", "fold": ""},
    ])
    provider = LocalFileMaskProvider(manifest)
    key = ImageKey("P1", "ST1", "A", "")
    assert provider.get(key, 1) is None                    # fold-2 mask not used for fold 1
    assert provider.get(key, None) is None                 # nor fold-agnostically
    assert provider.get(key, 2) is not None
    s = ImagingSettings(image_root=str(tmp_path), phase_count=2, target_size=(8, 9, 4), mask_mode="roi_crop",
                        mask_missing_policy="drop")
    table = ItemBuilder(s, provider, None).build(_study(("A", "B")), fold=1)
    assert len(table.items) == 0 and table.decisions[0].status == "missing"


# ----------------------------------------------------------------------------- annotation boxes / affine
def test_resize_box_matches_legacy_and_erase_never_hits_lesion():
    import_legacy("threshold_tuning")          # puts the legacy repo on sys.path (or skips)
    try:
        import mriBreastDuke.dataLoaders.annotation_boxes as legacy
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"legacy annotation_boxes not importable: {exc}")
    rng = np.random.default_rng(0)
    for _ in range(300):
        src = tuple(int(v) for v in rng.integers(8, 120, 3))
        dst = tuple(int(v) for v in rng.integers(4, 64, 3))
        a = [int(rng.integers(0, s)) for s in src]
        b = [int(rng.integers(x + 1, s + 1)) for x, s in zip(a, src)]
        box = (tuple(a), tuple(b))
        lesion = resize_box(box, src, dst)
        assert lesion == legacy.resize_box(box, src, dst)
        vol = np.ones((2,) + dst, np.float32)
        out, origin = erase_background_box(vol, lesion, rng, 8)
        region = (slice(None),) + tuple(slice(lo, hi) for lo, hi in zip(*lesion))
        assert (out[region] == 1).all()                    # lesion voxels never erased
        # the resized box covers every voxel whose centre maps into the source box
        for ax in range(3):
            centres = (np.arange(dst[ax]) + 0.5) * src[ax] / dst[ax]
            inside = np.flatnonzero((centres >= a[ax]) & (centres < b[ax]))
            if len(inside):
                assert lesion[0][ax] <= inside[0] and inside[-1] < lesion[1][ax]


def test_resized_affine_maps_voxel_centres():
    A = np.array([[0.0, -0.8, 0, 10], [0.9, 0, 0, -5], [0, 0, 2.5, 3], [0, 0, 0, 1]])
    in_shape, out_shape = (40, 30, 12), (16, 24, 5)
    B = resized_affine(A, in_shape, out_shape)
    for o in ([0, 0, 0], [15, 23, 4], [7, 3, 2]):
        src = (np.array(o) + 0.5) * np.array(in_shape) / np.array(out_shape) - 0.5
        np.testing.assert_allclose(B @ np.r_[o, 1], A @ np.r_[src, 1])


# ----------------------------------------------------------------------------- cohort audit with fold-trained masks
def test_cohort_mask_audit_with_fold_trained_masks_only(tmp_path):
    """cli._image_audits -> ImageBranch.audit builds items with fold=None (image_branch.py:178).

    With a mask manifest that only holds fold-trained rows (fold column set) every study is 'missing'
    at fold=None: with the default mask_missing_policy=error, `train` aborts before any fold although
    every fold has its mask; with drop/full_image the written mask_audit.csv reports 100 % missing.
    """
    from mribreast.models.image_branch import ImageBranch

    shape = (8, 9, 4)
    _write(tmp_path, "A.nii.gz", np.ones(shape, np.float32))
    m = np.zeros(shape, np.uint8)
    m[2, 2, 2] = 1
    _write(tmp_path, "m.nii.gz", m)
    manifest = _mask_manifest(tmp_path, [
        {"patient_id": "P1", "study_id": "ST1", "series_id": "A", "slice_id": "", "path": "m.nii.gz", "fold": str(k)}
        for k in (1, 2, 3)])
    cfg = {"image_root": str(tmp_path), "phase_count": 1, "target_size": (8, 9, 4), "mask_mode": "roi_crop",
           "mask_missing_policy": "error"}
    branch = ImageBranch(cfg, num_classes=2, mask_provider=LocalFileMaskProvider(manifest))
    audits = branch.audit(_study(("A",)))                  # raises MaskPolicyError today
    assert audits["mask_audit"]["status"].tolist() != ["missing"]


@pytest.mark.parametrize("suffix", [".png", ".tif"])
@pytest.mark.parametrize("mode", ["roi_crop", "mask_channel", "foreground_only"])
def test_manifest_2d_png_tiff_mask_alignment_hw_asymmetric(tmp_path, suffix, mode):
    import tifffile
    from PIL import Image

    H, W = 18, 30
    img = np.full((H, W), 10, np.uint8)
    msk = np.zeros((H, W), np.uint8)
    img[13:17, 1:5] = 250                                  # lesion bottom-left
    msk[13:17, 1:5] = 1
    img[0, W - 1] = 200                                    # decoy outside the lesion
    for name, arr in (("i", img), ("m", msk)):
        if suffix == ".png":
            Image.fromarray(arr).save(tmp_path / f"{name}{suffix}")
        else:
            tifffile.imwrite(str(tmp_path / f"{name}{suffix}"), arr)
    key = {"patient_id": "P1", "study_id": "ST1", "series_id": "X", "slice_id": "0"}
    pd.DataFrame([{**key, "path": f"i{suffix}"}]).to_csv(tmp_path / "images.csv", index=False)
    manifest = _mask_manifest(tmp_path, [{**key, "path": f"m{suffix}"}])
    s = ImagingSettings(image_dim="2d", image_source="manifest_2d", image_manifest_file=str(tmp_path / "images.csv"),
                        target_size_2d=(9, 15), mask_mode=mode, roi_margin=0.0)
    provider = LocalFileMaskProvider(manifest)
    from mribreast.imaging.datasets import load_manifest_for

    m = load_manifest_for(s)
    table = ItemBuilder(s, provider, m).build(_study(), fold=None)
    assert table.items["mask_action"].tolist() == ["use"]
    out = ImageDataset(table.items, s, channels=2 if mode == "mask_channel" else 1, provider=provider,
                       manifest=m).load_item(0)
    assert out.shape[1:] == (9, 15)
    if mode == "roi_crop":
        assert out[0].std() < 1e-3 or out[0].min() > 0     # crop contains only lesion pixels
    elif mode == "mask_channel":
        peak = np.unravel_index(np.argmax(out[0]), out[0].shape)
        assert out[1][peak] == 1 and out[1][:, -3:].sum() == 0
    else:
        assert out[0][:4].max() == 0                        # decoy row zeroed
        assert out[0][-3:, :3].max() > 0


def test_2d_subtraction_mask_check_uses_minuend_affine(tmp_path):
    """Builder validates the mask against series[0] (datasets.py:118); load_item re-validates against
    volume.affine, which for subtraction is the MINUEND (series[1]) grid (loader.py:96,
    datasets.py:332). Phases accepted by same_grid (atol 1e-3) but differing by > Geometry.matches atol
    1e-4 pass item building and then crash with MaskPolicyError during training/inference.
    """
    shape = (10, 12, 6)
    a0 = AFFINE.copy()
    a1 = AFFINE.copy()
    a1[0, 3] += 5e-4                                       # sub-micron header jitter between phases
    rng = np.random.default_rng(0)
    _write(tmp_path, "S0.nii.gz", rng.normal(100, 5, shape).astype(np.float32), a0)
    _write(tmp_path, "S1.nii.gz", rng.normal(120, 5, shape).astype(np.float32), a1)
    m = np.zeros(shape, np.uint8)
    m[3:6, 4:8, 2:4] = 1
    _write(tmp_path, "m.nii.gz", m, a0)
    manifest = _mask_manifest(tmp_path, [{"patient_id": "P1", "study_id": "ST1", "series_id": "S0", "slice_id": "",
                                          "path": "m.nii.gz"}])
    s = ImagingSettings(image_dim="2d", image_root=str(tmp_path), phase_count=2, subtraction_mode="post_minus_pre",
                        target_size_2d=(8, 8), slice_selection="mask_max_area", mask_mode="mask_channel")
    provider = LocalFileMaskProvider(manifest)
    table = ItemBuilder(s, provider, None).build(_study(("S0", "S1")), fold=None)
    assert table.items["mask_action"].tolist() == ["use"]
    out = ImageDataset(table.items, s, channels=2, provider=provider).load_item(0)   # raises today
    assert out.shape == (2, 8, 8)
