"""Box-prompted segmentation stage (docs/segmentation-mri-core.md): geometry, pipeline, manifest, resume, CLI.

Everything here runs without torch weights or a GPU, using a fake segmenter. The real-model checks are in
tests/unit/test_segmentation_mri_core.py and skip when the weights or the ``.[segment]`` extras are missing.
"""

from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from mribreast.contracts import ImageKey
from mribreast.imaging.geometry import read_header_geometry, series_path
from mribreast.imaging.masks.local import LocalFileMaskProvider, read_mask_manifest
from mribreast.imaging.masks.policies import AUDIT_COLUMNS
from mribreast.prepare.segmentation import geometry as g
from mribreast.prepare.segmentation import pipeline
from mribreast.prepare.segmentation.mri_core import SliceResult, dino_backbone_to_sam
from tests.fakes import FakeSliceSegmenter as FakeSegmenter

BREAST_MRI_207 = ((358, 374, 34), (365, 387, 40))   # half-open (column, row, slice), docs/imaging.md §4


# ------------------------------------------------------------------------------------------------ geometry
def test_breast_mri_207_box_maps_to_sam_xy():
    assert g.plane_axes(2) == (0, 1)
    assert list(g.slice_range(BREAST_MRI_207, 2, 60)) == [34, 35, 36, 37, 38, 39]
    box2d = g.box_in_plane(BREAST_MRI_207, 2)
    assert box2d == (358, 365, 374, 387)                    # H = column axis, W = row axis
    assert g.box_xyxy(box2d) == (374.0, 358.0, 387.0, 365.0)  # x along W (rows), y along H (columns)


def test_slice_range_margin_is_clipped_and_validated():
    box = ((1, 1, 0), (3, 3, 2))
    assert list(g.slice_range(box, 2, 4, margin=1)) == [0, 1, 2]
    with pytest.raises(ValueError):
        g.slice_range(((0, 0, 3), (1, 1, 6)), 2, 5)


def test_context_window_is_square_centred_and_shifted_inside():
    w = g.context_window((40, 50, 60, 66), (512, 512), context=3.0, min_side=8)
    assert w.side == 30 and (w.h0, w.w0) == (30, 48)
    edge = g.context_window((0, 4, 0, 4), (100, 100), context=3.0, min_side=64)
    assert (edge.h0, edge.w0, edge.side) == (0, 0, 64)        # shifted, not padded
    big = g.context_window((2, 4, 2, 4), (10, 20), context=1.0, min_side=32)
    assert big.side == 32 and big.h0 == -11 and big.w0 == -6  # larger than the slice: slice centred


def test_extract_and_paste_round_trip_with_padding():
    plane = np.arange(10 * 12, dtype=np.float32).reshape(10, 12)
    window = g.context_window((3, 5, 4, 7), plane.shape, context=1.0, min_side=16)
    crop = g.extract_window(plane, window, fill=-1.0)
    assert crop.shape == (16, 16) and (crop == -1).any()
    target = np.zeros(plane.shape, dtype=bool)
    g.paste_window(target, crop >= 0, window)
    assert target.all()
    inner = window.box_in_window((3, 5, 4, 7))
    h0, h1, w0, w1 = inner
    assert np.array_equal(crop[h0:h1, w0:w1], plane[3:5, 4:7])
    assert g.scale_box(inner, 16, 1024) == tuple(v * 64.0 for v in g.box_xyxy(inner))


def test_clip_and_expand_box():
    mask = np.ones((6, 6), dtype=bool)
    clipped, removed = g.clip_to_box(mask, (1, 3, 2, 5))
    assert clipped.sum() == 6 and removed == 30
    assert g.expand_box2d((10, 20, 10, 14), (24, 100), 0.1) == (9, 21, 9, 15)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_segment_volume_recovers_an_asymmetric_lesion_on_every_slice_axis(axis):
    rng = np.random.default_rng(0)
    volume = rng.normal(0, 1, (30, 26, 22)).astype(np.float32)
    box = ((6, 10, 4), (11, 21, 9))                         # extents 5 x 11 x 5: a transposition would show
    volume[6:11, 10:21, 4:9] += 50.0
    settings = pipeline.SegmentSettings(slice_axis=axis, min_window=8)
    mask, qc, best = pipeline.segment_volume(volume, box, settings, FakeSegmenter())
    expected = np.zeros(volume.shape, dtype=np.uint8)
    expected[6:11, 10:21, 4:9] = 1
    assert np.array_equal(mask, expected)
    assert qc["mask_voxels"] == qc["box_voxels"] == 5 * 11 * 5 and qc["mask_box_fraction"] == 1.0
    assert qc["n_slices_prompted"] == qc["n_slices_nonempty"] == box[1][axis] - box[0][axis]
    assert box[0][axis] <= best < box[1][axis]


def test_clip_to_box_removes_outside_voxels_and_reports_it():
    class Everything(FakeSegmenter):
        def segment(self, windows, boxes):
            return [SliceResult(np.ones(w.shape, np.float32), 0.5) for w in windows]

    volume = np.zeros((20, 20, 4), np.float32)
    box = ((5, 5, 1), (10, 10, 3))
    clipped, qc, _ = pipeline.segment_volume(volume, box, pipeline.SegmentSettings(box_margin=0.0, min_window=8),
                                             Everything())
    assert clipped.sum() == 5 * 5 * 2 and qc["frac_removed_by_clip"] > 0.5
    unclipped, qc2, _ = pipeline.segment_volume(volume, box, pipeline.SegmentSettings(clip_to_box=False,
                                                                                       min_window=8), Everything())
    assert unclipped.sum() > clipped.sum() and qc2["frac_removed_by_clip"] == 0.0


def test_settings_validation():
    with pytest.raises(ValueError):
        pipeline.SegmentSettings(input="post_minus_pre:0")
    with pytest.raises(ValueError):
        pipeline.SegmentSettings(input="subtraction:1")
    with pytest.raises(ValueError):
        pipeline.SegmentSettings(context=0.5)
    assert pipeline.parse_input("phase:0") == ("phase", 0)


def test_dino_backbone_mapping_handles_chunked_and_plain_blocks():
    state = {"teacher": {"backbone.blocks.0.1.mlp.fc1.weight": 1, "backbone.blocks.3.attn.qkv.bias": 2,
                         "backbone.patch_embed.proj.weight": 3, "backbone.pos_embed": 4, "dino_head.mlp.0.weight": 5,
                         "backbone.norm.weight": 6}}
    assert dino_backbone_to_sam(state) == {"image_encoder.blocks.1.mlp.lin1.weight": 1,
                                           "image_encoder.blocks.3.attn.qkv.bias": 2,
                                           "image_encoder.patch_embed.proj.weight": 3}


# ------------------------------------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from mribreast.synthetic.dataset import make_synthetic_dataset

    root = tmp_path_factory.mktemp("segsyn")
    info = make_synthetic_dataset(root / "data", n_patients=12, seed=5, shape=(20, 18, 8), size_2d=(24, 24),
                                  images_per_study=2)
    studies = info["studies"]
    cohort = root / "cohort.csv"
    studies[["patient_id", "study_id", "series_uids"]].to_csv(cohort, index=False)
    return info, studies, cohort


def _run(info, studies, out, **kw):
    from mribreast.imaging.annotations import load_annotation_boxes

    return pipeline.run_nifti(studies, load_annotation_boxes(info["annotation_boxes_file"]), info["image_root"],
                              kw.pop("segmenter", FakeSegmenter()), out, kw.pop("settings", pipeline.SegmentSettings()),
                              log=lambda _m: None, **kw)


def test_run_nifti_writes_a_valid_mask_manifest(synthetic, tmp_path):
    info, studies, _ = synthetic
    result = _run(info, studies, tmp_path / "seg", previews=2)
    assert (tmp_path / "seg" / "COMPLETE").is_file()
    manifest = read_mask_manifest(tmp_path / "seg" / "mask_manifest.csv")
    assert set(manifest["source"]) == {"predicted"} and set(manifest["prompt_source"]) == {"annotation_box"}
    assert set(manifest["training_provenance"]) == {"frozen_external"} and set(manifest["slice_id"]) == {""}
    assert result["summary"]["n_segmented"] == len(manifest) > 0
    provider = LocalFileMaskProvider(tmp_path / "seg" / "mask_manifest.csv")
    for row in manifest.itertuples(index=False):
        key = ImageKey(row.patient_id, row.study_id, row.series_id, "")
        record = provider.get(key)
        assert record.prompt_source == "annotation_box" and record.model_id == "fake"
        assert record.geometry.matches(read_header_geometry(series_path(info["image_root"], row.series_id)))
        assert record.array.max() == 1
        sliced = provider.get(ImageKey(row.patient_id, row.study_id, row.series_id, "3"))
        assert sliced is not None and sliced.array.ndim == 2
    described = provider.describe()
    assert described["requires_box_at_inference"] is True
    assert any("annotation boxes" in text for text in described["limitations"])
    assert "prompt_source" in AUDIT_COLUMNS
    qc = pd.read_csv(tmp_path / "seg" / "segmentation_qc.csv")
    assert len(qc) == len(studies) and (qc.loc[qc.status == "segmented", "frac_removed_by_clip"] < 0.5).all()
    assert len(list((tmp_path / "seg" / "previews").glob("*.png"))) == 2
    meta = json.loads((tmp_path / "seg" / "segmentation_manifest.json").read_text())
    assert meta["settings_fingerprint"] == pipeline.settings_fingerprint(pipeline.SegmentSettings(),
                                                                         FakeSegmenter().describe())
    assert any("box is needed" in text for text in meta["limitations"])


def test_rerun_reuses_masks_and_refuses_other_settings(synthetic, tmp_path):
    info, studies, _ = synthetic
    out = tmp_path / "seg"
    _run(info, studies, out)
    masks = sorted(out.glob("masks/*.nii.gz"))
    stamps = {p: os.stat(p).st_mtime_ns for p in masks}
    second = FakeSegmenter()
    _run(info, studies, out, segmenter=second)
    assert second.calls == [] and {p: os.stat(p).st_mtime_ns for p in masks} == stamps
    assert pd.read_csv(out / "segmentation_qc.csv").query("status == 'segmented'")["reused"].all()
    other = pipeline.SegmentSettings(context=4.0)
    with pytest.raises(FileExistsError):
        _run(info, studies, out, settings=other)
    _run(info, studies, out, settings=other, overwrite=True)
    assert json.loads((out / "segmentation_manifest.json").read_text())["settings"]["context"] == 4.0


def test_failed_study_is_reported_and_retried(synthetic, tmp_path):
    info, studies, _ = synthetic
    out = tmp_path / "seg"
    result = _run(info, studies, out, settings=pipeline.SegmentSettings(input="phase:9"))
    assert result["summary"]["n_errors"] > 0 and not (out / "COMPLETE").exists()
    assert read_mask_manifest(out / "mask_manifest.csv").empty


def test_missing_box_gets_no_manifest_row(synthetic, tmp_path):
    from mribreast.imaging.annotations import load_annotation_boxes

    info, studies, _ = synthetic
    boxes = load_annotation_boxes(info["annotation_boxes_file"])
    dropped = str(studies.iloc[0]["patient_id"])
    boxes.pop(dropped, None)
    pipeline.run_nifti(studies, boxes, info["image_root"], FakeSegmenter(), tmp_path / "seg", log=lambda _m: None)
    qc = pd.read_csv(tmp_path / "seg" / "segmentation_qc.csv")
    assert qc.loc[qc.patient_id == dropped, "status"].tolist() == ["missing_box"]
    assert dropped not in set(read_mask_manifest(tmp_path / "seg" / "mask_manifest.csv")["patient_id"])


def test_reference_masks_give_dice(synthetic, tmp_path):
    info, studies, _ = synthetic
    _run(info, studies, tmp_path / "a")
    reference = LocalFileMaskProvider(tmp_path / "a" / "mask_manifest.csv")
    result = _run(info, studies, tmp_path / "b", reference=reference)
    assert result["summary"]["dice"]["median"] == pytest.approx(1.0)


def test_ready_2d_images_with_lesion_box(synthetic, tmp_path):
    info, _, _ = synthetic
    result = pipeline.run_images_2d(info["image_manifest_file"], FakeSegmenter(), tmp_path / "seg2d",
                                    pipeline.SegmentSettings(min_window=8), log=lambda _m: None)
    manifest = read_mask_manifest(tmp_path / "seg2d" / "mask_manifest.csv")
    images = pd.read_csv(info["image_manifest_file"], dtype=str)
    assert result["summary"]["n_segmented"] == len(manifest) == len(images)
    provider = LocalFileMaskProvider(tmp_path / "seg2d" / "mask_manifest.csv")
    row = images.iloc[0]
    record = provider.get(ImageKey(row.patient_id, row.study_id, row.series_id, row.slice_id))
    h0, h1, w0, w1 = json.loads(row.lesion_box)
    assert record.array.shape == (24, 24) and record.array[h0:h1, w0:w1].mean() > 0.5
    assert record.array.sum() <= (h1 - h0 + 2) * (w1 - w0 + 2)


def test_ready_2d_without_lesion_box_column_is_refused(tmp_path):
    manifest = tmp_path / "m.csv"
    pd.DataFrame([{"patient_id": "p", "study_id": "s", "series_id": "x", "slice_id": "0", "path": "a.png"}]).to_csv(
        manifest, index=False)
    with pytest.raises(ValueError, match="lesion_box"):
        pipeline.run_images_2d(manifest, FakeSegmenter(), tmp_path / "out")


def test_cli_runs_with_injected_segmenter(synthetic, tmp_path, monkeypatch):
    from mribreast.cli import main
    from mribreast.prepare import cli as prepare_cli

    info, _, cohort = synthetic
    monkeypatch.setattr(prepare_cli, "make_segmenter", lambda args: FakeSegmenter())
    out = tmp_path / "seg"
    code = main(["prepare-data", "segment-mri-core", "--cohort_csv", str(cohort), "--annotation_boxes_file",
                 str(info["annotation_boxes_file"]), "--image_root", str(info["image_root"]), "--output_dir", str(out),
                 "--mri_foundation_repo", str(tmp_path), "--limit", "3"])
    assert code == 0 and len(pd.read_csv(out / "segmentation_qc.csv")) == 3
    # cohort built by the stages' own code (prepare would already need the mask manifest this produces)
    out2 = tmp_path / "seg_tables"
    code = main(["prepare-data", "segment-mri-core", "--clinical_features_file", str(info["clinical_features_file"]),
                 "--metadata_file", str(info["metadata_file"]), "--cohort_option", "profile=general",
                 "--cohort_option", "phase_order=explicit", "--annotation_boxes_file", str(info["annotation_boxes_file"]),
                 "--image_root", str(info["image_root"]), "--output_dir", str(out2), "--mri_foundation_repo",
                 str(tmp_path)])
    assert code == 0
    cohort_used = pd.read_csv(out2 / "segmentation_cohort.csv", dtype=str)
    meta = json.loads((out2 / "segmentation_manifest.json").read_text())
    assert meta["inputs"]["cohort_from"]["profile"] == "general" and len(cohort_used) > 0
    assert set(read_mask_manifest(out2 / "mask_manifest.csv")["study_id"]) <= set(cohort_used["study_id"])
    with pytest.raises(SystemExit):
        main(["prepare-data", "segment-mri-core", "--cohort_csv", str(cohort), "--metadata_file", "x",
              "--output_dir", str(out), "--mri_foundation_repo", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["prepare-data", "segment-mri-core", "--output_dir", str(out), "--mri_foundation_repo", str(tmp_path)])


def test_written_nifti_is_uint8_on_first_series_grid(synthetic, tmp_path):
    info, studies, _ = synthetic
    _run(info, studies, tmp_path / "seg")
    row = read_mask_manifest(tmp_path / "seg" / "mask_manifest.csv").iloc[0]
    image = nib.load(row.path)
    ref = nib.load(str(series_path(info["image_root"], row.series_id)))
    assert image.get_data_dtype() == np.uint8 and image.shape == ref.shape
    assert np.allclose(image.affine, ref.affine)
