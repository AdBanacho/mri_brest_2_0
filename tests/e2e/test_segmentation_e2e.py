"""Synthetic end to end: box-prompted segmentation (fake segmenter) -> train -> validate -> summarize.

The masks enter the classifier only through ``mask_provider=local_files`` (docs/segmentation-mri-core.md).
One 3D config (mask roi_crop) and one 2D-from-NIfTI config (mask-based slice selection + mask_channel).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mribreast.cli import main
from mribreast.imaging.annotations import load_annotation_boxes
from mribreast.prepare.segmentation import pipeline
from mribreast.synthetic.dataset import make_synthetic_dataset
from tests.fakes import FakeSliceSegmenter

pytestmark = pytest.mark.slow

COMMON = ["--num_folds", "3", "--threshold_calibration_folds", "2", "--epoch", "1", "--batch_size", "4",
          "--num_workers", "0", "--target_size", "16", "16", "8", "--target_size_2d", "16", "16",
          "--xgb_n_estimators", "10", "--xgb_n_jobs", "1", "--lasso_n_jobs", "1", "--branches", "image,tabular",
          "--feature_groups", "clinical,kinetic", "--mask_provider", "local_files", "--mask_kind", "predicted",
          "--mask_provenance", "frozen_external", "--mask_missing_policy", "full_image", "--profile", "general"]
MODES = {
    "3d_roi_crop": ["--image_dim", "3d", "--mask_mode", "roi_crop", "--mri_model", "resnet10"],
    "2d_mask_channel": ["--image_dim", "2d", "--image_source", "nifti", "--slice_selection", "mask_max_area",
                        "--slices_per_study", "2", "--mask_mode", "mask_channel", "--mri_model", "resnet18"],
}


@pytest.fixture(scope="module")
def segmented(tmp_path_factory):
    root = tmp_path_factory.mktemp("segE2E")
    info = make_synthetic_dataset(root / "data", n_patients=60, seed=3, shape=(16, 16, 8), size_2d=(16, 16),
                                  images_per_study=1)
    result = pipeline.run_nifti(info["studies"], load_annotation_boxes(info["annotation_boxes_file"]),
                                info["image_root"], FakeSliceSegmenter(), root / "seg",
                                pipeline.SegmentSettings(min_window=8), log=lambda _m: None)
    assert result["summary"]["n_errors"] == 0 and result["summary"]["n_segmented"] > 0
    return root, info, root / "seg" / "mask_manifest.csv"


@pytest.mark.parametrize("mode", list(MODES))
def test_segmentation_masks_drive_training(segmented, mode):
    root, info, mask_manifest = segmented
    argv = []
    for key in ("clinical_features_file", "metadata_file", "imaging_features_file", "annotation_boxes_file",
                "image_root"):
        argv += [f"--{key}", str(info[key])]
    argv += ["--mask_manifest_file", str(mask_manifest)]
    for key in ("prepared_root", "subtraction_root", "logs_root", "checkpoint_root", "validation_root", "summary_root"):
        argv += [f"--{key}", str(root / mode / key)]
    argv += COMMON + MODES[mode]
    assert main(["train", *argv]) == 0
    assert main(["validate", *argv, "--validate_num_workers", "0"]) == 0
    assert main(["summarize", "--validation_root", str(root / mode / "validation_root"),
                 "--summary_root", str(root / mode / "summary_root"), "--expected_folds", "3"]) == 0

    metas = sorted((root / mode / "checkpoint_root").rglob("image_branch.json"))
    assert len(metas) == 3
    for path in metas:
        meta = json.loads(Path(path).read_text())
        assert meta["mask"]["requires_box_at_inference"] is True
        assert meta["mask"]["provider"]["prompt_sources"] == {"annotation_box": int(len(pd.read_csv(mask_manifest)))}
        audits = list(Path(path).parent.glob("mask_audit*.csv"))
        assert audits
        audit = pd.concat(pd.read_csv(a, dtype=str, keep_default_na=False) for a in audits)
        used = audit[audit["action"] == "use"]
        assert len(used) and set(used["prompt_source"]) == {"annotation_box"}
    evals = list((root / mode / "validation_root").glob("tr-*/ev-*/COMPLETE"))
    assert len(evals) == 1
    metrics = pd.read_csv(evals[0].parent / "validation_metrics.csv")
    assert len(metrics) == 3 and metrics["fusion_auc_roc"].notna().any()
