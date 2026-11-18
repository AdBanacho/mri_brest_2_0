"""mribreast.legacy.names: legacy experiment names (build exactly, parse back)."""

from __future__ import annotations

import pytest

from mribreast.legacy import names

CFG = {"mri_model": "resnet18", "subtraction_mode": "post_minus_pre", "feature_groups": ("clinical", "kinetic"),
       "feature_model": "mlp", "feature_selector": "lasso", "lasso_cv_folds": 5, "lasso_cs": 20,
       "lasso_min_features": 1, "threshold_calibration_folds": 5, "lr": 1e-4, "sensitivity_lambda": 0.2,
       "positive_boost": 1.0, "batch_size": 4, "use_annotation_boxes": True, "experiment_id": "cfg0123456789ab"}


def test_build_matches_legacy_format():
    assert names.build_experiment_name(CFG) == (
        "ImagingFeaturesFusion_resnet18_post_minus_pre_clinical-kinetic_mlp_lasso-cv5-cs20-min1_clinical-safe-v2_"
        "auc-cal5_lr1e-04_sens0.2_boost1_bs4_annboxes_cfg0123456789ab")
    plain = {**CFG, "feature_groups": "kinetic,kinetic", "feature_selector": "none", "use_annotation_boxes": "false",
             "experiment_id": "", "subtraction_mode": "none", "lr": 3e-4}
    assert names.build_experiment_name(plain) == (
        "ImagingFeaturesFusion_resnet18_none_kinetic_mlp_auc-cal5_lr3e-04_sens0.2_boost1_bs4")
    assert names.legacy_run_id(plain, 0.25) .endswith("/fusion_alpha_0.25")


def test_lr_one_significant_digit_collision_is_reproduced():
    a = names.build_experiment_name({**CFG, "lr": 1.6e-4})
    b = names.build_experiment_name({**CFG, "lr": 2e-4})
    assert a == b  # documented legacy collision (docs/migration.md)


@pytest.mark.parametrize("cfg", [CFG, {**CFG, "feature_selector": "none", "use_annotation_boxes": False,
                                       "experiment_id": "", "feature_groups": ("morphology",)}])
def test_parse_roundtrip(cfg):
    parsed = names.parse_experiment_name(names.build_experiment_name(cfg))
    assert parsed["workflow"] == "fusion"
    for key in ("mri_model", "subtraction_mode", "feature_model", "feature_selector", "batch_size",
                "threshold_calibration_folds", "use_annotation_boxes", "experiment_id"):
        assert parsed[key] == cfg[key], key
    assert parsed["feature_groups"] == ",".join(cfg["feature_groups"])
    assert parsed["clinical_safe_v2"] is ("clinical" in cfg["feature_groups"] or None)


def test_stored_era_name_flags_pre_safe_v2():
    stored = ("ImagingFeaturesFusion_densenet121_none_clinical_xgboost_lasso-cv5-cs20-min1_auc-cal5_lr1e-04_"
              "sens0.05_boost1_bs4")
    parsed = names.parse_experiment_name(stored)
    assert parsed["clinical_safe_v2"] is False and parsed["experiment_id"] == ""


def test_historical_image_names():
    name = names.build_historical_image_name(1, 1e-3, 0.3, 2.0, 8, "consecutive", True)
    assert name == "DenseNet_lr_1e-03_sens_0.3_posboost_2.0_bs_8_sub_consecutive_annboxes"
    parsed = names.parse_experiment_name(name)
    assert parsed["workflow"] == "historical_image" and parsed["mri_model"] == "densenet121"
    assert parsed["inner_calibration"] is False and parsed["subtraction_mode"] == "consecutive"
    wd = names.build_historical_image_name("ResNet10", 1e-5, 1.0, 3.0, 8, weight_decay=1e-4)
    assert names.parse_experiment_name(wd)["weight_decay"] == 1e-4
    assert names.parse_experiment_name("something-else")["workflow"] == "unknown"


def test_split_run_id():
    assert names.split_run_id("A_b/fusion_alpha_0.5") == ("A_b", 0.5)
    assert names.split_run_id("A_b") == ("A_b", None)
