"""Field registry, parsing, combination rules and identity rules."""

import pytest

from mribreast.config import schema
from mribreast.config.schema import ConfigError


def test_registry_has_all_legacy_train_keys():
    legacy = ("mri_model subtraction_mode feature_groups feature_model feature_selector imaging_patient_id_column "
              "allow_missing_imaging_features include_sensitive use_annotation_boxes epoch num_folds "
              "threshold_calibration_folds batch_size num_workers positive_boost sensitivity_lambda lr "
              "xgb_n_estimators xgb_max_depth xgb_learning_rate xgb_subsample xgb_colsample_bytree xgb_n_jobs "
              "lasso_cv_folds lasso_cs lasso_max_iter lasso_tolerance lasso_min_features lasso_n_jobs "
              "lasso_plot_top_n mlp_hidden_layers mlp_alpha mlp_learning_rate mlp_max_iter xgb_extra_json "
              "mlp_extra_json mri_extra_json trainer_extra_json seed target_size").split()
    for key in legacy:
        assert schema.FIELDS[key].section == "train", key
    paths = ("clinical_features_file metadata_file imaging_features_file annotation_boxes_file image_root "
             "prepared_root subtraction_root logs_root checkpoint_root validation_root summary_root slurm_logs").split()
    for key in paths:
        assert schema.FIELDS[key].section == "path"
    for key in ("fusion_alpha", "expected_folds", "top_n", "include_incomplete"):
        assert key in schema.FIELDS


def test_legacy_defaults():
    cfg = schema.resolve("validate")
    assert cfg["seed"] == 42 and cfg["target_size"] == (256, 256, 64) and cfg["epoch"] == 30
    assert cfg["feature_groups"] == ("clinical",) and cfg["use_annotation_boxes"] is False
    assert cfg["early_stopping"] is False and cfg["fusion_alpha"] == 0.5
    assert cfg["lasso_tolerance"] == pytest.approx(1e-4)


def test_cli_parsing_legacy_forms():
    parser = schema.build_parser("train")
    ns = parser.parse_args(["--feature_groups", "clinical", "kinetic", "--target_size", "16", "16", "8",
                            "--use_annotation_boxes", "--imaging_patient_id_column", "Patient ID"])
    cfg = schema.resolve("train", vars(ns))
    assert cfg["feature_groups"] == ("clinical", "kinetic")
    assert cfg["target_size"] == (16, 16, 8)
    assert cfg["use_annotation_boxes"] is True
    ns = parser.parse_args(["--feature_groups", "clinical,kinetic,clinical", "--use_annotation_boxes", "false"])
    cfg = schema.resolve("train", vars(ns))
    assert cfg["feature_groups"] == ("clinical", "kinetic") and cfg["use_annotation_boxes"] is False


@pytest.mark.parametrize("overrides, message", [
    ({"mri_model": "vgg"}, "not in"),
    ({"feature_groups": "clinical,genomics"}, "unknown items"),
    ({"image_dim": "2d"}, "duke_legacy_3d requires"),
    ({"profile": "general", "image_dim": "2d", "mask_mode": "roi_crop"}, "mask_mode requires"),
    ({"profile": "general", "mask_provider": "local_files", "mask_roi_classes": ""}, "mask_roi_classes"),
    ({"label_scheme": "oncotype_3class_18_31"}, "binary-only"),
    ({"trainer_extra_json": '{"callbacks": []}'}, "cannot override"),
    ({"fusion_alpha": "1.5"}, "fusion_alpha"),
    ({"branches": "tabular", "primary_branch": "image"}, "was not trained"),
    ({"selection_split": "outer"}, "historical image-only"),
    ({"profile": "general", "slice_selection": "mask_max_area", "image_dim": "2d"}, "mask-based slice selection"),
    ({"xgb_extra_json": "[1]"}, "JSON object"),
    ({"num_folds": "1"}, "num_folds"),
    ({"target_size": "16 16"}, "three integers"),
])
def test_rejected_combinations(overrides, message):
    with pytest.raises(ConfigError, match=message):
        schema.resolve("validate", overrides)


def test_three_class_with_argmax_is_allowed():
    cfg = schema.resolve("validate", {"label_scheme": "oncotype_3class_18_31", "threshold_strategy": "fixed_0.5"})
    assert schema.num_classes(cfg) == 3


def test_train_id_ignores_parallelism_outputs_and_label_tag():
    base = schema.resolve("train")
    tid = schema.train_id(base)
    for key, value in [("num_workers", "0"), ("lasso_n_jobs", "1"), ("xgb_n_jobs", "2"), ("experiment_id", "x"),
                       ("checkpoint_root", "/elsewhere"), ("prepared_root", "/cache2")]:
        assert schema.train_id(schema.resolve("train", {key: value})) == tid, key
    assert schema.train_id(schema.resolve("train", {"lr": "2e-4"})) != tid
    assert schema.train_id(base, {"clinical_features_file": "sha256:abc"}) != tid


def test_train_id_is_canonical_for_equivalent_spellings():
    a = schema.resolve("train", {"lr": "1e-4", "feature_groups": "clinical,kinetic"})
    b = schema.resolve("train", {"lr": "0.0001", "feature_groups": ["clinical", "kinetic"]})
    assert schema.train_id(a) == schema.train_id(b)


def test_eval_id_depends_only_on_validate_identity():
    a = schema.resolve("validate", {"fusion_alpha": "0.25"})
    b = schema.resolve("validate", {"fusion_alpha": "0.25", "validate_num_workers": "0"})
    c = schema.resolve("validate", {"fusion_alpha": "0.75"})
    assert schema.eval_id(a) == schema.eval_id(b) != schema.eval_id(c)
    assert schema.eval_id(a).startswith("ev-a0.25-")
    assert schema.train_id(a) == schema.train_id(c)


def test_format_parse_roundtrip():
    cfg = schema.resolve("validate")
    for key, value in cfg.items():
        field = schema.FIELDS[key]
        assert schema.parse_value(field, schema.format_value(field, value)) == value, key
