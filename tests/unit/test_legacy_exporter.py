"""mribreast.legacy.exporter on synthetic data with the legacy headers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mribreast.legacy import exporter, synthetic_reports as syn
from mribreast.legacy._fs import fs
from mribreast.legacy.names import build_experiment_name

#: Header of the tracked legacy ``validationSummary/all_validation_metrics.csv`` (schema only, no values).
STORED_ALL_METRICS_HEADER = (
    "run_id,fold,checkpoint_path,inner_fit_rows,inner_calibration_rows,effective_inner_calibration_folds,"
    "outer_validation_rows,image_accuracy,image_balanced_accuracy,image_sensitivity,image_decision_threshold,"
    "image_specificity,image_auc_roc,xgboost_accuracy,xgboost_balanced_accuracy,xgboost_sensitivity,"
    "xgboost_decision_threshold,xgboost_specificity,xgboost_auc_roc,fusion_accuracy,fusion_balanced_accuracy,"
    "fusion_sensitivity,fusion_decision_threshold,fusion_specificity,fusion_auc_roc,tabular_accuracy,"
    "tabular_balanced_accuracy,tabular_sensitivity,tabular_specificity,tabular_auc_roc,tabular_decision_threshold,"
    "allow_missing_imaging_features,batch_size,checkpoint_root,device,experiment_name,feature_groups,feature_model,"
    "feature_selector,fusion_alpha,imaging_features_file,imaging_patient_id_column,include_sensitive,lasso_cs,"
    "lasso_cv_folds,lasso_max_iter,lasso_min_features,lasso_n_jobs,lasso_tolerance,lr,mri_model,num_folds,"
    "num_workers,output_dir,positive_boost,sensitivity_lambda,subtraction_mode,threshold_calibration_folds"
).split(",")
STORED_CONFIG_SUMMARY_PREFIX = ("run_id,mri_model,subtraction_mode,feature_groups,feature_model,feature_selector,"
                                "batch_size,positive_boost,sensitivity_lambda,lr,fusion_alpha,expected_folds,"
                                "completed_folds,observed_fold_ids,is_complete,inner_fit_rows_mean").split(",")


def test_fold_metrics_column_order_and_tabular_name():
    rng = np.random.default_rng(1)
    new = syn.synthetic_fold_metrics(rng, 2).rename(columns={"inner_fit_rows": "fit_rows"})
    legacy = exporter.to_legacy_fold_metrics(new, "mlp")
    assert list(legacy.columns) == [
        "fold", "checkpoint_path", "inner_fit_rows", "inner_calibration_rows", "effective_inner_calibration_folds",
        "outer_validation_rows", *[f"{b}_{m}" for b in ("image", "mlp", "fusion") for m in exporter.LEGACY_BRANCH_METRICS]]
    assert np.allclose(legacy["mlp_auc_roc"], new["tabular_auc_roc"])


def test_image_only_export_omits_missing_branches():
    frame = pd.DataFrame({"fold": [1], "image_auc_roc": [0.5], "image_accuracy": [0.4]})
    assert list(exporter.to_legacy_fold_metrics(frame, "xgboost").columns) == ["fold", "image_accuracy", "image_auc_roc"]


def test_synthetic_stored_era_summary_has_stored_headers(tmp_path):
    configs = syn.grid(mri_models=("densenet121",), sens=(0.05,), boosts=(1.0, 2.0))
    syn.write_synthetic_validation_root(tmp_path / "val", configs, stored_era=True)
    out = exporter.summarize_legacy(tmp_path / "val", tmp_path / "summary", plots=False)
    header = pd.read_csv(out / "all_validation_metrics.csv", nrows=0).columns.tolist()
    assert header == STORED_ALL_METRICS_HEADER
    conf = pd.read_csv(out / "configuration_summary.csv", nrows=0).columns.tolist()
    assert conf[: len(STORED_CONFIG_SUMMARY_PREFIX)] == STORED_CONFIG_SUMMARY_PREFIX
    ranked = pd.read_csv(out / "ranked_configurations.csv")
    assert ranked.columns[0] == "rank" and ranked["rank"].tolist() == list(range(1, len(configs) + 1))
    assert (ranked["tabular_auc_roc_mean"] == ranked["xgboost_auc_roc_mean"]).all()
    best = json.loads((out / "best_configuration.json").read_text(encoding="utf-8"))
    assert best["run_id"] == ranked.loc[0, "run_id"] and best["rank"] == 1
    report = (out / "validation_report.md").read_text(encoding="utf-8")
    assert "- Discovered configurations: 4" in report and best["run_id"] in report


def test_ranking_order_ties_nan_and_incomplete(tmp_path):
    configs = syn.grid(mri_models=("densenet121", "resnet18"), sens=(0.05,), boosts=(1.0,), groups=("clinical",))
    configs += [{**configs[0], "batch_size": 8}]
    # configs 0 and 1 tie on fusion AUC -> next keys decide; config 2 has only 3 folds (incomplete)
    syn.write_synthetic_validation_root(tmp_path / "v", configs, tie_groups={0: 0.7, 1: 0.7},
                                        incomplete={2: (1, 2, 3)})
    _, configurations, _ = exporter.collect_results(tmp_path / "v")
    ranked = exporter.rank_configurations(configurations)
    assert len(ranked) == 2 and not ranked["run_id"].str.contains("bs8").any()
    top = configurations.set_index("run_id").loc[ranked["run_id"]]
    assert (top["fusion_auc_roc_mean"] == 0.7).all()
    assert top["fusion_balanced_accuracy_mean"].is_monotonic_decreasing
    with_incomplete = exporter.rank_configurations(configurations, include_incomplete=True)
    assert len(with_incomplete) == 3


def test_write_legacy_run_files(tmp_path):
    cfg = {**syn.BASE_CFG, "mri_model": "resnet10", "feature_groups": "clinical", "feature_model": "xgboost",
           "sensitivity_lambda": 0.2, "positive_boost": 1.0, "experiment_id": "cfgabc"}
    metrics = syn.synthetic_fold_metrics(np.random.default_rng(3), 2)
    thr_metrics = pd.DataFrame([{"fold": f, "branch": b, "threshold": 0.4, "balanced_accuracy": 0.6,
                                 "sensitivity": 0.5, "specificity": 0.7, "n_rows": 42}
                                for f in (1, 2) for b in ("fusion", "tabular", "image")])
    preds = pd.DataFrame([{"fold": 1, "patient_id": f"P{i}", "study_id": f"S{i}", "label": i % 2, "branch": b,
                           "split": "outer_eval", "prob_0": 0.3, "prob_1": 0.7, "decision": 1}
                          for i in range(3) for b in ("image", "tabular", "fusion")])
    run = exporter.write_legacy_run(tmp_path, cfg, 0.25, metrics,
                                    thresholds={1: {"image": 0.1, "tabular": 0.2, "fusion": 0.3}},
                                    threshold_metrics=thr_metrics, predictions={"validation": preds})
    assert run == tmp_path / build_experiment_name(cfg) / "fusion_alpha_0.25"
    run = fs(run)  # legacy names exceed Windows MAX_PATH
    config = json.loads((run / "validation_config.json").read_text(encoding="utf-8"))
    assert set(config) == set(exporter.LEGACY_VALIDATE_ARGS) | {"experiment_name", "device"}
    assert config["feature_groups"] == ["clinical"] and config["target_size"] == [256, 256, 64]
    assert config["experiment_name"].endswith("_cfgabc") and config["num_workers"] == 4
    thresholds = json.loads((run / "fold_1" / "decision_thresholds.json").read_text(encoding="utf-8"))
    assert thresholds == {"fusion": 0.3, "image": 0.1, "xgboost": 0.2}
    thr = pd.read_csv(run / "threshold_calibration_metrics.csv")
    assert list(thr.columns) == list(exporter.THRESHOLD_COLUMNS)
    assert thr.loc[thr.fold == 1, "branch"].tolist() == ["image", "xgboost", "fusion"]
    wide = pd.read_csv(run / "fold_1" / "fusion_validation_predictions.csv")
    assert list(wide.columns) == ["patientId", "studyId", "label", "image_probability_0", "xgboost_probability_0",
                                  "fusion_probability_0", "image_probability_1", "xgboost_probability_1",
                                  "fusion_probability_1", "image_prediction", "xgboost_prediction",
                                  "fusion_prediction", "image_decision_threshold", "xgboost_decision_threshold",
                                  "fusion_decision_threshold"]
    summary = pd.read_csv(run / "validation_summary.csv")
    assert list(summary.columns) == ["metric", "mean", "std"] and "fold" not in set(summary["metric"])
    legacy_metrics = pd.read_csv(run / "validation_metrics.csv")
    expected_std = float(np.std(legacy_metrics["fusion_auc_roc"].to_numpy()))
    assert np.isclose(summary.set_index("metric").loc["fusion_auc_roc", "std"], expected_std, rtol=0, atol=1e-15)


def test_export_new_evaluation_dir(tmp_path):
    eval_dir = tmp_path / "validation" / "tr-0123456789ab" / "ev-a0.5-deadbeef"
    eval_dir.mkdir(parents=True)
    cfg = {**syn.BASE_CFG, "mri_model": "densenet121", "feature_groups": ["clinical"], "feature_model": "mlp",
           "sensitivity_lambda": 0.05, "positive_boost": 1.0, "fusion_alpha": 0.5, "label_scheme": "oncotype_binary_18"}
    (eval_dir / "evaluation_manifest.json").write_text(json.dumps({"resolved_config": cfg}), encoding="utf-8")
    syn.synthetic_fold_metrics(np.random.default_rng(0), 5).to_csv(eval_dir / "validation_metrics.csv", index=False)
    (eval_dir / "decision_thresholds.json").write_text(json.dumps({"1": {"image": 0.5, "tabular": 0.4, "fusion": 0.45}}),
                                                       encoding="utf-8")
    run = exporter.export_evaluation_dir(eval_dir, tmp_path / "legacy")
    assert run.parent.name.endswith("_tr-0123456789ab") and run.name == "fusion_alpha_0.5"
    run = fs(run)
    assert json.loads((run / "fold_1" / "decision_thresholds.json").read_text(encoding="utf-8"))["mlp"] == 0.4
    assert "mlp_auc_roc" in pd.read_csv(run / "validation_metrics.csv").columns
