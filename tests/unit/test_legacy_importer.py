"""mribreast.legacy.importer on synthetic legacy reports (stored-era and current-era layouts)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mribreast.legacy import exporter, importer, synthetic_reports as syn
from mribreast.legacy._fs import fs
from mribreast.legacy.checkpoints import legacy_fold_checkpoint_dir
from mribreast.legacy.names import build_experiment_name, build_historical_image_name


@pytest.fixture()
def stored_era(tmp_path):
    configs = syn.grid(mri_models=("densenet121",), sens=(0.05,), boosts=(1.0, 2.0))
    syn.write_synthetic_validation_root(tmp_path / "val", configs, stored_era=True)
    summary = exporter.summarize_legacy(tmp_path / "val", tmp_path / "summary", plots=False)
    return tmp_path, configs, summary


def test_summary_import_normalizes_and_flags(stored_era):
    tmp_path, configs, summary = stored_era
    out = importer.import_summary_dir(summary)
    folds, conf, best = out["folds"], out["configurations"], out["best"]
    assert len(folds) == 5 * len(configs) and len(conf) == len(configs) and len(best) == 1
    assert not any(c.startswith(("xgboost_", "mlp_")) for c in folds.columns)
    raw = pd.read_csv(summary / "all_validation_metrics.csv")
    assert np.allclose(folds["tabular_auc_roc"].to_numpy(), raw["xgboost_auc_roc"].to_numpy())
    assert {"fit_rows", "calibration_rows", "outer_eval_rows", "effective_inner_folds"} <= set(folds.columns)
    # stored era: names without _clinical-safe-v2 -> retraining required; split verified by legacy validation
    assert folds["requires_retraining"].all()
    assert set(folds["retraining_reasons"]) == {"pre_clinical_safe_v2_predictors"}
    assert set(folds["split_metadata"]) == {"verified_by_legacy_validation"}
    assert folds["notes"].str.contains("config_predates_keys:seed").all()
    assert folds["run_id"].str.match(r"^lg-[0-9a-f]{12}/ev-a0\.5-legacy$").all()
    assert set(folds["selection_split"]) == {"shared"} and set(folds["tabular_model"]) == {"xgboost"}
    assert conf.sort_values("rank")["legacy_run_id"].tolist() == pd.read_csv(summary / "ranked_configurations.csv")["run_id"].tolist()
    assert best.loc[0, "legacy_run_id"] == conf.sort_values("rank").iloc[0]["legacy_run_id"]


def test_current_era_validation_root_with_checkpoint_root(tmp_path):
    configs = syn.grid(mri_models=("resnet18",), sens=(0.2,), boosts=(1.0,), groups=("clinical",),
                       feature_models=("mlp",))
    configs = [{**c, "experiment_id": "cfg000000000001", "feature_selector": "none"} for c in configs]
    syn.write_synthetic_validation_root(tmp_path / "val", configs, alphas=(0.25, 0.75))
    name = build_experiment_name(configs[0])
    for fold in range(1, 6):
        d = fs(legacy_fold_checkpoint_dir(tmp_path / "ck", name, fold))
        d.mkdir(parents=True)
        (d / "best-epoch=01-val_auc_roc=0.6000-val_balanced_accuracy=0.5000-val_sensitivity=0.5000.ckpt").write_bytes(b"")
        (d / "tabular_preprocessor.joblib").write_bytes(b"")
        (d / "mlp_model.joblib").write_bytes(b"")
        if fold != 3:
            (d / "inner_calibration_split.csv").write_text(
                "studyId,patientId,label,split,requested_inner_folds,effective_inner_folds\n", encoding="utf-8")
    result = importer.import_validation_root(tmp_path / "val", checkpoint_root=tmp_path / "ck")
    folds, thresholds = result["folds"], result["thresholds"]
    assert sorted(folds["fusion_alpha"].unique()) == [0.25, 0.75]
    assert folds["train_id"].nunique() == 1 and folds["eval_id"].nunique() == 2
    assert set(folds["retraining_reasons"]) == {"split_metadata_missing"}  # fold 3 manifest missing
    assert set(folds["split_metadata"]) == {"missing"}
    assert set(thresholds["branch"]) == {"image", "tabular", "fusion"}
    assert thresholds["run_id"].notna().all()


def test_historical_image_and_old_checkpoint_formats_flagged():
    name = build_historical_image_name(1, 1e-4, 0.3, 1.0, 8)
    verdict = importer.assess(name, {})
    assert verdict["requires_retraining"] and "image_only_outer_selection" in verdict["retraining_reasons"]
    assert verdict["split_metadata"] == "none_written" and verdict["protocol"] == "historical_image_outer_selection"
    fusion = build_experiment_name({**syn.BASE_CFG, "mri_model": "fcn", "feature_groups": "kinetic",
                                    "feature_model": "xgboost", "sensitivity_lambda": 0.05, "positive_boost": 1.0})
    old_ckpt = "/x/best-epoch=03-val_balanced_accuracy=0.6100-val_sensitivity=0.7000.ckpt"
    verdict = importer.assess(fusion, {"seed": 42}, checkpoint_paths=[old_ckpt], has_calibration_rows=True)
    assert verdict["retraining_reasons"] == "checkpoint_format_balanced_accuracy"
    pre = importer.assess("ImagingFeaturesFusion_densenet121_none_clinical_xgboost_lr1e-04", {})
    assert pre["retraining_reasons"] == "no_inner_calibration"
    good = importer.assess(fusion, {"seed": 42}, has_calibration_rows=True)
    assert good["comparable_to_new"] and not good["requires_retraining"]


def test_cli_writes_only_output(stored_era, tmp_path):
    root, _, summary = stored_era
    before = sorted(os.listdir(summary))
    out = tmp_path / "imported"
    assert importer.main(["--summary_dir", str(summary), "--validation_root", str(root / "val"),
                          "--output_dir", str(out)]) == 0
    assert sorted(os.listdir(summary)) == before
    mapping = pd.read_csv(out / "run_id_map.csv")
    assert len(mapping) == 4 and mapping["requires_retraining"].all()
    assert "pre_clinical_safe_v2_predictors" in (out / "import_report.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        importer.main(["--summary_dir", str(summary), "--output_dir", str(summary / "x")])


def test_tabular_duplicate_mismatch_is_noted():
    frame = pd.DataFrame({"xgboost_auc_roc": [0.5], "tabular_auc_roc": [0.6]})
    normalized, notes = importer.normalize_metric_columns(frame, "xgboost")
    assert notes == ["tabular_duplicate_mismatch:auc_roc"] and normalized["tabular_auc_roc"].tolist() == [0.5]
