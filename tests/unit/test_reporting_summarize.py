"""summarize(): discovery, completeness, legacy ranking order, include_incomplete, outputs."""

import json

import numpy as np
import pandas as pd
import pytest

from mribreast.artifacts import mark_complete, read_json
from mribreast.protocol.train import run_training
from mribreast.protocol.validate import run_validation
from mribreast.reporting.summarize import collect_results, rank_configurations, summarize
from tests.fakes import import_legacy, make_cfg, make_fake_cohort, make_trainers

LEGACY_CONFIG = ("mri_model", "subtraction_mode", "feature_groups", "feature_model", "feature_selector", "batch_size",
                 "positive_boost", "sensitivity_lambda", "lr", "fusion_alpha")


def _write_run(root, train_id, eval_id, fusion, *, folds=3, complete=True, primary="fusion"):
    """Synthetic evaluation dir; ``fusion`` = (auc, ba, sens, spec) per fold or a constant tuple."""
    run = root / train_id / eval_id
    run.mkdir(parents=True)
    rows = []
    for fold in range(1, folds + 1):
        auc, ba, sens, spec = fusion
        rows.append({"fold": fold, "checkpoint_path": f"/x/{fold}", "inner_fit_rows": 10 + fold, "inner_calibration_rows": 4,
                     "effective_inner_calibration_folds": 3, "outer_validation_rows": 5,
                     "image_auc_roc": 0.5, "tabular_auc_roc": 0.6, "fusion_accuracy": 0.7,
                     "fusion_balanced_accuracy": ba, "fusion_sensitivity": sens, "fusion_decision_threshold": 0.5,
                     "fusion_specificity": spec, "fusion_auc_roc": auc})
    pd.DataFrame(rows).to_csv(run / "validation_metrics.csv", index=False)
    (run / "validation_config.json").write_text(json.dumps({
        "num_folds": 3, "mri_model": "resnet18", "feature_groups": ["clinical", "kinetic"], "fusion_alpha": 0.5,
        "primary_branch_resolved": primary, "branches": ["image", "tabular"]}), encoding="utf-8")
    if complete:
        mark_complete(run)
    return run


@pytest.fixture()
def synthetic_root(tmp_path):
    root = tmp_path / "validation"
    _write_run(root, "tr-a", "ev-a0.5-1", (0.70, 0.60, 0.50, 0.70))
    _write_run(root, "tr-b", "ev-a0.5-1", (0.70, 0.60, 0.55, 0.60))  # ties on auc+ba, wins on sensitivity
    _write_run(root, "tr-c", "ev-a0.5-1", (0.80, 0.50, 0.10, 0.90))  # best auc
    _write_run(root, "tr-d", "ev-a0.5-1", (0.70, 0.60, 0.55, 0.60))  # full tie with tr-b -> discovery order
    _write_run(root, "tr-e", "ev-a0.5-1", (0.95, 0.90, 0.90, 0.90), folds=2)  # missing fold -> incomplete
    _write_run(root, "tr-f", "ev-a0.5-1", (0.99, 0.90, 0.90, 0.90), complete=False)  # no COMPLETE flag
    _write_run(root, "tr-g", "ev-a0.5-1", (np.nan, 0.99, 0.99, 0.99))  # NaN auc -> last
    (root / "tr-a" / ".ev-x.staging-123").mkdir()  # staging dirs are ignored
    return root


def test_ranking_order_and_completeness(synthetic_root, tmp_path):
    result = summarize(synthetic_root, tmp_path / "summary", expected_folds=3, top_n=5, include_incomplete=False)
    assert result["ranked_run_ids"] == ["tr-c/ev-a0.5-1", "tr-b/ev-a0.5-1", "tr-d/ev-a0.5-1", "tr-a/ev-a0.5-1",
                                        "tr-g/ev-a0.5-1"]
    assert result["discovered"] == 7 and result["incomplete"] == 2
    assert any("tr-f" in w and "COMPLETE" in w for w in result["warnings"])
    assert any("tr-e" in w for w in result["warnings"])
    best = read_json(tmp_path / "summary" / "best_configuration.json")
    assert best["run_id"] == "tr-c/ev-a0.5-1" and best["rank"] == 1 and best["fusion_auc_roc_mean"] == pytest.approx(0.8)
    ranked = pd.read_csv(tmp_path / "summary" / "ranked_configurations.csv")
    legacy_prefix = ["rank", "run_id", *LEGACY_CONFIG, "train_id"]
    assert list(ranked.columns[: len(legacy_prefix)]) == legacy_prefix
    configs = pd.read_csv(tmp_path / "summary" / "configuration_summary.csv")
    assert configs.run_id.tolist() == sorted(configs.run_id)
    assert {"expected_folds", "completed_folds", "observed_fold_ids", "is_complete", "fusion_auc_roc_mean",
            "fusion_auc_roc_std", "primary_auc_roc_mean"} <= set(configs.columns)
    allm = pd.read_csv(tmp_path / "summary" / "all_validation_metrics.csv")
    assert allm.columns[0] == "run_id" and "mri_model" in allm.columns
    assert set(allm.feature_groups) == {"clinical,kinetic"}
    report = (tmp_path / "summary" / "validation_report.md").read_text(encoding="utf-8")
    assert "Incomplete configurations: 2" in report and "independent test set" in report


def test_include_incomplete(synthetic_root, tmp_path):
    result = summarize(synthetic_root, tmp_path / "s", expected_folds=3, top_n=3, include_incomplete=True)
    assert result["ranked_run_ids"][:2] == ["tr-f/ev-a0.5-1", "tr-e/ev-a0.5-1"]
    assert result["ranked"] == 7


def test_ranking_matches_legacy_implementation(synthetic_root):
    legacy = import_legacy("summarize_configurable_imaging_features_fusion")
    _, configs, _ = collect_results(synthetic_root, 3)
    for include in (False, True):
        ours = rank_configurations(configs, include)
        theirs = legacy.rank_configurations(configs, include)  # legacy keys: fusion_* means
        assert ours.run_id.tolist() == theirs.run_id.tolist()
        assert ours["rank"].tolist() == theirs["rank"].tolist()


def test_std_is_population_std(synthetic_root):
    _, configs, _ = collect_results(synthetic_root, 3)
    row = configs.set_index("run_id").loc["tr-a/ev-a0.5-1"]
    assert row["inner_fit_rows_std"] == pytest.approx(np.std([11, 12, 13]))  # ddof=0 (legacy), not 1.0
    assert row["inner_fit_rows_mean"] == 12


def test_summarize_real_evaluations_across_alphas(tmp_path):
    cohort = make_fake_cohort(40, multi_study=True)
    trainers = make_trainers()
    fps = {"clinical_features_file": "sha256:c"}
    run_training(make_cfg("train"), cohort, trainers, checkpoint_root=tmp_path / "ck", input_fingerprints=fps)
    for alpha in (0.0, 0.5, 1.0):
        run_validation(make_cfg("validate", fusion_alpha=alpha), cohort, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "val", input_fingerprints=fps)
    result = summarize(tmp_path / "val", tmp_path / "sum", expected_folds=5, top_n=20, include_incomplete=False)
    assert result["discovered"] == 3 and result["complete"] == 3 and len(result["ranked_run_ids"]) == 3
    ranked = pd.read_csv(tmp_path / "sum" / "ranked_configurations.csv")
    assert ranked.fusion_auc_roc_mean.is_monotonic_decreasing
    assert set(ranked.fusion_alpha) == {0.0, 0.5, 1.0}
    assert (ranked.primary_branch == "fusion").all() and not ranked.selection_uses_outer_eval.any()
    assert (tmp_path / "sum" / "fusion_metric_comparison.png").is_file()
    assert (tmp_path / "sum" / "branch_auc_comparison.png").is_file()


def test_no_runs_raises(tmp_path):
    (tmp_path / "v").mkdir()
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path / "v", tmp_path / "s", expected_folds=5, top_n=5, include_incomplete=False)
