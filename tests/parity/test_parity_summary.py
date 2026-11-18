"""Parity: legacy summarizer vs mribreast.legacy.exporter.summarize_legacy on the same synthetic tree.

Tolerance: CSV floats compared with rtol=1e-12, atol=0 (both sides compute the
same pandas/numpy reductions; observed difference is 0).  ``run_id`` path
separators are normalized ('\\' -> '/') because the legacy code uses
``str(Path.relative_to)`` (OS separator) while the port always writes '/'
(the separator of the stored Linux results).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from legacy_loader import load_file_module
from mribreast.legacy import exporter, synthetic_reports as syn

pytestmark = pytest.mark.parity

RTOL = 1e-12


def _tree(root):
    # short names (Windows MAX_PATH for the legacy code): no lasso, single-group choices
    configs = syn.grid(mri_models=("fcn", "resnet10"), sens=(0.05, 0.2), boosts=(1.0,), groups=("kinetic", "morphology"),
                       feature_models=("xgboost",))
    configs = [{**c, "feature_selector": "none"} for c in configs]
    configs[3] = {**configs[3], "feature_model": "mlp"}
    syn.write_synthetic_validation_root(root, configs, seed=7, alphas=(0.5, 1.0),
                                        incomplete={1: (1, 2, 4)}, tie_groups={2: 0.65, 4: 0.65, 5: 0.65})
    # NaN fold AUC, an empty metrics file, a metrics file without fold, and a nested summary dir (skipped)
    nan_run = syn.write_synthetic_validation_root(root, [{**configs[0], "batch_size": 8}], seed=8)[0]
    frame = pd.read_csv(nan_run / "validation_metrics.csv")
    frame.loc[0, "fusion_auc_roc"] = np.nan
    frame.to_csv(nan_run / "validation_metrics.csv", index=False)
    (root / "empty_run").mkdir()
    (root / "empty_run" / "validation_metrics.csv").write_text("fold,fusion_auc_roc\n", encoding="utf-8")
    (root / "nofold_run").mkdir()
    pd.DataFrame({"x": [1]}).to_csv(root / "nofold_run" / "validation_metrics.csv", index=False)
    (root / "summary" / "old").mkdir(parents=True)
    pd.DataFrame({"fold": [1]}).to_csv(root / "summary" / "old" / "validation_metrics.csv", index=False)
    return configs


def _read(path):
    frame = pd.read_csv(path)
    if "run_id" in frame.columns:
        frame["run_id"] = frame["run_id"].str.replace("\\", "/", regex=False)
    return frame


@pytest.mark.parametrize("include_incomplete", [False, True])
def test_summary_outputs_identical(legacy_repo, tmp_path_factory, include_incomplete):
    legacy = load_file_module(legacy_repo, "mriBreastDuke/summarize_configurable_imaging_features_fusion.py")
    root = tmp_path_factory.mktemp("v")
    _tree(root)
    out_legacy, out_new = root.parent / f"{root.name}_L", root.parent / f"{root.name}_N"
    legacy.generate_summary(argparse.Namespace(input_dir=str(root), output_dir=str(out_legacy), expected_folds=5,
                                               top_n=20, include_incomplete=include_incomplete))
    exporter.summarize_legacy(root, out_new, expected_folds=5, top_n=20, include_incomplete=include_incomplete)
    for name in ("all_validation_metrics.csv", "configuration_summary.csv", "ranked_configurations.csv"):
        left, right = _read(out_legacy / name), _read(out_new / name)
        if name == "configuration_summary.csv":   # legacy sorts by run_id with OS separators; compare sorted
            left = left.sort_values("run_id", kind="stable").reset_index(drop=True)
            right = right.sort_values("run_id", kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=RTOL, atol=0)
    ranked = _read(out_new / "ranked_configurations.csv")
    assert ranked["run_id"].tolist() == _read(out_legacy / "ranked_configurations.csv")["run_id"].tolist()
    best_l = json.loads((out_legacy / "best_configuration.json").read_text(encoding="utf-8"))
    best_n = json.loads((out_new / "best_configuration.json").read_text(encoding="utf-8"))
    best_l["run_id"] = best_l["run_id"].replace("\\", "/")
    assert best_l.keys() == best_n.keys()
    for key, value in best_l.items():
        if isinstance(value, float):
            assert best_n[key] == pytest.approx(value, rel=RTOL, abs=0), key
        else:
            assert best_n[key] == value, key
    report_l = (out_legacy / "validation_report.md").read_text(encoding="utf-8").replace("\\", "/")
    report_n = (out_new / "validation_report.md").read_text(encoding="utf-8").replace("\\", "/")
    assert report_l == report_n
    for png in ("fusion_metric_comparison.png", "branch_auc_comparison.png"):
        assert (out_legacy / png).is_file() == (out_new / png).is_file()


def test_stored_summary_headers_reproduced(legacy_repo, tmp_path):
    """Synthetic stored-era reports have exactly the headers of the tracked validationSummary (no values compared)."""
    stored = legacy_repo / "mriBreastDuke" / "validationSummary"
    configs = syn.grid(mri_models=("densenet121",), sens=(0.05,), boosts=(1.0,), groups=("clinical",))
    syn.write_synthetic_validation_root(tmp_path / "v", configs, stored_era=True)
    out = exporter.summarize_legacy(tmp_path / "v", tmp_path / "s", plots=False)
    for name in ("all_validation_metrics.csv", "configuration_summary.csv", "ranked_configurations.csv"):
        assert pd.read_csv(out / name, nrows=0).columns.tolist() == pd.read_csv(stored / name, nrows=0).columns.tolist()
    assert sorted(json.loads((out / "best_configuration.json").read_text(encoding="utf-8"))) == \
        sorted(json.loads((stored / "best_configuration.json").read_text(encoding="utf-8")))


def test_stored_ranking_order_is_reproduced(legacy_repo):
    """Re-ranking the stored configuration_summary.csv with the port reproduces ranked_configurations.csv order."""
    stored = legacy_repo / "mriBreastDuke" / "validationSummary"
    configurations = pd.read_csv(stored / "configuration_summary.csv")
    expected = pd.read_csv(stored / "ranked_configurations.csv")
    legacy = load_file_module(legacy_repo, "mriBreastDuke/summarize_configurable_imaging_features_fusion.py")
    ours = exporter.rank_configurations(configurations, include_incomplete=False)
    theirs = legacy.rank_configurations(configurations, include_incomplete=False)
    assert ours["run_id"].tolist() == theirs["run_id"].tolist() == expected["run_id"].tolist()
