"""Tabular preprocessing, LASSO, models, sample weights and TabularBranch round trip."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from mribreast.contracts import LABEL, PATIENT_ID, STUDY_ID, BranchTrainer, FittedPredictor, FoldData
from mribreast.data.predictor_policy import PredictorPolicyError
from mribreast.tabular import (
    GroupedEarlyStoppingMLP, LassoFeatureSelector, TabularPreprocessor, balanced_sample_weights,
    build_tabular_branch, lasso_stability, save_lasso_stability,
)
from mribreast.tabular.models import make_mlp, make_xgboost, require_mlp_sample_weight

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

BASE_CFG = {
    "profile": "duke_legacy_3d", "seed": 42, "positive_class": 1, "positive_boost": 1.0, "feature_model": "xgboost",
    "feature_selector": "none", "selector_preprocessing": "outer", "mlp_early_stopping": "internal",
    "lasso_cv_folds": 3, "lasso_cs": 4, "lasso_max_iter": 2000, "lasso_tolerance": 1e-3, "lasso_min_features": 1,
    "lasso_n_jobs": 1, "lasso_plot_top_n": 10, "xgb_n_estimators": 30, "xgb_max_depth": 2, "xgb_learning_rate": 0.1,
    "xgb_subsample": 0.8, "xgb_colsample_bytree": 0.8, "xgb_n_jobs": 1, "mlp_hidden_layers": "8", "mlp_alpha": 1e-4,
    "mlp_learning_rate": 1e-2, "mlp_max_iter": 60, "xgb_extra_json": {}, "mlp_extra_json": {},
    "prediction_time": "mri", "include_sensitive": False, "allowed_predictors_file": "",
}
CONT = ["age_at_diagnosis_years", "kinetic_peak"]
CAT = ["menopause", "tumor_laterality"]


def make_frame(n_patients: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        label = p % 2
        for s in range(1 + (p % 5 == 0)):
            rows.append({PATIENT_ID: f"P{p:03d}", STUDY_ID: f"S{p:03d}_{s}", LABEL: label,
                         "age_at_diagnosis_years": 50 + 5 * label + rng.normal(scale=5),
                         "kinetic_peak": label * 1.5 + rng.normal(), "menopause": float(rng.integers(0, 2)),
                         "tumor_laterality": rng.choice(["L", "R"])})
    frame = pd.DataFrame(rows)
    frame.loc[3, "age_at_diagnosis_years"] = np.nan
    frame.loc[4, "tumor_laterality"] = np.nan
    return frame


def test_preprocessor_names_impute_and_unknown_category():
    frame = make_frame()
    pre = TabularPreprocessor(CONT, CAT).fit(frame)
    names = pre.get_feature_names_out()
    assert names[:2] == ["continuous__age_at_diagnosis_years", "continuous__kinetic_peak"]
    assert "categorical__tumor_laterality_L" in names and "categorical__menopause_0.0" in names
    out = pre.transform(frame)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    np.testing.assert_allclose(out[:, :2].mean(axis=0), 0, atol=1e-5)
    unseen = frame.iloc[:2].copy()
    unseen["tumor_laterality"] = ["B", pd.NA]
    encoded = pre.transform(unseen)
    lat = [i for i, n in enumerate(names) if "tumor_laterality" in n]
    assert encoded[0, lat].sum() == 0                                           # unknown level -> all zeros
    assert encoded[1, lat].sum() == 1                                           # missing -> most frequent
    with pytest.raises(ValueError, match="Missing tabular"):
        pre.transform(frame.drop(columns=["kinetic_peak"]))


def test_sample_weights_balanced_times_positive_boost():
    labels = np.array([0, 0, 0, 1])
    np.testing.assert_allclose(balanced_sample_weights(labels), [4 / 6, 4 / 6, 4 / 6, 2.0])
    np.testing.assert_allclose(balanced_sample_weights(labels, positive_boost=1.5), [4 / 6, 4 / 6, 4 / 6, 3.0])
    three = np.array([0, 1, 2, 2])
    np.testing.assert_allclose(balanced_sample_weights(three, positive_boost=3.0), [4 / 3, 4 / 3, 2 / 3, 2 / 3])


def _lasso_inputs(seed=11):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(40), 2)
    labels = np.repeat(np.tile([0, 1], 20), 2)
    features = np.column_stack([(2 * labels - 1) + rng.normal(scale=0.15, size=80), rng.normal(size=(80, 5))])
    return features, labels, groups


def test_lasso_grouped_cv_deterministic_and_patients_intact():
    features, labels, groups = _lasso_inputs()
    a = LassoFeatureSelector(cv_folds=4, cs=8, max_iter=3000, random_state=7).fit(features, labels, groups=groups)
    b = LassoFeatureSelector(cv_folds=4, cs=8, max_iter=3000, random_state=7).fit(features, labels, groups=groups)
    assert a.get_support()[0] and a.effective_cv_folds_ == 4
    np.testing.assert_array_equal(a.coefficients_, b.coefficients_)
    for train, test in a.make_cv_splits(labels, groups):
        assert not set(groups[train]) & set(groups[test])
    capped = LassoFeatureSelector(cv_folds=5, cs=4).fit(features[:12], np.repeat([0, 1, 0, 1, 0, 1], 2),
                                                        groups=np.repeat(np.arange(6), 2))
    assert capped.effective_cv_folds_ == 3


def test_lasso_minimum_feature_fallback_and_report():
    rng = np.random.default_rng(0)
    labels = np.tile([0, 1], 20)
    noise = rng.normal(size=(40, 4))
    sel = LassoFeatureSelector(cv_folds=3, cs=1, min_features=2, max_iter=2000).fit(noise, labels)  # C=1e-4 -> all zero
    assert sel.minimum_feature_fallback_used_ and sel.output_dimension == 2
    report = sel.selection_report(["continuous__age", "continuous__kinetic_a", "continuous__morphology_b",
                                   "continuous__density_ordinal_mean"])
    assert list(report.columns) == ["feature_name", "feature_group", "selected", "lasso_importance", "lasso_rank",
                                    "minimum_feature_fallback_used", "coefficient_class_1"]
    assert set(report["feature_group"]) == {"clinical", "kinetic", "morphology", "density"}
    assert report["selected"].tolist()[:2] == [True, True]
    with pytest.raises(ValueError, match="min_features"):
        LassoFeatureSelector(min_features=9).fit(noise, labels)


def test_fold_local_preprocessing_differs_from_outer():
    frame = make_frame(40)
    frame.loc[frame.index[:10], "age_at_diagnosis_years"] += 30                 # make subfold statistics differ
    labels, groups = frame[LABEL].to_numpy(), frame[PATIENT_ID].to_numpy()
    outer_pre = TabularPreprocessor(CONT, CAT).fit(frame)
    features = outer_pre.transform(frame)
    sel = LassoFeatureSelector(cv_folds=3, cs=4, max_iter=2000, random_state=1)
    sel.fit(features, labels, groups=groups, raw=frame[CONT + CAT], preprocessor_factory=lambda: TabularPreprocessor(CONT, CAT))
    assert sel.selector_preprocessing_ == "fold_local" and len(sel.cv_details_) == sel.effective_cv_folds_
    outer_mean = outer_pre.transformer_.named_transformers_["continuous"].named_steps["scaler"].mean_
    for detail in sel.cv_details_:
        pre = detail["preprocessor"]
        assert pre.n_fit_rows_ == detail["n_train"] < len(frame)
        sub = frame.iloc[detail["train_index"]]
        scaler = pre.transformer_.named_transformers_["continuous"].named_steps["scaler"]
        imputer = pre.transformer_.named_transformers_["continuous"].named_steps["imputer"]
        expected = sub[CONT].fillna(pd.Series(imputer.statistics_, index=CONT)).mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-6)        # fitted on the subfold-train rows only
        assert not np.allclose(scaler.mean_, outer_mean)
    outer_sel = LassoFeatureSelector(cv_folds=3, cs=4, max_iter=2000, random_state=1).fit(features, labels, groups=groups)
    assert outer_sel.selector_preprocessing_ == "outer" and outer_sel.cv_details_ == []
    assert sel.Cs_.shape == (4,) and sel.cv_scores_.shape == (sel.effective_cv_folds_, 4)


def test_mlp_sample_weight_supported_and_factories():
    require_mlp_sample_weight()
    cfg = dict(BASE_CFG)
    mlp = make_mlp(cfg, seed=3)
    assert mlp.early_stopping and mlp.validation_fraction == 0.15 and mlp.n_iter_no_change == 20
    assert not make_mlp({**cfg, "mlp_early_stopping": "off"}, 3).early_stopping
    assert isinstance(make_mlp({**cfg, "mlp_early_stopping": "grouped"}, 3), GroupedEarlyStoppingMLP)
    with pytest.raises(ValueError, match="conflicts"):
        make_mlp({**cfg, "mlp_early_stopping": "grouped", "mlp_extra_json": {"early_stopping": True}}, 3)
    xgb = make_xgboost({**cfg, "xgb_extra_json": {"min_child_weight": 2}}, 3, seed=5)
    params = xgb.get_params()
    assert params["objective"] == "multi:softprob" and params["num_class"] == 3 and params["min_child_weight"] == 2
    assert params["tree_method"] == "hist" and params["random_state"] == 5


def test_mlp_grouped_early_stopping_respects_groups():
    frame = make_frame(60)
    pre = TabularPreprocessor(CONT, CAT).fit(frame)
    x, y, groups = pre.transform(frame), frame[LABEL].to_numpy(), frame[PATIENT_ID].to_numpy()
    model = make_mlp({**BASE_CFG, "mlp_early_stopping": "grouped", "mlp_max_iter": 80}, seed=0)
    model.fit(x, y, sample_weight=balanced_sample_weights(y), groups=groups)
    assert not set(model.holdout_groups_) & set(model.train_groups_)
    holdout_rows = np.isin(groups, model.holdout_groups_)
    assert set(groups[holdout_rows]) == set(model.holdout_groups_)             # every study of a held-out patient
    assert 0.05 < holdout_rows.mean() < 0.3
    assert 1 <= model.best_epoch_ <= model.n_epochs_run_ <= 80
    assert model.predict_proba(x).shape == (len(x), 2)
    with pytest.raises(ValueError, match="groups"):
        make_mlp({**BASE_CFG, "mlp_early_stopping": "grouped"}, 0).fit(x, y)


@pytest.mark.parametrize("model,selector,prep,stopping", [
    ("xgboost", "lasso", "outer", "internal"), ("xgboost", "none", "outer", "internal"),
    ("mlp", "lasso", "fold_local", "grouped"), ("mlp", "none", "outer", "off")])
def test_tabular_branch_round_trip(tmp_path, model, selector, prep, stopping):
    frame = make_frame(40)
    fit, evaluation = frame[frame[PATIENT_ID] < "P030"], frame[frame[PATIENT_ID] >= "P030"]
    cfg = {**BASE_CFG, "feature_model": model, "feature_selector": selector, "selector_preprocessing": prep,
           "mlp_early_stopping": stopping}
    branch = build_tabular_branch(cfg, continuous_columns=CONT, categorical_columns=CAT, num_classes=2)
    assert isinstance(branch, BranchTrainer) and branch.branch == "tabular"
    out = tmp_path / "fold_1" / "tabular"
    fitted = branch.fit(FoldData(fold=1, fit=fit, selection=evaluation.iloc[:0], seed=7, output_dir=out))
    assert isinstance(fitted, FittedPredictor)
    probs = fitted.predict_proba(evaluation)
    assert probs.shape == (len(evaluation), 2) and np.allclose(probs.sum(axis=1), 1)
    names = {p.name for p in out.iterdir()}
    assert {"preprocessor.joblib", "model.joblib", "feature_names.txt", "selected_feature_names.txt",
            "tabular_branch.json"} <= names
    if selector == "lasso":
        assert {"selector.joblib", "coefficients.csv", "lasso_feature_selection.csv", "lasso_feature_importance.png"} <= names
        report = pd.read_csv(out / "lasso_feature_selection.csv")
        assert (report["fold"] == 1).all()
    if model == "xgboost":
        assert "model_feature_importance.csv" in names
    loaded = branch.load(out, fold=1)
    np.testing.assert_allclose(loaded.predict_proba(evaluation), probs)
    assert loaded.info["predictor_policy"]["version"] == "1.0"
    assert loaded.info["seed"] == 42 and loaded.info["fold_seed"] == 7      # legacy: global seed for every fold
    with pytest.raises(ValueError, match="fold"):
        branch.load(out, fold=2)
    # predictions use only fitted state: shuffling evaluation rows permutes outputs
    perm = evaluation.sample(frac=1, random_state=0)
    np.testing.assert_allclose(loaded.predict_proba(perm), probs[[evaluation.index.get_loc(i) for i in perm.index]])


def test_tabular_branch_ignores_selection_rows_and_rejects_forbidden(tmp_path):
    frame = make_frame(40)
    fit, selection = frame[frame[PATIENT_ID] < "P030"], frame[frame[PATIENT_ID] >= "P030"]
    branch = build_tabular_branch(BASE_CFG, continuous_columns=CONT, categorical_columns=CAT, num_classes=2)
    a = branch.fit(FoldData(1, fit, selection, 7, tmp_path / "a")).predict_proba(frame)
    b = branch.fit(FoldData(1, fit, selection.iloc[:0], 7, tmp_path / "b")).predict_proba(frame)
    np.testing.assert_allclose(a, b)
    with pytest.raises(PredictorPolicyError):
        build_tabular_branch(BASE_CFG, continuous_columns=CONT + ["oncotype_score"], categorical_columns=CAT, num_classes=2)
    with pytest.raises(ValueError, match="miss classes"):
        branch.fit(FoldData(1, fit[fit[LABEL] == 0], selection, 7, tmp_path / "c"))


def test_three_class_branch(tmp_path):
    frame = make_frame(45)
    frame[LABEL] = frame[PATIENT_ID].str[1:].astype(int) % 3
    branch = build_tabular_branch({**BASE_CFG, "feature_selector": "lasso"}, continuous_columns=CONT,
                                  categorical_columns=CAT, num_classes=3)
    fitted = branch.fit(FoldData(1, frame, frame.iloc[:0], 0, tmp_path))
    assert fitted.predict_proba(frame).shape == (len(frame), 3)
    assert {"coefficient_class_0", "coefficient_class_1", "coefficient_class_2"} <= set(pd.read_csv(tmp_path / "coefficients.csv").columns)


def test_lasso_stability_aggregation(tmp_path):
    def report(fold, rows):
        return pd.DataFrame([{"fold": fold, "feature_name": n, "feature_group": "clinical", "selected": s,
                              "lasso_importance": imp, "lasso_rank": r} for n, s, imp, r in rows])
    reports = [report(1, [("a", True, 0.5, 1), ("b", True, 0.2, 2), ("c", False, 0.0, 3)]),
               report(2, [("a", True, 0.3, 1), ("b", False, 0.0, 2)]),
               report(3, [("b", True, 0.9, 1), ("a", False, 0.0, 2)])]
    table = lasso_stability(reports)
    assert table["feature_name"].tolist() == ["b", "a", "c"]          # tie on frequency (2/3) -> mean importance
    row = table.set_index("feature_name").loc["c"]
    assert row["feature_occurrence_count"] == 1 and row["selection_frequency"] == 0
    a = table.set_index("feature_name").loc["a"]
    assert a["selection_count"] == 2 and a["selection_frequency"] == pytest.approx(2 / 3)
    assert a["mean_lasso_importance"] == pytest.approx(0.8 / 3) and a["max_lasso_importance"] == 0.5
    paths = save_lasso_stability(reports, tmp_path, top_n=5)
    assert all(p.is_file() for p in paths.values())
