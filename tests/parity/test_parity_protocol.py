"""Parity of split/threshold/fusion/metric primitives with the reference repo on synthetic data.

Tolerances: split indices, effective fold counts, thresholds and their
records must be identical (no tolerance).  Fused probabilities: atol=1e-15
(both compute ``a*p + (1-a)*q`` and renormalise; observed 0).  Metrics:
atol=1e-12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from legacy_loader import import_legacy, load_file_module

pytestmark = pytest.mark.parity
splits = pytest.importorskip("mribreast.protocol.splits")
thresholds = pytest.importorskip("mribreast.protocol.thresholds")


def _cohort(seed: int, n_patients: int = 60, multi_study: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        label = int(rng.random() < 0.35)
        for s in range(1 + int(rng.random() < multi_study)):
            rows.append({"patient_id": f"Breast_MRI_{p:03d}", "study_id": f"1.2.{p}.{s}", "label": label})
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # legacy order dependence


@pytest.mark.parametrize("seed", range(6))
def test_inner_calibration_split_identical(legacy_repo, seed):
    legacy = load_file_module(legacy_repo, "mriBreastDuke/threshold_tuning.py")
    frame = _cohort(seed)
    labels, groups = frame["label"].to_numpy(), frame["patient_id"].to_numpy()
    for requested in (2, 5, 10):
        for grp in (groups, None):
            a = _outcome(legacy.make_inner_calibration_split, labels, grp, requested, seed)
            b = _outcome(splits.make_inner_calibration_split, labels, grp, requested, seed)
            assert a == b, (requested, grp is None)


def _outcome(func, labels, groups, requested, seed):
    """Indices as tuples, or the error class (both sides must fail identically, e.g. a class-less split)."""
    try:
        fit, cal, eff = func(labels, groups=groups, requested_folds=requested, random_state=seed)
    except ValueError as error:
        return ("ValueError", str(error).split(":")[0])
    return tuple(fit.tolist()), tuple(cal.tolist()), eff


@pytest.mark.parametrize("seed", [42, 7])
def test_split_plan_matches_legacy_fold_loop(legacy_repo, seed):
    """Legacy loop (n_fold_cv_run.run_5fold_cv / validate run_validation) transcribed with the legacy inner split."""
    legacy = load_file_module(legacy_repo, "mriBreastDuke/threshold_tuning.py")
    frame = _cohort(seed, n_patients=80)
    plan = splits.make_split_plan(frame, {"num_folds": 5, "threshold_calibration_folds": 5, "seed": seed,
                                          "selection_split": "shared"})
    labels, groups = frame["label"].to_numpy(), frame["patient_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(outer.split(frame, labels, groups), start=1):
        outer_train = frame.iloc[train_idx].reset_index(drop=True)
        fit_idx, cal_idx, eff = legacy.make_inner_calibration_split(
            outer_train["label"].to_numpy(), groups=outer_train["patient_id"].to_numpy(), requested_folds=5,
            random_state=seed + fold)
        got = plan[plan["fold"] == fold]
        roles = {r: set(got.loc[got["role"] == r, "study_id"]) for r in ("fit", "calibration", "outer_eval")}
        assert roles["fit"] == set(outer_train.iloc[fit_idx]["study_id"])
        assert roles["calibration"] == set(outer_train.iloc[cal_idx]["study_id"])
        assert roles["outer_eval"] == set(frame.iloc[val_idx]["study_id"])
        assert set(got["effective_inner_folds"]) == {eff}


@pytest.mark.parametrize("seed", range(20))
def test_threshold_selection_identical(legacy_repo, seed):
    legacy = load_file_module(legacy_repo, "mriBreastDuke/threshold_tuning.py")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 40))
    labels = np.r_[0, 1, rng.integers(0, 2, n - 2)]
    if seed % 3 == 0:    # heavy ties: few distinct scores, symmetric around 0.5
        probs = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7], n)
    else:
        probs = rng.random(n).round(int(rng.integers(1, 4)))
    assert thresholds.select_balanced_accuracy_threshold(labels, probs) == \
        legacy.select_balanced_accuracy_threshold(labels, probs)


def test_threshold_complete_tie_takes_lowest(legacy_repo):
    legacy = load_file_module(legacy_repo, "mriBreastDuke/threshold_tuning.py")
    labels, probs = np.array([0, 1]), np.array([0.4, 0.6])
    # 0.4 < t <= 0.6 all perfect; candidates 0.5 and 0.6 -> 0.5 wins on |t-0.5|
    assert legacy.select_balanced_accuracy_threshold(labels, probs)["threshold"] == 0.5
    labels, probs = np.array([1, 0]), np.array([0.4, 0.6])   # inverse ranking: 0.4 and 0.6 tie at |0.1| -> first (lowest)
    a = legacy.select_balanced_accuracy_threshold(labels, probs)
    assert thresholds.select_balanced_accuracy_threshold(labels, probs) == a


def test_fusion_and_metrics(legacy_repo):
    fusion = pytest.importorskip("mribreast.protocol.fusion")
    metrics = pytest.importorskip("mribreast.reporting.metrics")
    legacy = import_legacy(legacy_repo, "mriBreastDuke.classificators.xgboost_fusion")
    rng = np.random.default_rng(0)
    p_img = rng.dirichlet([1, 1], 50)
    p_tab = rng.dirichlet([1, 1], 50)
    labels = np.r_[0, 1, rng.integers(0, 2, 48)]
    for alpha in (0.0, 0.25, 0.5, 1.0):
        assert np.allclose(fusion.fuse_probabilities(p_img, p_tab, alpha), legacy.fuse_probabilities(p_img, p_tab, alpha),
                           rtol=0, atol=1e-15)
    for threshold in (0.3, 0.5, 0.71):
        old = legacy.probability_metrics(labels, p_img, "image", threshold=threshold)
        new = metrics.binary_metrics(labels, p_img[:, 1], threshold)
        for key, value in new.items():
            assert value == pytest.approx(old[f"image_{key}"], abs=1e-12), key
