"""Threshold selection: legacy reference (all tie-breaks), p >= thr, strategies, multiclass argmax."""

import numpy as np
import pytest

from mribreast.protocol import thresholds as th
from tests.fakes import import_legacy


def test_reference_against_legacy_on_tie_heavy_inputs():
    legacy = import_legacy("threshold_tuning")
    rng = np.random.default_rng(0)
    grids = ([0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.25, 0.5, 0.75, 1.0], [0.4, 0.6], [0.3, 0.7])
    compared = 0
    for trial in range(400):
        grid = grids[trial % len(grids)]
        n = int(rng.integers(2, 9))
        labels = rng.integers(0, 2, n)
        if len(np.unique(labels)) < 2:
            continue
        probs = rng.choice(grid, n) if trial % 3 else rng.random(n)
        assert th.select_balanced_accuracy_threshold(labels, probs) == legacy.select_balanced_accuracy_threshold(labels, probs)
        np.testing.assert_array_equal(th.predictions_at_threshold(probs, 0.5), legacy.predictions_at_threshold(probs, 0.5))
        compared += 1
    assert compared > 300


def test_documented_tie_breaks():
    # balanced accuracy first
    assert th.select_balanced_accuracy_threshold(np.array([0, 0, 1, 1]), np.array([0.6, 0.65, 0.8, 0.9]))["threshold"] == 0.8
    # equal balanced accuracy -> higher sensitivity: t=0 (all positive, sens 1) beats t=1 (sens 0)
    r = th.select_balanced_accuracy_threshold(np.array([1, 0]), np.array([0.4, 0.6]))
    # equal BA and sensitivity (t=0 and t=0.4) -> closest to 0.5
    assert r["threshold"] == 0.4 and r["sensitivity"] == 1.0 and r["balanced_accuracy"] == 0.5
    # all scores 1.0: t in {0, 0.5, 1} identical -> 0.5 (distance 0)
    assert th.select_balanced_accuracy_threshold(np.array([0, 1]), np.array([1.0, 1.0]))["threshold"] == 0.5
    # float asymmetry of |t-0.5|: 0.7 is (in float64) closer to 0.5 than 0.3
    assert -abs(0.7 - 0.5) > -abs(0.3 - 0.5)


def test_decision_is_greater_or_equal():
    probs = np.array([[0.6, 0.4], [0.4, 0.6], [0.3, 0.7]])
    np.testing.assert_array_equal(th.decide(probs, 0.6), [0, 1, 1])
    np.testing.assert_array_equal(th.decide(probs, 0.6000001), [0, 0, 1])
    # positive_class=0 maps decisions back to class ids
    np.testing.assert_array_equal(th.decide(probs, 0.6, positive_class=0), [0, 1, 1])


def test_rejects_non_binary_and_bad_inputs():
    with pytest.raises(ValueError, match="binary labels"):
        th.select_balanced_accuracy_threshold(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))
    with pytest.raises(ValueError):
        th.predictions_at_threshold(np.array([0.2, 1.2]), 0.5)
    with pytest.raises(ValueError):
        th.decide(np.array([[0.5, 0.5]]), None)


def test_fixed_strategy_and_table():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.45, 0.55], [0.4, 0.6], [0.2, 0.8]])
    thresholds, table = th.tune_branch_thresholds("fixed_0.5", labels, {"image": probs}, 2)
    assert thresholds == {"image": 0.5}
    assert table.loc[0, "specificity"] == 0.5 and table.loc[0, "sensitivity"] == 1.0
    thresholds, table = th.tune_branch_thresholds("balanced_accuracy", labels, {"image": probs, "fusion": probs}, 2)
    assert thresholds["image"] == 0.6 and list(table.branch) == ["image", "fusion"]
    assert list(table.columns[:5]) == ["branch", "threshold", "balanced_accuracy", "sensitivity", "specificity"]


def test_multiclass_uses_argmax_without_threshold():
    labels = np.array([0, 1, 2, 2])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.4, 0.35, 0.25]])
    thresholds, table = th.tune_branch_thresholds("fixed_0.5", labels, {"image": probs}, 3)
    assert thresholds == {"image": None} and table.loc[0, "strategy"] == "argmax"
    np.testing.assert_array_equal(th.decide(probs, None), [0, 1, 2, 0])
