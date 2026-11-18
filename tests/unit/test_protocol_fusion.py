"""Fusion alignment checks and the alpha extremes."""

import numpy as np
import pandas as pd
import pytest

from mribreast.protocol.fusion import AlignmentError, align_to_reference, branch_frame, fuse_aligned, fuse_probabilities
from tests.fakes import import_legacy


def _frame(ids, probs):
    return branch_frame(ids, np.asarray(probs, dtype=float), np.asarray(probs).shape[1], "x")


def test_alignment_reorders_by_study_id():
    ref = ["a", "b", "c"]
    tab = _frame(["c", "a", "b"], [[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    np.testing.assert_allclose(align_to_reference(ref, tab, 2, "tabular"), [[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])


@pytest.mark.parametrize("ids,match", [(["a", "b"], "missing"), (["a", "b", "b"], "duplicate"),
                                       (["a", "b", "c", "d"], "unexpected")])
def test_alignment_rejects_missing_duplicate_extra(ids, match):
    probs = np.tile([0.5, 0.5], (len(ids), 1))
    frame = pd.DataFrame({"study_id": ids, "prob_0": probs[:, 0], "prob_1": probs[:, 1]})
    with pytest.raises(AlignmentError, match=match):
        align_to_reference(["a", "b", "c"], frame, 2, "image")


def test_alignment_rejects_class_order_and_bad_rows():
    frame = pd.DataFrame({"study_id": ["a"], "prob_1": [0.2], "prob_0": [0.8]})
    with pytest.raises(AlignmentError, match="class columns"):
        align_to_reference(["a"], frame, 2, "image")
    frame = pd.DataFrame({"study_id": ["a"], "prob_0": [0.2], "prob_1": [0.2]})
    with pytest.raises(AlignmentError, match="sum to 1"):
        align_to_reference(["a"], frame, 2, "image")
    with pytest.raises(AlignmentError, match="shape"):
        branch_frame(["a", "b"], np.array([[0.5, 0.5]]), 2, "image")
    with pytest.raises(AlignmentError):
        branch_frame(["a"], np.array([[0.2, 0.3, 0.5]]), 2, "image")


def test_alpha_extremes_equal_single_branch():
    rng = np.random.default_rng(0)
    img = rng.dirichlet([1, 1], 50)
    tab = rng.dirichlet([1, 1], 50)
    np.testing.assert_allclose(fuse_probabilities(img, tab, 0.0), tab, rtol=0, atol=1e-15)
    np.testing.assert_allclose(fuse_probabilities(img, tab, 1.0), img, rtol=0, atol=1e-15)
    ids = [str(i) for i in range(50)]
    fused = fuse_aligned(ids, {"image": _frame(ids, img), "tabular": _frame(ids, tab)}, 2, 0.3)
    np.testing.assert_allclose(fused["fusion"], 0.3 * img + 0.7 * tab, atol=1e-15)
    with pytest.raises(ValueError):
        fuse_probabilities(img, tab, 1.5)


def test_fusion_matches_legacy():
    legacy = import_legacy("threshold_tuning")  # ensures repo on sys.path
    try:
        from mriBreastDuke.classificators.xgboost_fusion import fuse_probabilities as legacy_fuse
    except ImportError as exc:  # classificators package imports torch/monai
        pytest.skip(f"legacy fusion not importable: {exc}")
    assert legacy is not None
    rng = np.random.default_rng(1)
    img, tab = rng.dirichlet([1, 1, 1], 20), rng.dirichlet([1, 1, 1], 20)
    for alpha in (0.0, 0.25, 0.5, 1.0):
        np.testing.assert_array_equal(fuse_probabilities(img, tab, alpha), legacy_fuse(img, tab, alpha))
