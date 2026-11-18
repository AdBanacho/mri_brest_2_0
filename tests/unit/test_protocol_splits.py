"""Split plan: legacy bit-identity, determinism, patient disjointness, roles, persistence."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from mribreast.artifacts import file_fingerprint
from mribreast.protocol import splits
from tests.fakes import import_legacy, make_cfg, make_fake_cohort


def _roles(plan, fold):
    part = plan[plan.fold == fold]
    return {role: set(g.patient_id) for role, g in part.groupby("role")}


def test_inner_split_matches_legacy_bit_for_bit():
    legacy = import_legacy("threshold_tuning")
    rng = np.random.default_rng(3)
    for trial in range(25):
        n = int(rng.integers(20, 80))
        groups = np.array([f"p{int(g)}" for g in rng.integers(0, n // 2, n)])
        group_label = {g: int(rng.integers(0, 2)) for g in np.unique(groups)}
        labels = np.array([group_label[g] for g in groups])
        for kwargs in ({"groups": groups}, {"groups": None}):
            try:
                expected = legacy.make_inner_calibration_split(labels, requested_folds=5, random_state=42 + trial, **kwargs)
            except (ValueError, RuntimeError) as exc:
                with pytest.raises(type(exc)):
                    splits.make_inner_calibration_split(labels, requested_folds=5, random_state=42 + trial, **kwargs)
                continue
            got = splits.make_inner_calibration_split(labels, requested_folds=5, random_state=42 + trial, **kwargs)
            np.testing.assert_array_equal(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])
            assert got[2] == expected[2]


def test_plan_reproduces_legacy_fold_loop():
    """Legacy: outer SGKF(seed) over cohort rows, inner split of outer-train rows with seed+fold."""
    legacy = import_legacy("threshold_tuning")
    cohort = make_fake_cohort(60, multi_study=True, seed=5)
    frame = cohort.frame
    cfg = make_cfg("train", num_folds=5, threshold_calibration_folds=5, seed=42)
    plan = splits.make_split_plan(frame, cfg)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    labels, groups = frame.label.to_numpy(), frame.patient_id.to_numpy()
    for fold, (train_idx, val_idx) in enumerate(outer.split(frame, labels, groups), start=1):
        outer_train = frame.iloc[train_idx].reset_index(drop=True)
        fit_idx, cal_idx, eff = legacy.make_inner_calibration_split(
            outer_train.label.to_numpy(), groups=outer_train.patient_id.to_numpy(), requested_folds=5,
            random_state=42 + fold)
        part = plan[plan.fold == fold]
        assert part[part.role == "fit"].study_id.tolist() == outer_train.iloc[fit_idx].study_id.tolist()
        assert part[part.role == "calibration"].study_id.tolist() == outer_train.iloc[cal_idx].study_id.tolist()
        assert part[part.role == "outer_eval"].study_id.tolist() == frame.iloc[val_idx].study_id.tolist()
        assert set(part.effective_inner_folds) == {eff} and set(part.inner_seed) == {42 + fold}


def test_plan_is_deterministic_and_patient_disjoint_with_multi_study_patients():
    cohort = make_fake_cohort(45, multi_study=True, seed=1)
    assert cohort.frame.patient_id.value_counts().max() == 3
    cfg = make_cfg("train")
    a = splits.make_split_plan(cohort.frame, cfg)
    b = splits.make_split_plan(cohort.frame, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert splits.split_hash(a) == splits.split_hash(b)
    for fold in splits.fold_numbers(a):
        roles = _roles(a, fold)
        assert not roles["fit"] & roles["calibration"]
        assert not roles["fit"] & roles["outer_eval"]
        assert not roles["calibration"] & roles["outer_eval"]
        # every study of a patient lands in the same role
        part = a[a.fold == fold]
        assert (part.groupby("patient_id").role.nunique() == 1).all()
    eval_rows = a[a.role == "outer_eval"]
    assert eval_rows.study_id.is_unique and set(eval_rows.study_id) == set(cohort.frame.study_id)
    other = splits.make_split_plan(cohort.frame, make_cfg("train", seed=7))
    assert splits.split_hash(other) != splits.split_hash(a)


def test_effective_inner_folds_recorded_when_requested_exceeds_groups():
    cohort = make_fake_cohort(30, seed=2)  # few patients: requested 20 inner folds cannot be honoured
    plan = splits.make_split_plan(cohort.frame, make_cfg("train", threshold_calibration_folds=20))
    assert set(plan.requested_inner_folds) == {20}
    for fold in splits.fold_numbers(plan):
        part = plan[plan.fold == fold]
        outer_train = part[part.role != "outer_eval"]
        smallest = outer_train.groupby("label").patient_id.nunique().min()
        assert set(part.effective_inner_folds) == {min(20, smallest)}


@pytest.mark.parametrize("mode,roles", [("shared", {"fit", "calibration", "outer_eval"}),
                                        ("separate", {"fit", "selection", "threshold", "outer_eval"})])
def test_selection_split_roles(mode, roles):
    cohort = make_fake_cohort(60, multi_study=True, seed=4)
    plan = splits.make_split_plan(cohort.frame, make_cfg("train", selection_split=mode))
    assert set(plan.role) == roles
    if mode == "separate":
        for fold in splits.fold_numbers(plan):
            r = _roles(plan, fold)
            assert not r["selection"] & r["threshold"]
            assert not (r["selection"] | r["threshold"]) & r["fit"]


def test_outer_mode_fits_on_whole_outer_train():
    cohort = make_fake_cohort(30, seed=4)
    plan = splits.make_split_plan(cohort.frame, make_cfg("train", selection_split="outer", branches="image"))
    assert set(plan.role) == {"fit", "outer_eval"}
    assert set(plan.effective_inner_folds) == {0} and set(plan.inner_seed) == {-1}
    for fold in splits.fold_numbers(plan):
        part = plan[plan.fold == fold]
        assert len(part[part.role == "fit"]) + len(part[part.role == "outer_eval"]) == len(cohort.frame)


def test_validators_reject_leaks_and_label_conflicts():
    cohort = make_fake_cohort(30, multi_study=True, seed=0)
    plan = splits.make_split_plan(cohort.frame, make_cfg("train"))
    leaked = plan.copy()
    victim = leaked[(leaked.fold == 1) & (leaked.role == "outer_eval")].patient_id.iloc[0]
    other_study = leaked[(leaked.fold == 1) & (leaked.patient_id == victim)].index
    # move one of this patient's rows to fit in fold 1 -> leakage
    leaked.loc[other_study[:1], "role"] = "fit"
    multi = cohort.frame.patient_id.value_counts()
    if multi[victim] > 1:
        with pytest.raises(splits.SplitError, match="both"):
            splits.validate_split_plan(leaked)
    conflicting = cohort.frame.copy()
    first = conflicting.index[conflicting.patient_id == multi.index[0]]
    conflicting.loc[first[0], "label"] = 1 - conflicting.loc[first[0], "label"]
    with pytest.raises(splits.SplitError, match="more than one label"):
        splits.make_split_plan(conflicting, make_cfg("train"))
    with pytest.raises(splits.SplitError, match="labels differ"):
        relabelled = cohort.frame.copy()
        relabelled["label"] = 1 - relabelled["label"]
        splits.validate_split_plan(plan, relabelled)


def test_missing_class_in_role_rejected():
    cohort = make_fake_cohort(30, seed=0)
    plan = splits.make_split_plan(cohort.frame, make_cfg("train"))
    broken = plan.copy()
    cal = broken[(broken.fold == 1) & (broken.role == "calibration")]
    broken = broken.drop(cal[cal.label == 1].index)
    with pytest.raises(splits.SplitError):
        splits.validate_split_plan(broken)


def test_save_load_roundtrip_hash_equals_file_fingerprint(tmp_path):
    cohort = make_fake_cohort(30, multi_study=True)
    plan = splits.make_split_plan(cohort.frame, make_cfg("train"))
    path = tmp_path / "split_manifest.csv"
    h = splits.save_split_plan(plan, path)
    assert h == splits.split_hash(plan) == file_fingerprint(path)
    loaded = splits.load_split_plan(path)
    assert splits.split_hash(loaded) == h
    assert list(loaded.columns) == list(splits.SPLIT_COLUMNS)
