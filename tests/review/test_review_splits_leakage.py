"""Independent review: patient splits, role separation, fold-local fitting, frozen plan identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from mribreast.artifacts import ArtifactMismatch
from mribreast.contracts import LABEL, PATIENT_ID, STUDY_ID
from mribreast.protocol import splits
from mribreast.protocol.train import run_training
from mribreast.protocol.validate import run_validation
from mribreast.tabular.lasso import LassoFeatureSelector
from mribreast.tabular.models import GroupedEarlyStoppingMLP
from mribreast.tabular.preprocessing import TabularPreprocessor
from tests.fakes import FakeTrainer, import_legacy, make_cfg, make_fake_cohort

FPS = {"clinical_features_file": "sha256:c", "metadata_file": "sha256:m"}


# ----------------------------------------------------------------------------- split plan
@pytest.mark.parametrize("mode", ["shared", "separate", "outer"])
@pytest.mark.parametrize("seed", [0, 7, 42, 123])
def test_roles_patient_disjoint_multi_study(mode, seed):
    cohort = make_fake_cohort(60, multi_study=True, seed=seed)
    plan = splits.make_split_plan(cohort.frame, make_cfg("train", num_folds=4, threshold_calibration_folds=3,
                                                         seed=seed, selection_split=mode, branches="image"))
    eval_patients_by_fold = []
    for fold in splits.fold_numbers(plan):
        part = plan[plan.fold == fold]
        by_role = {r: set(g.patient_id) for r, g in part.groupby("role")}
        roles = sorted(by_role)
        for i, a in enumerate(roles):
            for b in roles[i + 1:]:
                assert not (by_role[a] & by_role[b]), (fold, a, b)
        # selection / threshold roles never contain outer_eval patients except in 'outer'
        for r in splits.selection_roles(mode) + splits.threshold_roles(mode):
            if mode != "outer":
                assert not (set(part[part.role == r].patient_id) & by_role["outer_eval"])
        eval_patients_by_fold.append(by_role["outer_eval"])
        # every study of a patient lands in the same role
        assert part.groupby("patient_id")["role"].nunique().max() == 1
    for i in range(len(eval_patients_by_fold)):
        for j in range(i + 1, len(eval_patients_by_fold)):
            assert not (eval_patients_by_fold[i] & eval_patients_by_fold[j])


def test_separate_mode_partitions_shared_calibration():
    """separate: fit identical to shared; selection U threshold == shared calibration, disjoint."""
    cohort = make_fake_cohort(80, multi_study=True, seed=3)
    shared = splits.make_split_plan(cohort.frame, make_cfg("train", num_folds=3, threshold_calibration_folds=3,
                                                           selection_split="shared"))
    sep = splits.make_split_plan(cohort.frame, make_cfg("train", num_folds=3, threshold_calibration_folds=3,
                                                        selection_split="separate"))
    for fold in (1, 2, 3):
        s = shared[shared.fold == fold]
        p = sep[sep.fold == fold]
        assert set(s[s.role == "fit"].study_id) == set(p[p.role == "fit"].study_id)
        sel, thr = set(p[p.role == "selection"].patient_id), set(p[p.role == "threshold"].patient_id)
        assert not sel & thr
        assert sel | thr == set(s[s.role == "calibration"].patient_id)


def test_duke_style_ids_bit_identical_to_legacy_loop():
    """Legacy loop (validate_configurable_imaging_features_fusion.run_validation) with Duke-style string ids."""
    legacy = import_legacy("threshold_tuning")
    rng = np.random.default_rng(11)
    rows = []
    for i in rng.permutation(90):
        lab = int(i % 2)
        for j in range(1 + int(i) % 2):
            rows.append({PATIENT_ID: f"Breast_MRI_{i + 1:03d}", STUDY_ID: f"1.3.6.{i}.{j}", LABEL: lab})
    frame = pd.DataFrame(rows)
    plan = splits.make_split_plan(frame, make_cfg("train", num_folds=5, threshold_calibration_folds=5, seed=42))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(frame, frame[LABEL].to_numpy(), frame[PATIENT_ID].to_numpy()), 1):
        ot = frame.iloc[tr].reset_index(drop=True)
        fit, cal, _ = legacy.make_inner_calibration_split(ot[LABEL].to_numpy(np.int64), groups=ot[PATIENT_ID].to_numpy(),
                                                          requested_folds=5, random_state=42 + fold)
        part = plan[plan.fold == fold]
        assert part[part.role == "fit"].study_id.tolist() == ot.iloc[fit].study_id.tolist()
        assert part[part.role == "calibration"].study_id.tolist() == ot.iloc[cal].study_id.tolist()
        assert part[part.role == "outer_eval"].study_id.tolist() == frame.iloc[va].study_id.tolist()


# ----------------------------------------------------------------------------- train / validate data usage
class RecordingTrainer(FakeTrainer):
    """Records the study ids every predict_proba call sees during validation."""

    seen: list[list[str]]

    def load(self, directory, fold):
        predictor = super().load(directory, fold)
        inner = predictor.predict_proba
        log = self.seen

        def wrapped(frame):
            log.append((fold, frame[STUDY_ID].astype(str).tolist()))
            return inner(frame)

        predictor.predict_proba = wrapped
        return predictor


@pytest.mark.parametrize("mode", ["shared", "separate"])
def test_fit_selection_threshold_never_touch_outer_eval(tmp_path, mode):
    cohort = make_fake_cohort(48, multi_study=True, seed=1)
    trainer = RecordingTrainer("image", "signal_image")
    trainer.seen = []
    overrides = dict(branches="image", selection_split=mode, num_folds=3, threshold_calibration_folds=3)
    train_dir = run_training(make_cfg("train", **overrides), cohort, {"image": trainer},
                             checkpoint_root=tmp_path / "ck", input_fingerprints=FPS)
    plan = splits.load_split_plan(train_dir / "split_manifest.csv")
    pid = dict(zip(cohort.frame[STUDY_ID].astype(str), cohort.frame[PATIENT_ID].astype(str)))
    for call in trainer.calls:
        part = plan[plan.fold == call["fold"]]
        ev = set(part[part.role == "outer_eval"].patient_id)
        assert not {pid[s] for s in call["fit"]} & ev
        assert not {pid[s] for s in call["selection"]} & ev
        assert not {pid[s] for s in call["fit"]} & {pid[s] for s in call["selection"]}
    run_validation(make_cfg("validate", **overrides), cohort, {"image": trainer}, checkpoint_root=tmp_path / "ck",
                   validation_root=tmp_path / "val", input_fingerprints=FPS)
    # calls alternate: threshold frame, then outer_eval frame per fold
    for (fold, thr_ids), (fold2, ev_ids) in zip(trainer.seen[0::2], trainer.seen[1::2]):
        assert fold == fold2
        part = plan[plan.fold == fold]
        assert set(ev_ids) == set(part[part.role == "outer_eval"].study_id)
        assert not {pid[s] for s in thr_ids} & {pid[s] for s in ev_ids}
        assert set(thr_ids) == set(part[part.role.isin(splits.threshold_roles(mode))].study_id)


def test_tampered_split_manifest_rejected_by_validation(tmp_path):
    cohort = make_fake_cohort(40, multi_study=True, seed=2)
    overrides = dict(branches="image", num_folds=3, threshold_calibration_folds=3)
    trainers = {"image": FakeTrainer("image", "signal_image")}
    train_dir = run_training(make_cfg("train", **overrides), cohort, trainers, checkpoint_root=tmp_path / "ck",
                             input_fingerprints=FPS)
    path = train_dir / "split_manifest.csv"
    plan = splits.load_split_plan(path)
    # swap the roles of one fit patient and one outer_eval patient in fold 1 (still a "valid" plan shape)
    f1 = plan.fold == 1
    a = plan[f1 & (plan.role == "fit")].patient_id.iloc[0]
    b = plan[f1 & (plan.role == "outer_eval")].patient_id.iloc[0]
    plan.loc[f1 & (plan.patient_id == a), "role"] = "TMP"
    plan.loc[f1 & (plan.patient_id == b), "role"] = "fit"
    plan.loc[f1 & (plan.role == "TMP"), "role"] = "outer_eval"
    plan.to_csv(path, index=False, lineterminator="\n")
    with pytest.raises((ArtifactMismatch, splits.SplitError)):
        run_validation(make_cfg("validate", **overrides), cohort, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "val", input_fingerprints=FPS)


def test_validation_rejects_other_data(tmp_path):
    cohort = make_fake_cohort(40, multi_study=True, seed=2)
    overrides = dict(branches="image", num_folds=3, threshold_calibration_folds=3)
    trainers = {"image": FakeTrainer("image", "signal_image")}
    run_training(make_cfg("train", **overrides), cohort, trainers, checkpoint_root=tmp_path / "ck",
                 input_fingerprints=FPS)
    other = make_fake_cohort(40, multi_study=True, seed=2)
    other.frame.loc[0, "age"] = 99.0   # same studies, different content
    with pytest.raises(ArtifactMismatch, match="cohort fingerprint"):
        run_validation(make_cfg("validate", **overrides), other, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "val", input_fingerprints=FPS)


# ----------------------------------------------------------------------------- tabular fold-local fitting
def _tab_frame(n_patients=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        lab = i % 2
        for j in range(1 + i % 3):
            rows.append({PATIENT_ID: f"P{i}", LABEL: lab, "x1": rng.normal(lab, 1) if rng.random() > .2 else np.nan,
                         "x2": rng.normal(10 * i, 5), "c": rng.choice(["a", "b", "c"]) if i % 7 else None})
    return pd.DataFrame(rows)


def test_fold_local_lasso_refits_preprocessing_per_subfold_on_train_rows_only():
    frame = _tab_frame()
    pre = TabularPreprocessor(["x1", "x2"], ["c"]).fit(frame)
    X = pre.transform(frame)
    y = frame[LABEL].to_numpy()
    groups = frame[PATIENT_ID].to_numpy()
    sel = LassoFeatureSelector(cv_folds=3, cs=3, max_iter=200, n_jobs=1, random_state=0)
    sel.fit(X, y, groups=groups, raw=frame[["x1", "x2", "c"]],
            preprocessor_factory=lambda: TabularPreprocessor(["x1", "x2"], ["c"]))
    splits_ = sel.make_cv_splits(y, groups)
    assert len(sel.cv_details_) == len(splits_)
    for detail, (train, test) in zip(sel.cv_details_, splits_):
        np.testing.assert_array_equal(detail["train_index"], train)
        assert not set(groups[train]) & set(groups[test])
        p = detail["preprocessor"]
        assert p.n_fit_rows_ == len(train)
        scaler = p.transformer_.named_transformers_["continuous"].named_steps["scaler"]
        imputer = p.transformer_.named_transformers_["continuous"].named_steps["imputer"]
        sub = frame.iloc[train]
        np.testing.assert_allclose(imputer.statistics_, sub[["x1", "x2"]].median().to_numpy())
        filled = sub[["x1", "x2"]].fillna(sub[["x1", "x2"]].median())
        np.testing.assert_allclose(scaler.mean_, filled.mean().to_numpy(), rtol=1e-6)


def test_grouped_mlp_holdout_patients_disjoint():
    frame = _tab_frame(60, seed=4)
    X = TabularPreprocessor(["x1", "x2"], ["c"]).fit(frame).transform(frame)
    y = frame[LABEL].to_numpy()
    g = frame[PATIENT_ID].to_numpy()
    m = GroupedEarlyStoppingMLP(hidden_layer_sizes=(4,), max_iter=5, random_state=0, validation_fraction=0.15,
                                n_iter_no_change=2)
    m.fit(X, y, groups=g)
    assert not set(m.holdout_groups_) & set(m.train_groups_)
    assert set(m.holdout_groups_) | set(m.train_groups_) == set(g)


def test_validate_split_plan_checks_patient_ids_in_every_fold():
    """validate_split_plan compares plan patient ids with the cohort only for the FIRST fold (splits.py:240-244).

    A plan whose fold-2 rows relabel one study of a multi-study outer_eval patient with an alias patient id
    and move it to 'fit' passes validation although the real patient is then in fit AND outer_eval.
    (Reachable when an externally supplied / stale split_manifest.csv is reused by _resolve_plan.)
    """
    cohort = make_fake_cohort(40, multi_study=True, seed=5)
    frame = cohort.frame
    plan = splits.make_split_plan(frame, make_cfg("train", num_folds=3, threshold_calibration_folds=3))
    f2 = plan.fold == 2
    ev = plan[f2 & (plan.role == "outer_eval")]
    multi = ev.groupby("patient_id").filter(lambda g: len(g) > 1)
    victim_study = multi.study_id.iloc[0]
    # fold 2: the study moves to fit under an alias id; fold 3: it becomes outer_eval (each study still
    # outer_eval exactly once); fold 1 (the only fold whose ids are compared with the cohort) is untouched.
    plan.loc[f2 & (plan.study_id == victim_study), ["patient_id", "role"]] = ["ALIAS", "fit"]
    plan.loc[(plan.fold == 3) & (plan.study_id == victim_study), ["patient_id", "role"]] = ["ALIAS", "outer_eval"]
    real = multi.patient_id.iloc[0]
    part2 = plan[f2]
    assert real in set(part2[part2.role == "outer_eval"].patient_id)
    assert frame.set_index("study_id").loc[victim_study, "patient_id"] == real   # real owner of the fit study
    with pytest.raises(splits.SplitError):
        splits.validate_split_plan(plan, frame, check_labels=False)


def test_label_leakage_guard_only_catches_exact_copies():
    """check_label_leakage (predictor_policy.py:150-166) uses np.allclose(values, target) only.

    An allowed-prefix column that is an affine transform of the label / score, or the label as text,
    is not rejected. Documents the guard's limit (name policy is the primary defence).
    """
    from mribreast.data.predictor_policy import PredictorPolicyError, check_label_leakage

    frame = pd.DataFrame({LABEL: [0, 1] * 10, "score": np.linspace(5, 40, 20)})
    frame["kinetic_copy"] = frame[LABEL]
    frame["kinetic_flip"] = 1 - frame[LABEL]
    frame["kinetic_scaled"] = frame["score"] / 100.0
    frame["morphology_text"] = frame[LABEL].map({0: "low", 1: "high"})
    with pytest.raises(PredictorPolicyError):
        check_label_leakage(frame, ["kinetic_copy"], score=frame["score"])
    leaked = []
    for col in ("kinetic_flip", "kinetic_scaled", "morphology_text"):
        try:
            check_label_leakage(frame, [col], score=frame["score"])
            leaked.append(col)
        except PredictorPolicyError:
            pass
    assert leaked == ["kinetic_flip", "kinetic_scaled", "morphology_text"]   # passing = limitation confirmed


class DroppingTrainer(FakeTrainer):
    """Returns NaN rows (mask policy 'drop') for every study whose id ends with '.0' and patient index % 4 == 0."""

    def load(self, directory, fold):
        predictor = super().load(directory, fold)
        inner = predictor.predict_proba

        def wrapped(frame):
            p = inner(frame).copy()
            drop = frame[PATIENT_ID].str[1:].astype(int).to_numpy() % 4 == 0
            p[drop] = np.nan
            return p

        predictor.predict_proba = wrapped
        return predictor


def test_drop_policy_denominators_reported(tmp_path):
    cohort = make_fake_cohort(48, multi_study=True, seed=9)
    overrides = dict(branches="image,tabular", num_folds=3, threshold_calibration_folds=3)
    trainers = {"image": DroppingTrainer("image", "signal_image"), "tabular": FakeTrainer("tabular", "signal_tabular")}
    run_training(make_cfg("train", **overrides), cohort, trainers, checkpoint_root=tmp_path / "ck",
                 input_fingerprints=FPS)
    out = run_validation(make_cfg("validate", **overrides), cohort, trainers, checkpoint_root=tmp_path / "ck",
                         validation_root=tmp_path / "val", input_fingerprints=FPS)
    excluded = pd.read_csv(out / "excluded_studies.csv", dtype={STUDY_ID: str})
    dropped = set(cohort.frame.loc[cohort.frame[PATIENT_ID].str[1:].astype(int) % 4 == 0, STUDY_ID])
    assert set(excluded.loc[excluded.split == "outer_eval", STUDY_ID]) == dropped
    metrics = pd.read_csv(out / "validation_metrics.csv")
    assert (metrics.outer_validation_rows - metrics.excluded_outer_validation_studies
            == metrics.evaluated_outer_validation_studies).all()
    preds = pd.read_csv(out / "predictions_validation.csv", dtype={STUDY_ID: str})
    assert not set(preds[STUDY_ID]) & dropped                       # same denominator for every branch + fusion
    assert preds.groupby("branch")[STUDY_ID].nunique().nunique() == 1
