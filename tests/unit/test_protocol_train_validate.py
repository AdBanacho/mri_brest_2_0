"""run_training / run_validation with fake branches: artifacts, resume, rejection, frozen split, modes."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from mribreast.artifacts import ArtifactMismatch, is_complete, read_json, write_json_atomic
from mribreast.config import schema
from mribreast.contracts import prob_columns
from mribreast.protocol import splits
from mribreast.protocol.manifests import directory_fingerprints
from mribreast.protocol.train import run_training
from mribreast.protocol.validate import run_validation
from mribreast.reporting.metrics import confusion
from tests.fakes import FakeTrainer, make_cfg, make_fake_cohort, make_trainers

FPS = {"clinical_features_file": "sha256:clinical", "metadata_file": "sha256:meta"}


def _run(tmp_path, cohort=None, trainers=None, root="ck", fps=FPS, **overrides):
    cohort = cohort or make_fake_cohort(40, multi_study=True)
    branches = overrides.get("branches", "image,tabular")
    trainers = trainers or make_trainers(tuple(branches.split(",")), num_classes=cohort.label.num_classes)
    train_dir = run_training(make_cfg("train", **overrides), cohort, trainers, checkpoint_root=tmp_path / root,
                             input_fingerprints=fps)
    out = run_validation(make_cfg("validate", **overrides), cohort, trainers, checkpoint_root=tmp_path / root,
                         validation_root=tmp_path / f"val_{root}", input_fingerprints=fps)
    return train_dir, out, trainers, cohort


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("default"))


def test_artifact_layout_and_manifests(default_run):
    train_dir, out, _, cohort = default_run
    for name in ("train_manifest.json", "cohort.csv", "cohort_audit.csv", "split_manifest.csv", "COMPLETE"):
        assert (train_dir / name).is_file(), name
    manifest = read_json(train_dir / "train_manifest.json")
    for key in ("schema_version", "train_id", "code_version", "environment", "resolved_config", "label_definition",
                "input_fingerprints", "cohort_fingerprint", "split_hash", "branches", "num_classes"):
        assert key in manifest
    assert "oncotype_score" not in pd.read_csv(train_dir / "cohort.csv").columns
    for fold in (1, 2, 3):
        fm = read_json(train_dir / f"fold_{fold}" / "fold_manifest.json")
        assert fm["train_id"] == manifest["train_id"] and fm["split_hash"] == manifest["split_hash"]
        assert set(fm["branch_artifacts"]) == {"image", "tabular"}
    expected = {"evaluation_manifest.json", "validation_config.json", "validation_metrics.csv", "validation_summary.csv",
                "decision_thresholds.json", "threshold_calibration_metrics.csv", "predictions_calibration.csv",
                "predictions_validation.csv", "branch_comparison.csv", "confusion_matrices.json", "validation_report.md",
                "COMPLETE"} | {f"{k}_{b}.png" for k in ("roc", "confusion") for b in ("image", "tabular", "fusion")}
    assert expected <= {p.name for p in out.iterdir()}
    ev = read_json(out / "evaluation_manifest.json")
    assert ev["train_id"] == manifest["train_id"] and ev["selection_uses_outer_eval"] is False
    assert ev["roles"] == {"fit": ["fit"], "selection": ["calibration"], "threshold": ["calibration"],
                           "evaluation": ["outer_eval"]}
    assert out.parent.name == manifest["train_id"] and out.name.startswith("ev-a0.5-")
    metrics = pd.read_csv(out / "validation_metrics.csv")
    legacy_head = ["fold", "checkpoint_path", "inner_fit_rows", "inner_calibration_rows",
                   "effective_inner_calibration_folds", "outer_validation_rows"]
    assert list(metrics.columns[:6]) == legacy_head
    for branch in ("image", "tabular", "fusion"):
        assert [c for c in metrics.columns if c.startswith(branch + "_")] == [
            f"{branch}_{m}" for m in ("accuracy", "balanced_accuracy", "sensitivity", "decision_threshold",
                                      "specificity", "auc_roc")]
    assert {"run_id"} <= set(metrics.columns)
    assert metrics.outer_validation_rows.sum() == len(cohort.frame)
    preds = pd.read_csv(out / "predictions_validation.csv")
    assert list(preds.columns) == ["fold", "patient_id", "study_id", "label", "branch", "split", "prob_0", "prob_1",
                                   "decision"]
    report = (out / "validation_report.md").read_text(encoding="utf-8")
    assert "independent test set" in report and "selection_uses_outer_eval" in report


def test_pooled_confusion_uses_per_fold_thresholds(default_run):
    _, out, _, _ = default_run
    thresholds = read_json(out / "decision_thresholds.json")["folds"]
    preds = pd.read_csv(out / "predictions_validation.csv")
    cms = read_json(out / "confusion_matrices.json")["branches"]
    for branch in ("image", "tabular", "fusion"):
        part = preds[preds.branch == branch]
        for fold, group in part.groupby("fold"):
            thr = thresholds[str(fold)][branch]
            np.testing.assert_array_equal(group.decision, (group.prob_1 >= thr).astype(int))
        pooled = confusion(part.label.to_numpy(), part.decision.to_numpy(), 2).tolist()
        assert cms[branch]["pooled"] == pooled
        assert np.array_equal(np.sum([cms[branch]["folds"][f] for f in cms[branch]["folds"]], axis=0), pooled)
    distinct = {round(thresholds[f]["fusion"], 12) for f in thresholds}
    assert len(distinct) > 1  # genuinely fold-specific thresholds
    summary = pd.read_csv(out / "validation_summary.csv").set_index("metric")
    fusion = preds[preds.branch == "fusion"]
    acc = float((fusion.label == fusion.decision).mean())
    assert summary.loc["fusion_accuracy", "pooled"] == pytest.approx(acc)


def test_thresholds_tuned_on_calibration_only(default_run):
    train_dir, out, _, _ = default_run
    plan = splits.load_split_plan(train_dir / "split_manifest.csv")
    cal = pd.read_csv(out / "predictions_calibration.csv", dtype={"study_id": str})
    for fold, group in cal.groupby("fold"):
        expected = set(plan[(plan.fold == fold) & (plan.role == "calibration")].study_id)
        assert set(group.study_id) == expected
    table = pd.read_csv(out / "threshold_calibration_metrics.csv")
    thresholds = read_json(out / "decision_thresholds.json")["folds"]
    for row in table.itertuples():
        assert row.threshold == pytest.approx(thresholds[str(row.fold)][row.branch])


@pytest.mark.parametrize("alpha,twin", [(0.0, "tabular"), (1.0, "image")])
def test_alpha_extremes_match_single_branch(tmp_path, alpha, twin):
    _, out, _, _ = _run(tmp_path, fusion_alpha=alpha)
    metrics = pd.read_csv(out / "validation_metrics.csv")
    for m in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc_roc", "decision_threshold"):
        np.testing.assert_allclose(metrics[f"fusion_{m}"], metrics[f"{twin}_{m}"], atol=1e-12)
    preds = pd.read_csv(out / "predictions_validation.csv")
    np.testing.assert_array_equal(preds[preds.branch == "fusion"].decision.to_numpy(),
                                  preds[preds.branch == twin].decision.to_numpy())


def test_resume_after_interrupted_fold(tmp_path):
    cohort = make_fake_cohort(40, multi_study=True)
    cfg = make_cfg("train")
    crashing = make_trainers(fail_on_fold=2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_training(cfg, cohort, crashing, checkpoint_root=tmp_path, input_fingerprints=FPS)
    train_dir = tmp_path / schema.train_id(cfg, FPS)
    assert is_complete(train_dir / "fold_1") and not (train_dir / "fold_2").exists()
    assert not is_complete(train_dir)
    assert not [p for p in train_dir.iterdir() if ".staging-" in p.name]
    fold1_mtime = (train_dir / "fold_1" / "image" / "fake_image.json").stat().st_mtime_ns
    fresh = make_trainers()
    run_training(cfg, cohort, fresh, checkpoint_root=tmp_path, input_fingerprints=FPS)
    assert [c["fold"] for c in fresh["image"].calls] == [2, 3]
    assert [c["fold"] for c in fresh["tabular"].calls] == [2, 3]
    assert (train_dir / "fold_1" / "image" / "fake_image.json").stat().st_mtime_ns == fold1_mtime
    assert is_complete(train_dir)
    again = make_trainers()
    run_training(cfg, cohort, again, checkpoint_root=tmp_path, input_fingerprints=FPS)
    assert again["image"].calls == []


def test_rejects_artifacts_from_another_configuration(tmp_path):
    train_dir, _, trainers, cohort = _run(tmp_path)
    other_cfg = make_cfg("train", lr=0.5)
    other_dir = tmp_path / "ck" / schema.train_id(other_cfg, FPS)
    shutil.copytree(train_dir, other_dir)  # artifacts of config A placed under config B's id
    with pytest.raises(ArtifactMismatch, match="train_id"):
        run_training(other_cfg, cohort, trainers, checkpoint_root=tmp_path / "ck", input_fingerprints=FPS)
    with pytest.raises(ArtifactMismatch, match="train_id"):
        run_validation(make_cfg("validate", lr=0.5), cohort, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "v2", input_fingerprints=FPS)
    # changed cohort (data) under the same train_id
    changed = make_fake_cohort(40, multi_study=True, seed=9)
    with pytest.raises(ArtifactMismatch, match="cohort fingerprint"):
        run_training(make_cfg("train"), changed, trainers, checkpoint_root=tmp_path / "ck", input_fingerprints=FPS)
    # tampered label definition
    manifest = read_json(train_dir / "train_manifest.json")
    manifest["label_definition"]["cutoffs"] = [20.0]
    write_json_atomic(train_dir / "train_manifest.json", manifest)
    with pytest.raises(ArtifactMismatch, match="label definition"):
        run_validation(make_cfg("validate"), cohort, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "v3", input_fingerprints=FPS)


def test_rejects_modified_split_or_fold_artifacts(tmp_path):
    train_dir, _, trainers, cohort = _run(tmp_path)
    artifact = train_dir / "fold_2" / "tabular" / "fake_tabular.json"
    params = json.loads(artifact.read_text())
    params["coef"][1][0] += 1.0
    artifact.write_text(json.dumps(params))
    with pytest.raises(ArtifactMismatch, match="changed since training"):
        run_validation(make_cfg("validate", fusion_alpha=0.3), cohort, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "v", input_fingerprints=FPS)
    split = train_dir / "split_manifest.csv"
    text = split.read_text().splitlines()
    split.write_text("\n".join(text[:1] + text[2:] + text[1:2]) + "\n")  # reorder rows -> different hash
    with pytest.raises(ArtifactMismatch, match="split_hash"):
        run_training(make_cfg("train"), cohort, trainers, checkpoint_root=tmp_path / "ck", input_fingerprints=FPS)


def test_existing_split_manifest_is_frozen_and_reused(tmp_path):
    cohort = make_fake_cohort(40, multi_study=True)
    cfg = make_cfg("train")
    train_dir = tmp_path / schema.train_id(cfg, FPS)
    train_dir.mkdir(parents=True)
    foreign = splits.make_split_plan(cohort.frame, make_cfg("train", seed=7))
    h = splits.save_split_plan(foreign, train_dir / "split_manifest.csv")
    assert h != splits.split_hash(splits.make_split_plan(cohort.frame, cfg))
    run_training(cfg, cohort, make_trainers(), checkpoint_root=tmp_path, input_fingerprints=FPS)
    manifest = read_json(train_dir / "train_manifest.json")
    assert manifest["split_hash"] == h and manifest["split_reused"] is True


def test_frozen_split_outer_eval_labels_only_change_evaluation(tmp_path):
    """Swap labels of fold-1 outer-eval patients: fold-1 fitted artifacts, selection data and thresholds are
    unchanged; only fold-1 evaluation metrics change."""
    cohort = make_fake_cohort(40, multi_study=True)
    train_a, out_a, _, _ = _run(tmp_path, cohort=cohort, root="a")
    plan = splits.load_split_plan(train_a / "split_manifest.csv")
    eval_patients = set(plan[(plan.fold == 1) & (plan.role == "outer_eval")].patient_id)
    swapped_frame = cohort.frame.copy()
    mask = swapped_frame.patient_id.isin(eval_patients)
    swapped_frame.loc[mask, "label"] = 1 - swapped_frame.loc[mask, "label"]
    swapped = make_fake_cohort(40, multi_study=True)
    swapped.frame = swapped_frame
    train_b_dir = tmp_path / "b" / train_a.name
    train_b_dir.mkdir(parents=True)
    shutil.copyfile(train_a / "split_manifest.csv", train_b_dir / "split_manifest.csv")  # frozen split
    train_b, out_b, _, _ = _run(tmp_path, cohort=swapped, root="b")
    assert train_b == train_b_dir
    assert (train_b / "split_manifest.csv").read_bytes() == (train_a / "split_manifest.csv").read_bytes()
    for branch in ("image", "tabular"):
        assert directory_fingerprints(train_a / "fold_1" / branch) == directory_fingerprints(train_b / "fold_1" / branch)
        # other folds DO see these patients during fitting -> the test is sensitive
        assert directory_fingerprints(train_a / "fold_2" / branch) != directory_fingerprints(train_b / "fold_2" / branch)
    thr_a = read_json(out_a / "decision_thresholds.json")["folds"]
    thr_b = read_json(out_b / "decision_thresholds.json")["folds"]
    assert thr_a["1"] == thr_b["1"]
    cal_a = pd.read_csv(out_a / "predictions_calibration.csv")
    cal_b = pd.read_csv(out_b / "predictions_calibration.csv")
    pd.testing.assert_frame_equal(cal_a[cal_a.fold == 1], cal_b[cal_b.fold == 1])
    m_a = pd.read_csv(out_a / "validation_metrics.csv").set_index("fold")
    m_b = pd.read_csv(out_b / "validation_metrics.csv").set_index("fold")
    for branch in ("image", "tabular", "fusion"):
        assert m_a.loc[1, f"{branch}_decision_threshold"] == m_b.loc[1, f"{branch}_decision_threshold"]
        assert m_a.loc[1, f"{branch}_auc_roc"] != pytest.approx(m_b.loc[1, f"{branch}_auc_roc"])
    pv_a = pd.read_csv(out_a / "predictions_validation.csv")
    pv_b = pd.read_csv(out_b / "predictions_validation.csv")
    f1a, f1b = pv_a[pv_a.fold == 1], pv_b[pv_b.fold == 1]
    np.testing.assert_array_equal(f1a[prob_columns(2) + ["decision"]].to_numpy(), f1b[prob_columns(2) + ["decision"]].to_numpy())
    assert (f1a.label.to_numpy() != f1b.label.to_numpy()).all()


@pytest.mark.parametrize("mode", ["separate", "outer"])
def test_selection_split_modes_feed_branches_and_flags(tmp_path, mode):
    branches = "image" if mode == "outer" else "image,tabular"
    cohort = make_fake_cohort(60, multi_study=True, seed=3)
    train_dir, out, trainers, _ = _run(tmp_path, cohort=cohort, selection_split=mode, branches=branches)
    plan = splits.load_split_plan(train_dir / "split_manifest.csv")
    sel_role = {"separate": "selection", "outer": "outer_eval"}[mode]
    thr_role = {"separate": "threshold", "outer": "outer_eval"}[mode]
    for call in trainers["image"].calls:
        part = plan[plan.fold == call["fold"]]
        assert set(call["selection"]) == set(part[part.role == sel_role].study_id)
        assert set(call["fit"]) == set(part[part.role == "fit"].study_id)
    ev = read_json(out / "evaluation_manifest.json")
    assert ev["selection_uses_outer_eval"] is (mode == "outer")
    assert ev["threshold_uses_outer_eval"] is (mode == "outer")
    cal = pd.read_csv(out / "predictions_calibration.csv", dtype={"study_id": str})
    for fold, group in cal.groupby("fold"):
        assert set(group.study_id) == set(plan[(plan.fold == fold) & (plan.role == thr_role)].study_id)
    if mode == "outer":
        assert "optimistic" in (out / "validation_report.md").read_text(encoding="utf-8")


def test_three_class_flow_uses_argmax(tmp_path):
    cohort = make_fake_cohort(54, multi_study=True, num_classes=3, seed=2)
    _, out, _, _ = _run(tmp_path, cohort=cohort, label_scheme="oncotype_3class_18_31", branches="image")
    metrics = pd.read_csv(out / "validation_metrics.csv")
    assert "image_decision_threshold" not in metrics.columns
    assert {"image_accuracy", "image_balanced_accuracy", "image_sensitivity", "image_specificity",
            "image_auc_roc"} <= set(metrics.columns)
    preds = pd.read_csv(out / "predictions_validation.csv")
    np.testing.assert_array_equal(preds.decision, preds[prob_columns(3)].to_numpy().argmax(axis=1))
    thresholds = read_json(out / "decision_thresholds.json")
    assert thresholds["threshold_strategy"] == "argmax"
    assert all(v["image"] is None for v in thresholds["folds"].values())
    cms = read_json(out / "confusion_matrices.json")
    assert np.array(cms["branches"]["image"]["pooled"]).shape == (3, 3)
    assert read_json(out / "evaluation_manifest.json")["decision_rule"] == "argmax"


def test_patient_report_unit_aggregates_studies(tmp_path):
    cohort = make_fake_cohort(40, multi_study=True)
    _, study_out, _, _ = _run(tmp_path, cohort=cohort, root="s")
    _, patient_out, _, _ = _run(tmp_path, cohort=cohort, root="s", report_unit="patient")
    s = pd.read_csv(study_out / "predictions_validation.csv", dtype={"study_id": str})
    p = pd.read_csv(patient_out / "predictions_validation.csv", dtype={"study_id": str})
    assert len(p) == cohort.frame.patient_id.nunique() * 3  # 3 branches
    for (patient, branch), group in s.groupby(["patient_id", "branch"]):
        row = p[(p.patient_id == patient) & (p.branch == branch)]
        assert len(row) == 1
        assert row.prob_1.iloc[0] == pytest.approx(group.prob_1.mean())
        assert row.study_id.iloc[0] == "|".join(group.study_id)
    metrics = pd.read_csv(patient_out / "validation_metrics.csv")
    assert metrics.evaluation_units.sum() == cohort.frame.patient_id.nunique()


def test_after_all_folds_hook_and_lasso_stability(tmp_path):
    class LassoFake(FakeTrainer):
        def fit(self, data):
            predictor = super().fit(data)
            pd.DataFrame({"feature_name": ["a", "b"], "feature_group": ["clinical", "kinetic"],
                          "selected": [True, data.fold % 2 == 0], "lasso_importance": [1.0, 0.5 * data.fold],
                          "lasso_rank": [1, 2]}).to_csv(data.output_dir / "lasso_feature_selection.csv", index=False)
            return predictor

    trainers = {"image": FakeTrainer("image", "signal_image", with_hook=True),
                "tabular": LassoFake("tabular", "signal_tabular")}
    train_dir = run_training(make_cfg("train"), make_fake_cohort(40), trainers, checkpoint_root=tmp_path,
                             input_fingerprints=FPS)
    assert trainers["image"].hook_calls == [train_dir]
    stability = pd.read_csv(train_dir / "lasso_feature_stability.csv")
    assert list(stability.feature_name) == ["a", "b"]
    assert stability.selection_frequency.tolist() == pytest.approx([1.0, 1 / 3])


def test_validation_is_idempotent_and_needs_complete_training(tmp_path):
    train_dir, out, trainers, cohort = _run(tmp_path)
    mtime = (out / "validation_metrics.csv").stat().st_mtime_ns
    again = run_validation(make_cfg("validate"), cohort, trainers, checkpoint_root=tmp_path / "ck",
                           validation_root=tmp_path / "val_ck", input_fingerprints=FPS)
    assert again == out and (out / "validation_metrics.csv").stat().st_mtime_ns == mtime
    (train_dir / "COMPLETE").unlink()
    with pytest.raises(FileNotFoundError, match="not complete"):
        run_validation(make_cfg("validate", fusion_alpha=0.2), cohort, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "val_ck", input_fingerprints=FPS)
    with pytest.raises(FileNotFoundError, match="no training run"):
        run_validation(make_cfg("validate"), cohort, trainers, checkpoint_root=tmp_path / "ck",
                       validation_root=tmp_path / "val_ck", input_fingerprints={**FPS, "metadata_file": "sha256:other"})
