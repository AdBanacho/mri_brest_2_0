"""Training loop over outer folds: frozen split, per-branch fitting, atomic fold publication, resume.

Only :mod:`mribreast.contracts` Protocols are used; concrete branches are
injected by the composition root (``cli.py``) as ``trainers``.

Branch artifact contract (see docs/protocol.md):
* ``trainer.fit(FoldData)`` may write into ``FoldData.output_dir`` (a staging
  directory); afterwards the loop calls ``predictor.save(output_dir)``, which must
  therefore be idempotent. The directory is renamed on publication, so artifacts
  must reference each other by *relative* paths.
* Optional duck-typed hook ``trainer.after_all_folds(train_dir)`` runs once
  all folds are complete (e.g. cross-fold LASSO stability plots).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from mribreast import CONFIG_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION
from mribreast.artifacts import (ArtifactMismatch, atomic_dir, code_version, environment_versions, file_fingerprint,
                                 is_complete, mark_complete, write_json_atomic, write_text_atomic)
from mribreast.config import schema
from mribreast.contracts import LABEL, STUDY_ID, BranchTrainer, CohortTable, FoldData
from mribreast.protocol import splits
from mribreast.protocol.manifests import (FOLD_MANIFEST, SPLIT_MANIFEST, TRAIN_MANIFEST, cohort_csv_text,
                                          directory_fingerprints, prepared_frame, text_fingerprint, verify_fold,
                                          verify_train_manifest)

log = logging.getLogger(__name__)

LASSO_REPORT = "lasso_feature_selection.csv"


def check_trainers(cfg: Mapping[str, Any], trainers: Mapping[str, BranchTrainer]) -> list[str]:
    branches = list(cfg["branches"])
    missing = [b for b in branches if b not in trainers]
    if missing:
        raise ValueError(f"no trainer supplied for branch(es) {missing}")
    for name in branches:
        if getattr(trainers[name], "branch", name) != name:
            raise ValueError(f"trainer registered as {name!r} reports branch {trainers[name].branch!r}")
    return branches


def _csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def _resolve_plan(cfg: Mapping[str, Any], frame: pd.DataFrame, split_path: Path) -> tuple[pd.DataFrame, int, bool]:
    """Load + verify a frozen plan, or create and save a new one. Returns (plan, label_changes, reused)."""
    if split_path.is_file():
        plan = splits.load_split_plan(split_path)
        splits.validate_split_plan(plan, frame, check_labels=False)
        splits.check_plan_classes(plan, frame)
        if splits.fold_numbers(plan) != list(range(1, int(cfg["num_folds"]) + 1)):
            raise ArtifactMismatch(f"{split_path}: folds {splits.fold_numbers(plan)} != num_folds={cfg['num_folds']}")
        expected_roles = {"shared": {"fit", "calibration", "outer_eval"},
                          "separate": {"fit", "selection", "threshold", "outer_eval"},
                          "outer": {"fit", "outer_eval"}}[cfg["selection_split"]]
        if set(plan["role"]) != expected_roles:
            raise ArtifactMismatch(f"{split_path}: roles {sorted(set(plan['role']))} do not match "
                                   f"selection_split={cfg['selection_split']}")
        current = frame.set_index(STUDY_ID)[LABEL]
        changes = int((plan[LABEL].to_numpy() != current.loc[plan[STUDY_ID]].to_numpy()).sum())
        if changes:
            log.warning("frozen split %s: %d (fold, study) rows carry labels that differ from the current cohort",
                        split_path, changes)
        return plan, changes, True
    plan = splits.make_split_plan(frame, cfg)
    splits.save_split_plan(plan, split_path)
    return plan, 0, False


def aggregate_lasso_stability(train_dir: Path, fold_numbers: list[int]) -> Path | None:
    """Legacy cross-fold LASSO stability (``n_fold_cv_run.py`` lines 676-708) from per-fold reports."""
    reports = []
    for fold in fold_numbers:
        path = train_dir / f"fold_{fold}" / "tabular" / LASSO_REPORT
        if path.is_file():
            report = pd.read_csv(path)
            if "fold" not in report.columns:
                report.insert(0, "fold", fold)
            reports.append(report)
    needed = {"feature_name", "feature_group", "selected", "lasso_importance", "lasso_rank"}
    if not reports or not all(needed <= set(r.columns) for r in reports):
        return None
    combined = pd.concat(reports, ignore_index=True)
    write_text_atomic(train_dir / "lasso_feature_selection_by_fold.csv", _csv_text(combined))
    stability = (
        combined.groupby(["feature_name", "feature_group"], as_index=False)
        .agg(feature_occurrence_count=("fold", "nunique"), selection_count=("selected", "sum"),
             mean_lasso_importance=("lasso_importance", "mean"), max_lasso_importance=("lasso_importance", "max"),
             mean_lasso_rank=("lasso_rank", "mean"))
        .assign(selection_frequency=lambda r: r["selection_count"] / len(reports))
        .sort_values(["selection_frequency", "mean_lasso_importance", "feature_name"],
                     ascending=[False, False, True], kind="stable")
    )
    out = train_dir / "lasso_feature_stability.csv"
    write_text_atomic(out, _csv_text(stability))
    return out


def run_training(cfg: Mapping[str, Any], cohort: CohortTable, trainers: Mapping[str, BranchTrainer], *,
                 checkpoint_root: Path, input_fingerprints: Mapping[str, str],
                 extra_audits: Mapping[str, pd.DataFrame] | None = None) -> Path:
    """Fit every branch on every outer fold; return ``<checkpoint_root>/<train_id>``.

    Resumable: complete folds whose ``fold_manifest.json`` matches the current
    ``train_id`` + ``split_hash`` are skipped; an interrupted fold is rebuilt.
    An existing train directory with a different identity raises
    :class:`ArtifactMismatch`.
    """
    num_classes = schema.num_classes(cfg)
    branches = check_trainers(cfg, trainers)
    tid = schema.train_id(cfg, input_fingerprints)
    train_dir = Path(checkpoint_root) / tid
    train_dir.mkdir(parents=True, exist_ok=True)
    frame = prepared_frame(cohort, num_classes)
    if cohort.label.scheme != cfg["label_scheme"]:
        raise ValueError(f"cohort label scheme {cohort.label.scheme!r} != config {cfg['label_scheme']!r}")
    cohort_text = cohort_csv_text(frame)
    cohort_fp = text_fingerprint(cohort_text)
    label_def = cohort.label.to_json()
    manifest_path = train_dir / TRAIN_MANIFEST
    split_path = train_dir / SPLIT_MANIFEST

    if manifest_path.is_file():
        if not split_path.is_file():
            raise ArtifactMismatch(f"{train_dir}: manifest present but {SPLIT_MANIFEST} missing")
        s_hash = file_fingerprint(split_path)
        verify_train_manifest(manifest_path, train_id=tid, label_definition=label_def,
                              input_fingerprints=input_fingerprints, cohort_fingerprint=cohort_fp, split_hash=s_hash)
        plan, _, _ = _resolve_plan(cfg, frame, split_path)
        log.info("resuming %s", train_dir)
    else:
        plan, label_changes, reused = _resolve_plan(cfg, frame, split_path)
        s_hash = file_fingerprint(split_path)
        write_text_atomic(train_dir / "cohort.csv", cohort_text)
        write_text_atomic(train_dir / "cohort_audit.csv", _csv_text(cohort.audit) if len(cohort.audit.columns) else "\n")
        for name, table in (extra_audits or {}).items():
            write_text_atomic(train_dir / f"{name}.csv", _csv_text(table))
        write_json_atomic(manifest_path, {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "train_id": tid,
            "train_identity": schema.train_identity(cfg, input_fingerprints),
            "code_version": code_version(),
            "environment": environment_versions(),
            "resolved_config": schema.to_jsonable(cfg),
            "label_definition": label_def,
            "input_fingerprints": dict(input_fingerprints),
            "cohort_fingerprint": cohort_fp,
            "cohort_notes": list(cohort.notes),
            "continuous_columns": list(cohort.continuous_columns),
            "categorical_columns": list(cohort.categorical_columns),
            "split_hash": s_hash,
            "split_reused": reused,
            "split_label_changes": label_changes,
            "branches": branches,
            "num_classes": num_classes,
            "num_folds": int(cfg["num_folds"]),
            "selection_split": cfg["selection_split"],
            "profile": cfg["profile"],
            "studies": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

    folds = splits.fold_numbers(plan)
    if is_complete(train_dir):
        for fold in folds:
            verify_fold(train_dir / f"fold_{fold}", train_id=tid, split_hash=s_hash, fold=fold, check_files=False)
        log.info("train %s already complete", tid)
        return train_dir

    mode = cfg["selection_split"]
    for fold in folds:
        fold_dir = train_dir / f"fold_{fold}"
        if is_complete(fold_dir):
            verify_fold(fold_dir, train_id=tid, split_hash=s_hash, fold=fold, check_files=False)
            log.info("fold %d complete - skipped", fold)
            continue
        fit = splits.role_frame(plan, frame, fold, ("fit",))
        selection = splits.role_frame(plan, frame, fold, splits.selection_roles(mode))
        meta = splits.fold_meta(plan, fold)
        with atomic_dir(fold_dir) as staging:
            artifacts: dict[str, dict[str, str]] = {}
            for branch in branches:
                out = staging / branch
                out.mkdir(parents=True, exist_ok=True)
                data = FoldData(fold=fold, fit=fit.copy(), selection=selection.copy(), seed=int(cfg["seed"]),
                                output_dir=out)
                predictor = trainers[branch].fit(data)
                predictor.save(out)
                artifacts[branch] = directory_fingerprints(out)
            write_json_atomic(staging / FOLD_MANIFEST, {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "config_schema_version": CONFIG_SCHEMA_VERSION,
                "train_id": tid,
                "fold": fold,
                "split_hash": s_hash,
                "selection_split": mode,
                "selection_roles": list(splits.selection_roles(mode)),
                "roles": {k: v for k, v in meta.items() if k.endswith("_rows")},
                "requested_inner_folds": meta["requested_inner_folds"],
                "effective_inner_folds": meta["effective_inner_folds"],
                "outer_seed": meta["outer_seed"],
                "inner_seed": meta["inner_seed"],
                "fit_studies": int(len(fit)),
                "selection_studies": int(len(selection)),
                "branch_artifacts": artifacts,
            })
            mark_complete(staging, {"train_id": tid, "fold": fold})
        log.info("fold %d published", fold)

    for branch in branches:
        hook = getattr(trainers[branch], "after_all_folds", None)
        if callable(hook):
            hook(train_dir)
    if "tabular" in branches and not (train_dir / "lasso_feature_stability.csv").exists():
        aggregate_lasso_stability(train_dir, folds)
    mark_complete(train_dir, {"train_id": tid, "split_hash": s_hash, "folds": folds})
    return train_dir
