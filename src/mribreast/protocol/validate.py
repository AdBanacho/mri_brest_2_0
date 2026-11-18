"""Validation: load saved fold artifacts, verify identity, tune thresholds on the threshold split,
evaluate once on the outer fold, publish ``<validation_root>/<train_id>/<eval_id>/`` atomically.

Data usage per ``selection_split`` (recorded in ``evaluation_manifest.json``):

=========  =======================  ======================  ===========
mode       checkpoint selection     threshold tuning        evaluation
=========  =======================  ======================  ===========
shared     calibration              calibration             outer_eval
separate   selection                threshold               outer_eval
outer      outer_eval (historical)  outer_eval              outer_eval
=========  =======================  ======================  ===========
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mribreast import CONFIG_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION
from mribreast.artifacts import (ArtifactMismatch, atomic_dir, code_version, environment_versions, file_fingerprint,
                                 is_complete, mark_complete, read_json, write_json_atomic, write_text_atomic)
from mribreast.config import schema
from mribreast.contracts import (LABEL, PATIENT_ID, STUDY_ID, BranchTrainer, CohortTable, predictions_frame)
from mribreast.protocol import splits
from mribreast.protocol.fusion import branch_frame, fuse_aligned
from mribreast.protocol.manifests import (EVALUATION_MANIFEST, SPLIT_MANIFEST, TRAIN_MANIFEST, check_schema,
                                          cohort_csv_text, prepared_frame, text_fingerprint, verify_fold,
                                          verify_train_manifest)
from mribreast.protocol.thresholds import decide, tune_branch_thresholds
from mribreast.reporting import plots, tables
from mribreast.reporting.report import validation_report

log = logging.getLogger(__name__)


def _units(table: pd.DataFrame, probs: Mapping[str, np.ndarray], unit: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Aggregate study rows to the report unit. ``patient``: mean probability per patient (first-seen order)."""
    if unit == "study":
        return table.reset_index(drop=True), dict(probs)
    if unit != "patient":
        raise ValueError(f"unknown report_unit {unit!r}")
    codes, uniques = pd.factorize(table[PATIENT_ID].astype(str), sort=False)
    out = pd.DataFrame({PATIENT_ID: uniques.astype(str)})
    out[STUDY_ID] = [ "|".join(table[STUDY_ID].astype(str).to_numpy()[codes == i]) for i in range(len(uniques))]
    labels = table[LABEL].astype(int).to_numpy()
    out[LABEL] = [int(labels[codes == i][0]) for i in range(len(uniques))]
    aggregated = {}
    for branch, p in probs.items():
        sums = np.zeros((len(uniques), p.shape[1]))
        np.add.at(sums, codes, p)
        aggregated[branch] = sums / np.bincount(codes)[:, None]
    return out, aggregated


def _predict(trainers: Mapping[str, BranchTrainer], predictors: Mapping[str, Any], frame: pd.DataFrame,
             num_classes: int, alpha: float) -> dict[str, np.ndarray]:
    ids = frame[STUDY_ID].astype(str).tolist()
    per_branch = {b: branch_frame(ids, predictors[b].predict_proba(frame), num_classes, b) for b in predictors}
    return fuse_aligned(ids, per_branch, num_classes, alpha)


def _csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def run_validation(cfg: Mapping[str, Any], cohort: CohortTable, trainers: Mapping[str, BranchTrainer], *,
                   checkpoint_root: Path, validation_root: Path, input_fingerprints: Mapping[str, str]) -> Path:
    """Evaluate a complete training run; return the published evaluation directory."""
    num_classes = schema.num_classes(cfg)
    positive = int(cfg["positive_class"])
    tid = schema.train_id(cfg, input_fingerprints)
    eid = schema.eval_id(cfg)
    run_id = f"{tid}/{eid}"
    train_dir = Path(checkpoint_root) / tid
    out_dir = Path(validation_root) / tid / eid
    branches = list(cfg["branches"])
    missing = [b for b in branches if b not in trainers]
    if missing:
        raise ValueError(f"no trainer supplied for branch(es) {missing}")

    # ---------------------------------------------------------------- verify training artifacts
    manifest_path = train_dir / TRAIN_MANIFEST
    split_path = train_dir / SPLIT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no training run for this configuration: {manifest_path} (train_id={tid})")
    if not is_complete(train_dir):
        raise FileNotFoundError(f"training run {train_dir} is not complete; rerun train to resume")
    frame = prepared_frame(cohort, num_classes)
    cohort_fp = text_fingerprint(cohort_csv_text(frame))
    s_hash = file_fingerprint(split_path)
    train_manifest = verify_train_manifest(manifest_path, train_id=tid, label_definition=cohort.label.to_json(),
                                           input_fingerprints=input_fingerprints, cohort_fingerprint=cohort_fp,
                                           split_hash=s_hash)
    plan = splits.load_split_plan(split_path)
    splits.validate_split_plan(plan, frame, check_labels=False)
    splits.check_plan_classes(plan, frame)
    mode = cfg["selection_split"]
    folds = splits.fold_numbers(plan)

    if is_complete(out_dir):
        existing = read_json(out_dir / EVALUATION_MANIFEST)
        check_schema(existing, out_dir / EVALUATION_MANIFEST)
        if (existing.get("train_id"), existing.get("eval_id"), existing.get("split_hash")) != (tid, eid, s_hash):
            raise ArtifactMismatch(f"{out_dir}: existing evaluation belongs to another run")
        log.info("evaluation %s already complete", run_id)
        return out_dir

    strategy = cfg["threshold_strategy"] if num_classes == 2 else "argmax"
    alpha = float(cfg["fusion_alpha"])
    unit = cfg["report_unit"]
    primary = schema.primary_branch(cfg)
    eval_branches = tables.ordered_branches(branches + (["fusion"] if set(branches) == {"image", "tabular"} else []))
    thr_roles = splits.threshold_roles(mode)
    class_names = list(cohort.label.class_names)

    fold_rows: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []
    calibration_preds: list[pd.DataFrame] = []
    validation_preds: list[pd.DataFrame] = []
    thresholds_json: dict[str, dict[str, float | None]] = {}
    pooled: dict[str, dict[str, list[np.ndarray]]] = {b: {"labels": [], "probs": [], "decisions": []}
                                                      for b in eval_branches}
    fold_curves: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {b: [] for b in eval_branches}
    confusion: dict[str, dict[str, Any]] = {b: {"folds": {}} for b in eval_branches}

    for fold in folds:
        fold_dir = train_dir / f"fold_{fold}"
        if not is_complete(fold_dir):
            raise FileNotFoundError(f"{fold_dir} is not complete")
        verify_fold(fold_dir, train_id=tid, split_hash=s_hash, fold=fold, check_files=True)
        predictors = {b: trainers[b].load(fold_dir / b, fold) for b in branches}
        thr_frame = splits.role_frame(plan, frame, fold, thr_roles)
        eval_frame = splits.role_frame(plan, frame, fold, ("outer_eval",))
        fit_rows = len(splits.role_rows(plan, fold, ("fit",)))
        heldout_rows = len(splits.role_rows(plan, fold, splits.inner_heldout_roles()))
        meta = splits.fold_meta(plan, fold)

        cal_units, cal_probs = _units(thr_frame, _predict(trainers, predictors, thr_frame, num_classes, alpha), unit)
        ev_units, ev_probs = _units(eval_frame, _predict(trainers, predictors, eval_frame, num_classes, alpha), unit)
        cal_labels = cal_units[LABEL].to_numpy(dtype=np.int64)
        ev_labels = ev_units[LABEL].to_numpy(dtype=np.int64)

        thresholds, table = tune_branch_thresholds(strategy if num_classes == 2 else "fixed_0.5", cal_labels,
                                                   {b: cal_probs[b] for b in eval_branches}, num_classes, positive)
        table.insert(0, "fold", fold)
        table["calibration_rows"] = len(cal_units)
        table["threshold_split"] = "+".join(thr_roles)
        table["report_unit"] = unit
        threshold_tables.append(table)
        thresholds_json[str(fold)] = {b: thresholds[b] for b in eval_branches}

        image_pred = predictors.get("image")
        checkpoint = getattr(image_pred, "checkpoint_path", None) if image_pred is not None else None
        row: dict[str, Any] = {
            "fold": fold,
            "checkpoint_path": str(checkpoint) if checkpoint else str(fold_dir / branches[0]),
            "inner_fit_rows": fit_rows,
            "inner_calibration_rows": heldout_rows,
            "effective_inner_calibration_folds": meta["effective_inner_folds"],
            "outer_validation_rows": len(eval_frame),
        }
        for b in eval_branches:
            cal_dec = decide(cal_probs[b], thresholds[b], positive)
            ev_dec = decide(ev_probs[b], thresholds[b], positive)
            metrics = tables.branch_metrics(ev_labels, ev_probs[b], thresholds[b], num_classes, positive)
            row.update(tables.prefixed(b, metrics, num_classes))
            calibration_preds.append(predictions_frame(fold, cal_units, cal_probs[b], b, "+".join(thr_roles))
                                     .assign(decision=cal_dec))
            validation_preds.append(predictions_frame(fold, ev_units, ev_probs[b], b, "outer_eval")
                                    .assign(decision=ev_dec))
            pooled[b]["labels"].append(ev_labels)
            pooled[b]["probs"].append(ev_probs[b])
            pooled[b]["decisions"].append(ev_dec)
            fold_curves[b].append((f"fold {fold}", ev_labels, ev_probs[b]))
            confusion[b]["folds"][str(fold)] = tables.confusion_payload(ev_labels, ev_dec, num_classes)
        row.update({
            "requested_inner_folds": meta["requested_inner_folds"],
            "selection_rows": len(splits.role_rows(plan, fold, splits.selection_roles(mode))),
            "threshold_rows": len(thr_frame),
            "evaluation_units": len(ev_units),
            "report_unit": unit,
            "selection_split": mode,
            "run_id": run_id,
        })
        fold_rows.append(row)
        log.info("fold %d evaluated", fold)

    # ---------------------------------------------------------------- pooled + tables
    metrics_df = pd.DataFrame(fold_rows)
    pooled_metrics: dict[str, dict[str, float]] = {}
    pooled_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for b in eval_branches:
        y = np.concatenate(pooled[b]["labels"])
        p = np.concatenate(pooled[b]["probs"])
        d = np.concatenate(pooled[b]["decisions"])
        pooled_arrays[b] = (y, p, d)
        pooled_metrics[b] = tables.decision_metrics(y, d, p, num_classes, positive)
        confusion[b]["pooled"] = tables.confusion_payload(y, d, num_classes)
    summary = tables.summary_table(metrics_df, pooled_metrics)
    comparison = tables.branch_comparison(metrics_df, pooled_metrics, eval_branches, primary)

    selection_uses_outer = mode == "outer"
    eval_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "train_id": tid,
        "eval_id": eid,
        "run_id": run_id,
        "eval_identity": schema.eval_identity(cfg),
        "code_version": code_version(),
        "environment": environment_versions(),
        "train_code_version": train_manifest.get("code_version"),
        "split_hash": s_hash,
        "cohort_fingerprint": cohort_fp,
        "label_definition": cohort.label.to_json(),
        "num_classes": num_classes,
        "num_folds": len(folds),
        "branches_trained": branches,
        "branches_evaluated": eval_branches,
        "primary_branch": primary,
        "fusion_alpha": alpha,
        "threshold_strategy": strategy,
        "decision_rule": "p[:, positive_class] >= threshold" if num_classes == 2 else "argmax",
        "report_unit": unit,
        "selection_split": mode,
        "roles": {"fit": ["fit"], "selection": list(splits.selection_roles(mode)), "threshold": list(thr_roles),
                  "evaluation": ["outer_eval"]},
        "selection_uses_outer_eval": selection_uses_outer,
        "threshold_uses_outer_eval": "outer_eval" in thr_roles,
        "pooled_metrics": pooled_metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    validation_config = {**schema.to_jsonable(cfg), "train_id": tid, "eval_id": eid, "run_id": run_id,
                         "primary_branch_resolved": primary, "num_classes": num_classes,
                         "threshold_strategy_effective": strategy, "experiment_name": tid}

    with atomic_dir(out_dir) as staging:
        write_json_atomic(staging / EVALUATION_MANIFEST, eval_manifest)
        write_json_atomic(staging / "validation_config.json", validation_config)
        write_text_atomic(staging / "validation_metrics.csv", _csv(metrics_df))
        write_text_atomic(staging / "validation_summary.csv", _csv(summary))
        write_json_atomic(staging / "decision_thresholds.json", {
            "threshold_strategy": strategy, "positive_class": positive if num_classes == 2 else None,
            "threshold_split": list(thr_roles), "report_unit": unit, "folds": thresholds_json})
        write_text_atomic(staging / "threshold_calibration_metrics.csv",
                          _csv(pd.concat(threshold_tables, ignore_index=True)))
        write_text_atomic(staging / "predictions_calibration.csv", _csv(pd.concat(calibration_preds, ignore_index=True)))
        write_text_atomic(staging / "predictions_validation.csv", _csv(pd.concat(validation_preds, ignore_index=True)))
        write_text_atomic(staging / "branch_comparison.csv", _csv(comparison))
        write_json_atomic(staging / "confusion_matrices.json", {"class_names": class_names, "labels": list(range(num_classes)),
                                                               "rows": "true", "columns": "predicted",
                                                               "branches": confusion})
        for b in eval_branches:
            y, p, d = pooled_arrays[b]
            plots.save_roc(staging / f"roc_{b}.png", f"{b}: pooled outer-fold ROC", fold_curves[b], y, p,
                           num_classes, positive)
            plots.save_confusion(staging / f"confusion_{b}.png",
                                 f"{b}: pooled confusion (fold-specific decisions)",
                                 np.asarray(confusion[b]["pooled"]), class_names)
        write_text_atomic(staging / "validation_report.md", validation_report(
            manifest=eval_manifest, cfg=cfg, metrics=metrics_df, summary=summary, comparison=comparison,
            thresholds=thresholds_json, class_names=class_names))
        mark_complete(staging, {"train_id": tid, "eval_id": eid, "folds": folds})
    return out_dir
