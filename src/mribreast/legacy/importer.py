"""Import historical legacy validation results into a normalized table.

Sources (any combination; nothing is written into them):

* a legacy **summary directory** (``validation_charts/summary`` or the tracked
  ``mriBreastDuke/validationSummary``): ``all_validation_metrics.csv``,
  ``configuration_summary.csv``, ``ranked_configurations.csv``,
  ``best_configuration.json``;
* a legacy **validation root**: ``<experiment_name>/fusion_alpha_<a>/`` with
  ``validation_metrics.csv``, ``validation_config.json``,
  ``validation_summary.csv``, ``threshold_calibration_metrics.csv`` and
  ``fold_<k>/decision_thresholds.json``;
* optionally a legacy **checkpoint root** (``<experiment_name>/fold_<k>/checkpoints``)
  to verify that split metadata (``inner_calibration_split.csv``) and the
  tabular artifacts exist.

Normalization (docs/migration.md):

* metric columns use the new pattern ``<branch>_<metric>`` with branch in
  ``image, tabular, fusion``; legacy ``xgboost_*``/``mlp_*`` columns become
  ``tabular_*`` (the duplicates written by the legacy summarizer are checked
  for equality and dropped);
* configuration values use the new field names (fields.tsv);
* every row gets ``run_id = <train_id>/<eval_id>`` in a dedicated legacy
  namespace: ``train_id = "lg-" + sha256(experiment_name)[:12]`` and
  ``eval_id = "ev-a<alpha>-legacy"`` (never collides with new ``tr-`` ids),
  plus ``legacy_run_id = <experiment_name>/fusion_alpha_<alpha>``;
* flags ``requires_retraining`` / ``retraining_reasons`` /
  ``split_metadata`` / ``comparable_to_new`` mark runs that cannot be carried
  over as fresh results.

Imported numbers are **historical context only**; they are never re-emitted
as new results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.legacy._fs import display, fs
from mribreast.legacy.checkpoints import FORMAT_AUC, inventory_experiment, parse_checkpoint_name
from mribreast.legacy.exporter import CLASSIFICATION_METRICS, discover_runs
from mribreast.legacy.names import parse_experiment_name, split_run_id

BRANCHES = ("image", "tabular", "fusion")
METRICS = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc_roc", "decision_threshold")
TABULAR_MODELS = ("xgboost", "mlp")
#: legacy column -> normalized column
COUNT_RENAMES = {"inner_fit_rows": "fit_rows", "inner_calibration_rows": "calibration_rows",
                 "effective_inner_calibration_folds": "effective_inner_folds",
                 "outer_validation_rows": "outer_eval_rows"}
#: legacy validate-CLI / config keys that map to a different new key
CONFIG_RENAMES = {"num_workers": "validate_num_workers"}
CONFIG_KEYS = ("mri_model", "subtraction_mode", "feature_groups", "feature_model", "feature_selector", "batch_size",
               "positive_boost", "sensitivity_lambda", "lr", "threshold_calibration_folds", "num_folds", "seed",
               "target_size", "use_annotation_boxes", "include_sensitive", "allow_missing_imaging_features",
               "imaging_patient_id_column", "lasso_cv_folds", "lasso_cs", "lasso_max_iter", "lasso_tolerance",
               "lasso_min_features", "experiment_id", "label_scheme", "branches", "selection_split", "profile")
#: keys the legacy validate CLI accepted at SHA 127a2604 but which older stored configs lack
CURRENT_VALIDATE_ONLY_KEYS = ("seed", "target_size", "experiment_id", "use_annotation_boxes", "clinical_features_file",
                              "metadata_file", "image_root", "mri_extra_json")
IDENTITY_COLUMNS = ("run_id", "train_id", "eval_id", "legacy_run_id", "legacy_experiment_name", "fusion_alpha", "fold")
FLAG_COLUMNS = ("workflow", "protocol", "tabular_model", "clinical_safe_v2", "split_metadata", "checkpoint_format",
                "requires_retraining", "retraining_reasons", "comparable_to_new", "notes", "source")

RETRAINING_REASONS = {
    "image_only_outer_selection": "historical image-only run: checkpoint chosen on the outer fold that is also "
                                  "reported (no inner selection set), StratifiedKFold without patient grouping, "
                                  "argmax decisions",
    "no_inner_calibration": "run predates the inner fit/calibration split (no auc-cal<k> in the name)",
    "pre_clinical_safe_v2_predictors": "clinical branch trained on the pre-'clinical-safe-v2' predictor list "
                                       "(receptor/grade/stage columns); not reproducible with the current policy",
    "split_metadata_missing": "fold_k/checkpoints/inner_calibration_split.csv is missing; the fit/calibration "
                              "membership cannot be verified or converted to split_manifest.csv",
    "checkpoint_format_balanced_accuracy": "best checkpoint selected by balanced accuracy (historical era)",
    "checkpoint_format_sensitivity": "best checkpoint selected by sensitivity (historical era)",
    "checkpoint_format_unknown": "best checkpoint filename not in any legacy format",
    "unknown_experiment_name": "experiment name does not match any legacy naming scheme",
}


def legacy_train_id(experiment_name: str) -> str:
    return "lg-" + hashlib.sha256(experiment_name.encode("utf-8")).hexdigest()[:12]


def legacy_eval_id(alpha: float | None) -> str:
    return f"ev-a{alpha:g}-legacy" if alpha is not None and not _isnan(alpha) else "ev-legacy"


def _isnan(value: Any) -> bool:
    try:
        return bool(isinstance(value, float) and math.isnan(value))
    except TypeError:
        return False


def _bool(value: Any) -> bool | None:
    if value is None or _isnan(value):
        return None
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


# --------------------------------------------------------------------------- metric normalization
def normalize_metric_columns(frame: pd.DataFrame, feature_model: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Rename ``<feature_model>_*`` -> ``tabular_*`` (also ``_mean``/``_std`` suffixed columns).

    When both exist (legacy summarizer duplicates) they must be equal (NaN-aware);
    otherwise a note is returned and the model-named values win.
    """
    notes: list[str] = []
    out = frame.copy()
    models = [feature_model] if feature_model else [m for m in TABULAR_MODELS
                                                    if any(c.startswith(f"{m}_") for c in out.columns)]
    for model in models:
        for column in list(out.columns):
            if not column.startswith(f"{model}_"):
                continue
            rest = column[len(model) + 1:]
            base = re.sub(r"_(mean|std)$", "", rest)
            if base not in CLASSIFICATION_METRICS:
                continue
            target = f"tabular_{rest}"
            if target in out.columns:
                a = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(out[target], errors="coerce").to_numpy(dtype=float)
                if not np.allclose(a, b, equal_nan=True, rtol=0, atol=0):
                    notes.append(f"tabular_duplicate_mismatch:{rest}")
            out[target] = out[column]
            out = out.drop(columns=[column])
    return out.rename(columns=COUNT_RENAMES), notes


def _config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        new_key = CONFIG_RENAMES.get(key, key)
        if isinstance(value, list):
            value = ",".join(map(str, value)) if key == "feature_groups" else " ".join(map(str, value))
        out[new_key] = value
    return out


# --------------------------------------------------------------------------- assessment
def assess(experiment_name: str, config: Mapping[str, Any], *, checkpoint_paths: Sequence[str] = (),
           has_calibration_rows: bool | None = None, inventory_reasons: Sequence[str] | None = None) -> dict[str, Any]:
    """Classify one legacy run and list the reasons it must be retrained (if any)."""
    parsed = parse_experiment_name(experiment_name)
    reasons: list[str] = []
    notes: list[str] = []
    workflow = parsed["workflow"]
    if workflow == "historical_image":
        reasons.append("image_only_outer_selection")
        protocol = "historical_image_outer_selection"
    elif workflow == "fusion":
        protocol = "legacy_fusion_inner_calibration"
        if parsed.get("clinical_safe_v2") is False:
            reasons.append("pre_clinical_safe_v2_predictors")
    else:
        protocol = "unknown"
        if experiment_name.startswith("ImagingFeaturesFusion_") and "_auc-cal" not in experiment_name:
            reasons.append("no_inner_calibration")
        else:
            reasons.append("unknown_experiment_name")
    formats = sorted({parse_checkpoint_name(p).format for p in checkpoint_paths if isinstance(p, str) and p})
    for fmt in formats:
        if fmt != FORMAT_AUC:
            reasons.append(f"checkpoint_format_{fmt}")
    if inventory_reasons is not None:
        split_metadata = "missing" if any(r.startswith("split_") for r in inventory_reasons) else "present"
        for reason in inventory_reasons:
            if reason.startswith("split_metadata_missing") and "split_metadata_missing" not in reasons:
                reasons.append("split_metadata_missing")
            elif not reason.startswith(("split_", "checkpoint_format_")):
                notes.append(f"checkpoint_root:{reason}")
    elif workflow == "fusion" and has_calibration_rows:
        # The legacy validator refuses to run unless inner_calibration_split.csv matches the
        # reconstructed split, so a stored fusion result implies the manifest existed then.
        split_metadata = "verified_by_legacy_validation"
    elif workflow == "historical_image":
        split_metadata = "none_written"
    else:
        split_metadata = "unknown"
    missing_cfg = [k for k in CURRENT_VALIDATE_ONLY_KEYS if config and k not in config]
    if missing_cfg:
        notes.append("config_predates_keys:" + ",".join(missing_cfg) + " (defaults assumed: seed=42, "
                     "target_size=256 256 64, no annotation boxes)")
    if parsed.get("experiment_id", "") == "" and workflow == "fusion":
        notes.append("no_experiment_id: run predates the Slurm controller identity suffix")
    return {
        "workflow": workflow, "protocol": protocol, "clinical_safe_v2": parsed.get("clinical_safe_v2"),
        "split_metadata": split_metadata, "checkpoint_format": ",".join(formats) or "",
        "requires_retraining": bool(reasons), "retraining_reasons": ";".join(dict.fromkeys(reasons)),
        "comparable_to_new": not reasons, "notes": ";".join(notes), "parsed": parsed,
    }


def _identity(experiment_name: str, alpha: float | None) -> dict[str, Any]:
    train_id, eval_id = legacy_train_id(experiment_name), legacy_eval_id(alpha)
    legacy_run_id = f"{experiment_name}/fusion_alpha_{alpha:g}" if alpha is not None else experiment_name
    return {"run_id": f"{train_id}/{eval_id}", "train_id": train_id, "eval_id": eval_id,
            "legacy_run_id": legacy_run_id, "legacy_experiment_name": experiment_name, "fusion_alpha": alpha}


def _finish(frame: pd.DataFrame, leading: Sequence[str]) -> pd.DataFrame:
    metric_cols = [f"{b}_{m}{s}" for b in BRANCHES for m in METRICS for s in ("", "_mean", "_std")
                   if f"{b}_{m}{s}" in frame.columns]
    first = [c for c in (*leading, *FLAG_COLUMNS) if c in frame.columns]
    config_cols = [c for c in CONFIG_KEYS if c in frame.columns and c not in first]
    rest = [c for c in frame.columns if c not in first and c not in metric_cols and c not in config_cols]
    return frame[first + config_cols + metric_cols + rest]


# --------------------------------------------------------------------------- readers
def _inventory_reasons(checkpoint_root: str | Path | None, experiment_name: str, config: Mapping[str, Any]) -> list[str] | None:
    if checkpoint_root is None:
        return None
    num_folds = int(config.get("num_folds", 5) or 5)
    feature_model = config.get("feature_model") or None
    selector = str(config.get("feature_selector", "none")) == "lasso"
    reasons: list[str] = []
    for inv in inventory_experiment(checkpoint_root, experiment_name, num_folds, expect_tabular=feature_model,
                                    expect_selector=selector):
        reasons += [f"{r}@fold{inv.fold}" if not r.startswith("split_metadata_missing") else r for r in inv.reasons]
    return reasons


def import_fold_rows(frame: pd.DataFrame, *, config_lookup: Mapping[str, Mapping[str, Any]] | None = None,
                     checkpoint_root: str | Path | None = None, source: str = "summary") -> pd.DataFrame:
    """Normalize legacy per-fold rows (``all_validation_metrics.csv`` or concatenated run files).

    ``frame`` must contain ``run_id`` (legacy) and ``fold``.  Configuration
    values come from the row itself (summary files carry them) or from
    ``config_lookup[run_id]``.
    """
    rows = []
    for legacy_run_id, part in frame.groupby("run_id", sort=False):
        experiment_name, alpha = split_run_id(str(legacy_run_id))
        config = dict((config_lookup or {}).get(str(legacy_run_id), {}))
        first = part.iloc[0].to_dict()
        if not config:
            config = {k: v for k, v in first.items() if not str(k).startswith(tuple(f"{b}_" for b in (*BRANCHES, *TABULAR_MODELS)))
                      and k not in ("run_id", "fold", "checkpoint_path", *COUNT_RENAMES)}
        if alpha is None and "fusion_alpha" in config:
            alpha = float(config["fusion_alpha"])
        feature_model = str(config.get("feature_model") or "") or None
        normalized, notes = normalize_metric_columns(part.drop(columns=["run_id"]), feature_model)
        verdict = assess(experiment_name, config,
                         checkpoint_paths=[p for p in part.get("checkpoint_path", pd.Series(dtype=str)).tolist()
                                           if isinstance(p, str)],
                         has_calibration_rows="inner_calibration_rows" in part.columns,
                         inventory_reasons=_inventory_reasons(checkpoint_root, experiment_name, config))
        cfg_values = _config_from_mapping(config)
        parsed = verdict.pop("parsed")
        for key in ("label_scheme", "branches"):
            cfg_values.setdefault(key, parsed.get(key))
        if parsed.get("workflow") == "fusion":
            cfg_values.setdefault("selection_split", "shared")
            cfg_values.setdefault("profile", "duke_legacy_3d")
        elif parsed.get("workflow") == "historical_image":
            cfg_values.setdefault("selection_split", "outer")
            cfg_values.setdefault("profile", "duke_legacy_3d")
        normalized = normalized.drop(columns=[c for c in cfg_values if c in normalized.columns])
        ident = _identity(experiment_name, alpha)
        extra_notes = ";".join([verdict["notes"], *notes]).strip(";")
        for record in normalized.to_dict("records"):
            if "checkpoint_path" in record and isinstance(record["checkpoint_path"], str):
                parsed_ckpt = parse_checkpoint_name(record["checkpoint_path"])
                record["checkpoint_epoch"] = parsed_ckpt.epoch
                record["selection_auc_roc"] = parsed_ckpt.auc_roc
            rows.append({**ident, **record, **{k: v for k, v in verdict.items() if k != "notes"},
                         "notes": extra_notes, "tabular_model": feature_model or "", "source": source,
                         **{k: v for k, v in cfg_values.items() if k in CONFIG_KEYS}})
    if not rows:
        return pd.DataFrame(columns=[*IDENTITY_COLUMNS, *FLAG_COLUMNS])
    out = pd.DataFrame(rows)
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce").astype("Int64")
    return _finish(out, IDENTITY_COLUMNS)


def import_summary_dir(summary_dir: str | Path, *, checkpoint_root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Read a legacy summary directory. Returns ``{"folds", "configurations", "best"}`` frames."""
    summary_dir = Path(summary_dir)
    out: dict[str, pd.DataFrame] = {}
    all_path = summary_dir / "all_validation_metrics.csv"
    if all_path.is_file():
        folds = pd.read_csv(all_path)
        out["folds"] = import_fold_rows(folds, checkpoint_root=checkpoint_root, source=f"summary:{all_path.name}")
    config_path = summary_dir / "configuration_summary.csv"
    ranked_path = summary_dir / "ranked_configurations.csv"
    if config_path.is_file():
        configs = pd.read_csv(config_path)
        if ranked_path.is_file():
            ranked = pd.read_csv(ranked_path)[["run_id", "rank"]]
            configs = configs.merge(ranked, on="run_id", how="left", validate="one_to_one")
        else:
            configs["rank"] = pd.NA
        out["configurations"] = import_configuration_rows(configs, checkpoint_root=checkpoint_root,
                                                          source=f"summary:{config_path.name}")
    best_path = summary_dir / "best_configuration.json"
    if best_path.is_file():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        out["best"] = import_configuration_rows(pd.DataFrame([best]), checkpoint_root=checkpoint_root,
                                                source=f"summary:{best_path.name}")
    if not out:
        raise FileNotFoundError(f"no legacy summary files in {summary_dir}")
    return out


def import_configuration_rows(configs: pd.DataFrame, *, checkpoint_root: str | Path | None = None,
                              source: str = "summary") -> pd.DataFrame:
    """Normalize ``configuration_summary.csv`` / ``ranked_configurations.csv`` / ``best_configuration.json`` rows."""
    rows = []
    for record in configs.to_dict("records"):
        legacy_run_id = str(record.pop("run_id"))
        experiment_name, alpha = split_run_id(legacy_run_id)
        if alpha is None and record.get("fusion_alpha") is not None:
            alpha = float(record["fusion_alpha"])
        feature_model = str(record.get("feature_model") or "") or None
        normalized, notes = normalize_metric_columns(pd.DataFrame([record]), feature_model)
        record = normalized.iloc[0].to_dict()
        config = {k: v for k, v in record.items() if k in CONFIG_KEYS}
        verdict = assess(experiment_name, config, has_calibration_rows="calibration_rows_mean" in record
                         or "inner_calibration_rows_mean" in record,
                         inventory_reasons=_inventory_reasons(checkpoint_root, experiment_name,
                                                              {**config, "num_folds": record.get("expected_folds", 5)}))
        verdict.pop("parsed")
        verdict["notes"] = ";".join([verdict["notes"], *notes]).strip(";")
        # configuration rows carry no config_predates note (summary files never list seed etc.)
        verdict["notes"] = ";".join(n for n in verdict["notes"].split(";") if n and not n.startswith("config_predates"))
        rows.append({**_identity(experiment_name, alpha), **record, **verdict, "tabular_model": feature_model or "",
                     "source": source})
    out = pd.DataFrame(rows)
    return _finish(out, (*IDENTITY_COLUMNS[:6], "rank", "is_complete", "expected_folds", "completed_folds"))


def import_validation_root(validation_root: str | Path, *, checkpoint_root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Walk ``<root>/<experiment>/fusion_alpha_<a>/`` run directories (skips ``<root>/summary``).

    Returns ``{"folds", "thresholds"}``; ``thresholds`` is long
    (run_id, fold, branch, threshold) from ``fold_<k>/decision_thresholds.json``
    with the tabular model name normalized to ``tabular``.
    """
    validation_root = fs(validation_root)
    frames, lookup, threshold_rows = [], {}, []
    for run_dir, metrics_path in discover_runs(validation_root):
        legacy_run_id = run_dir.relative_to(validation_root).as_posix()
        metrics = pd.read_csv(metrics_path)
        if metrics.empty or "fold" not in metrics.columns:
            continue
        config_path = run_dir / "validation_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        lookup[legacy_run_id] = config
        metrics.insert(0, "run_id", legacy_run_id)
        frames.append(metrics)
        feature_model = config.get("feature_model")
        for thr_path in sorted(run_dir.glob("fold_*/decision_thresholds.json")):
            fold = int(thr_path.parent.name.split("_", 1)[1])
            for branch, value in json.loads(thr_path.read_text(encoding="utf-8")).items():
                threshold_rows.append({"legacy_run_id": legacy_run_id, "fold": fold,
                                       "branch": "tabular" if branch == feature_model or branch in TABULAR_MODELS
                                       else branch, "threshold": float(value)})
    if not frames:
        raise FileNotFoundError(f"No legacy validation_metrics.csv below {display(validation_root)}")
    folds = import_fold_rows(pd.concat(frames, ignore_index=True, sort=False), config_lookup=lookup,
                             checkpoint_root=checkpoint_root, source="validation_root")
    thresholds = pd.DataFrame(threshold_rows, columns=["legacy_run_id", "fold", "branch", "threshold"])
    if not thresholds.empty:
        ids = folds[["legacy_run_id", "run_id"]].drop_duplicates()
        thresholds = thresholds.merge(ids, on="legacy_run_id", how="left")[
            ["run_id", "legacy_run_id", "fold", "branch", "threshold"]]
    return {"folds": folds, "thresholds": thresholds}


def run_id_map(*frames: pd.DataFrame) -> pd.DataFrame:
    """Unique legacy -> new run-id mapping with the retraining verdict."""
    cols = ["legacy_run_id", "legacy_experiment_name", "fusion_alpha", "run_id", "train_id", "eval_id", "workflow",
            "protocol", "requires_retraining", "retraining_reasons", "split_metadata"]
    parts = [f[[c for c in cols if c in f.columns]] for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).drop_duplicates("legacy_run_id").sort_values("legacy_run_id") \
        .reset_index(drop=True)


def retraining_report(mapping: pd.DataFrame) -> list[str]:
    lines = ["# Legacy import report", "",
             f"- Legacy runs: {len(mapping)}",
             f"- Requiring retraining: {int(mapping['requires_retraining'].sum()) if len(mapping) else 0}", ""]
    for reason, text in RETRAINING_REASONS.items():
        hits = mapping[mapping["retraining_reasons"].fillna("").str.contains(reason, regex=False)] if len(mapping) else []
        if len(hits):
            lines.append(f"## {reason} ({len(hits)})")
            lines.append("")
            lines.append(text)
            lines.append("")
            lines += [f"- `{r}`" for r in hits["legacy_run_id"]]
            lines.append("")
    lines.append("Imported metrics are historical context, not new results.")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mribreast import-legacy",
                                     description="Normalize historical legacy validation CSV/JSON (read-only).")
    parser.add_argument("--summary_dir", "--summary-dir", default=None)
    parser.add_argument("--validation_root", "--validation-root", default=None)
    parser.add_argument("--checkpoint_root", "--checkpoint-root", default=None,
                        help="Optional legacy checkpoint root to verify split metadata and fold artifacts.")
    parser.add_argument("--output_dir", "--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.summary_dir and not args.validation_root:
        parser.error("give --summary_dir and/or --validation_root")
    out = Path(args.output_dir)
    for source in (args.summary_dir, args.validation_root):
        if source and (out.resolve() == Path(source).resolve() or Path(source).resolve() in out.resolve().parents):
            parser.error("--output_dir must not be inside a legacy input directory (inputs stay read-only)")
    out.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    if args.summary_dir:
        summary = import_summary_dir(args.summary_dir, checkpoint_root=args.checkpoint_root)
        for key, frame in summary.items():
            frame.to_csv(out / f"legacy_summary_{key}.csv", index=False)
            frames.append(frame)
    if args.validation_root:
        runs = import_validation_root(args.validation_root, checkpoint_root=args.checkpoint_root)
        runs["folds"].to_csv(out / "legacy_run_folds.csv", index=False)
        runs["thresholds"].to_csv(out / "legacy_run_thresholds.csv", index=False)
        frames.append(runs["folds"])
    mapping = run_id_map(*frames)
    mapping.to_csv(out / "run_id_map.csv", index=False)
    (out / "import_report.md").write_text("\n".join(retraining_report(mapping)) + "\n", encoding="utf-8")
    print(f"import-legacy: {len(mapping)} legacy runs, "
          f"{int(mapping['requires_retraining'].sum()) if len(mapping) else 0} require retraining -> {out}")
    return 0


def iter_reasons(mapping: pd.DataFrame) -> Iterable[tuple[str, str]]:
    for record in mapping.itertuples(index=False):
        for reason in str(getattr(record, "retraining_reasons", "") or "").split(";"):
            if reason:
                yield record.legacy_run_id, reason
