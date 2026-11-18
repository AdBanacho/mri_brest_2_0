"""Write new evaluation results in the legacy CSV/JSON layout, and the legacy summary.

Legacy layout (``validate_configurable_imaging_features_fusion.py`` +
``summarize_configurable_imaging_features_fusion.py`` at SHA 127a2604)::

    <legacy_root>/<experiment_name>/fusion_alpha_<a>/
        validation_metrics.csv            one row per fold, legacy column order
        validation_summary.csv            metric,mean,std (np.std, ddof=0)
        validation_config.json            vars(args) of the legacy validate CLI + experiment_name + device
        threshold_calibration_metrics.csv fold,branch,threshold,balanced_accuracy,sensitivity,specificity,calibration_rows
        fold_<k>/decision_thresholds.json {"fusion": t, "image": t, "<feature_model>": t} (sorted, indent 2)
        fold_<k>/threshold_calibration_metrics.csv
        fold_<k>/threshold_calibration_predictions.csv
        fold_<k>/fusion_validation_predictions.csv
    <summary_root>/ all_validation_metrics.csv configuration_summary.csv ranked_configurations.csv
                    best_configuration.json validation_report.md [fusion_metric_comparison.png branch_auc_comparison.png]

Branch naming: the new pipeline names the tabular branch ``tabular``; legacy
per-run files name it after the model (``xgboost``/``mlp``) and the legacy
summarizer then *adds* ``tabular_*`` duplicates.  The exporter writes the
model-named columns; :func:`summarize_legacy` restores the ``tabular_*``
duplicates exactly as the legacy summarizer did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.legacy._fs import display, fs
from mribreast.legacy.names import alpha_dir, build_experiment_name

# --------------------------------------------------------------------------- column contracts
#: ``probability_metrics`` insertion order (decision_threshold only when a threshold is supplied).
LEGACY_BRANCH_METRICS = ("accuracy", "balanced_accuracy", "sensitivity", "decision_threshold", "specificity", "auc_roc")
LEGACY_FOLD_INFO = ("fold", "checkpoint_path", "inner_fit_rows", "inner_calibration_rows",
                    "effective_inner_calibration_folds", "outer_validation_rows")
#: Summarizer constants (verbatim).
CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc_roc", "decision_threshold")
RANKING_COLUMNS = ("fusion_auc_roc_mean", "fusion_balanced_accuracy_mean", "fusion_sensitivity_mean",
                   "fusion_specificity_mean")
CONFIG_COLUMNS = ("mri_model", "subtraction_mode", "feature_groups", "feature_model", "feature_selector", "batch_size",
                  "positive_boost", "sensitivity_lambda", "lr", "fusion_alpha")
THRESHOLD_COLUMNS = ("fold", "branch", "threshold", "balanced_accuracy", "sensitivity", "specificity",
                     "calibration_rows")
#: New-pipeline column aliases accepted for the legacy row-count columns.
ROW_COUNT_ALIASES = {
    "inner_fit_rows": ("inner_fit_rows", "fit_rows", "n_fit"),
    "inner_calibration_rows": ("inner_calibration_rows", "calibration_rows", "threshold_rows", "n_calibration"),
    "effective_inner_calibration_folds": ("effective_inner_calibration_folds", "effective_inner_folds"),
    "outer_validation_rows": ("outer_validation_rows", "outer_eval_rows", "n_outer_eval"),
    "checkpoint_path": ("checkpoint_path", "image_checkpoint"),
}
#: Keys of ``vars(args)`` of the legacy validate CLI, in parser order (validation_config.json is sorted anyway).
LEGACY_VALIDATE_ARGS = (
    "mri_model", "subtraction_mode", "feature_groups", "feature_model", "feature_selector", "imaging_features_file",
    "imaging_patient_id_column", "allow_missing_imaging_features", "include_sensitive", "use_annotation_boxes",
    "annotation_boxes_file", "clinical_features_file", "metadata_file", "image_root", "prepared_root",
    "subtraction_root", "target_size", "seed", "experiment_id", "mri_extra_json", "lasso_cv_folds", "lasso_cs",
    "lasso_max_iter", "lasso_tolerance", "lasso_min_features", "lasso_n_jobs", "num_folds",
    "threshold_calibration_folds", "batch_size", "num_workers", "positive_boost", "sensitivity_lambda", "lr",
    "fusion_alpha", "checkpoint_root", "output_dir",
)
BRANCH_ORDER = ("image", "tabular", "fusion")


def legacy_branch_name(branch: str, feature_model: str) -> str:
    return feature_model if branch == "tabular" else branch


def _first_present(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def to_legacy_fold_metrics(metrics: pd.DataFrame, feature_model: str) -> pd.DataFrame:
    """New per-fold metrics (``<branch>_<metric>``, branch in image/tabular/fusion) -> legacy columns/order.

    Columns that the new frame lacks are omitted (legacy files also omitted
    ``*_decision_threshold`` when no threshold was tuned); extra new columns are dropped.
    """
    if "fold" not in metrics.columns:
        raise ValueError("metrics need a 'fold' column")
    out = pd.DataFrame(index=metrics.index)
    for legacy in LEGACY_FOLD_INFO:
        source = "fold" if legacy == "fold" else _first_present(metrics, ROW_COUNT_ALIASES[legacy])
        if source is not None:
            out[legacy] = metrics[source].to_numpy()
    for branch in BRANCH_ORDER:
        name = legacy_branch_name(branch, feature_model)
        for metric in LEGACY_BRANCH_METRICS:
            source = f"{branch}_{metric}"
            if source in metrics.columns:
                out[f"{name}_{metric}"] = metrics[source].to_numpy()
    return out.reset_index(drop=True)


def legacy_validation_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """``validation_summary.csv``: numeric columns except ``fold``; mean and ``np.std`` (ddof=0)."""
    numeric = fold_metrics.select_dtypes(include="number")
    rows = []
    for column in numeric.columns:
        if column == "fold":
            continue
        values = numeric[column].dropna().to_numpy()
        if len(values):
            rows.append({"metric": column, "mean": float(values.mean()), "std": float(values.std())})
    return pd.DataFrame(rows, columns=["metric", "mean", "std"])


def legacy_validation_config(cfg: Mapping[str, Any], fusion_alpha: float, *, experiment_name: str | None = None,
                             device: str = "cpu", output_dir: str = "", experiment_id: str | None = None) -> dict[str, Any]:
    """Reconstruct the legacy ``validation_config.json`` (``vars(args)`` + experiment_name + device).

    ``cfg`` uses new field names; ``num_workers`` comes from ``validate_num_workers``
    (legacy ``VALIDATE[num_workers]``).  Values keep legacy JSON types (lists for
    ``feature_groups``/``target_size``, the raw JSON string for ``mri_extra_json``).
    """
    groups = cfg["feature_groups"]
    groups = [g for g in (groups.split(",") if isinstance(groups, str) else groups) if g]
    target = cfg.get("target_size", (256, 256, 64))
    target = [int(v) for v in (target.split() if isinstance(target, str) else target)]
    extra = cfg.get("mri_extra_json", {})
    values = {
        "mri_model": cfg["mri_model"], "subtraction_mode": cfg["subtraction_mode"], "feature_groups": groups,
        "feature_model": cfg["feature_model"], "feature_selector": cfg.get("feature_selector", "none"),
        "imaging_features_file": str(cfg.get("imaging_features_file", "")),
        "imaging_patient_id_column": cfg.get("imaging_patient_id_column", "Patient ID"),
        "allow_missing_imaging_features": bool(cfg.get("allow_missing_imaging_features", False)),
        "include_sensitive": bool(cfg.get("include_sensitive", False)),
        "use_annotation_boxes": bool(cfg.get("use_annotation_boxes", False)),
        "annotation_boxes_file": str(cfg.get("annotation_boxes_file", "")),
        "clinical_features_file": str(cfg.get("clinical_features_file", "")),
        "metadata_file": str(cfg.get("metadata_file", "")), "image_root": str(cfg.get("image_root", "")),
        "prepared_root": str(cfg.get("prepared_root", "")), "subtraction_root": str(cfg.get("subtraction_root", "")),
        "target_size": target, "seed": int(cfg.get("seed", 42)),
        "experiment_id": experiment_id if experiment_id is not None else str(cfg.get("experiment_id", "") or ""),
        "mri_extra_json": extra if isinstance(extra, str) else json.dumps(extra, sort_keys=True),
        "lasso_cv_folds": int(cfg.get("lasso_cv_folds", 5)), "lasso_cs": int(cfg.get("lasso_cs", 20)),
        "lasso_max_iter": int(cfg.get("lasso_max_iter", 5000)), "lasso_tolerance": float(cfg.get("lasso_tolerance", 1e-4)),
        "lasso_min_features": int(cfg.get("lasso_min_features", 1)), "lasso_n_jobs": int(cfg.get("lasso_n_jobs", 8)),
        "num_folds": int(cfg.get("num_folds", 5)),
        "threshold_calibration_folds": int(cfg.get("threshold_calibration_folds", 5)),
        "batch_size": int(cfg.get("batch_size", 4)),
        "num_workers": int(cfg.get("validate_num_workers", cfg.get("num_workers", 4))),
        "positive_boost": float(cfg.get("positive_boost", 1.0)),
        "sensitivity_lambda": float(cfg.get("sensitivity_lambda", 0.05)), "lr": float(cfg.get("lr", 1e-4)),
        "fusion_alpha": float(fusion_alpha), "checkpoint_root": str(cfg.get("checkpoint_root", "")),
        "output_dir": output_dir,
    }
    assert tuple(values) == LEGACY_VALIDATE_ARGS
    name = experiment_name or build_experiment_name({**cfg, "experiment_id": values["experiment_id"]})
    return {**values, "experiment_name": name, "device": device}


def _dump_json(path: Path, payload: Any) -> None:
    path = fs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = fs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def to_legacy_threshold_metrics(thresholds: pd.DataFrame, feature_model: str) -> pd.DataFrame:
    """New ``threshold_calibration_metrics.csv`` -> legacy columns; ``tabular`` branch renamed to the model."""
    frame = thresholds.copy()
    if "calibration_rows" not in frame.columns:
        for alias in ("n_rows", "rows", "threshold_rows"):
            if alias in frame.columns:
                frame["calibration_rows"] = frame[alias]
                break
    frame["branch"] = frame["branch"].map(lambda b: legacy_branch_name(str(b), feature_model))
    order = {legacy_branch_name(b, feature_model): i for i, b in enumerate(BRANCH_ORDER)}
    frame = frame.assign(_o=frame["branch"].map(order)).sort_values(["fold", "_o"], kind="stable")
    return frame[[c for c in THRESHOLD_COLUMNS if c in frame.columns]].reset_index(drop=True)


def to_legacy_predictions(predictions: pd.DataFrame, feature_model: str, thresholds: Mapping[str, float] | None,
                          num_classes: int) -> pd.DataFrame:
    """Long new predictions (one row per study x branch) of ONE fold/split -> legacy wide prediction CSV."""
    wide = None
    for branch in BRANCH_ORDER:
        part = predictions[predictions["branch"] == branch]
        if part.empty:
            continue
        name = legacy_branch_name(branch, feature_model)
        cols = {f"prob_{k}": f"{name}_probability_{k}" for k in range(num_classes)}
        keep = part[["patient_id", "study_id", "label", *cols]].rename(columns=cols)
        if "decision" in part.columns:
            keep[f"{name}_prediction"] = part["decision"].to_numpy()
        wide = keep if wide is None else wide.merge(keep, on=["patient_id", "study_id", "label"], how="outer",
                                                    sort=False, validate="one_to_one")
    if wide is None:
        raise ValueError("no branch predictions")
    wide = wide.rename(columns={"patient_id": "patientId", "study_id": "studyId"})
    names = [legacy_branch_name(b, feature_model) for b in BRANCH_ORDER
             if f"{legacy_branch_name(b, feature_model)}_probability_0" in wide.columns]
    ordered = ["patientId", "studyId", "label"]
    for k in range(num_classes):
        ordered += [f"{n}_probability_{k}" for n in names]
    ordered += [f"{n}_prediction" for n in names if f"{n}_prediction" in wide.columns]
    out = wide[ordered].copy()
    for name, value in (thresholds or {}).items():
        out[f"{name}_decision_threshold"] = float(value)
    return out


def write_legacy_run(legacy_root: str | Path, cfg: Mapping[str, Any], fusion_alpha: float, fold_metrics: pd.DataFrame,
                     *, thresholds: Mapping[int, Mapping[str, float]] | None = None,
                     threshold_metrics: pd.DataFrame | None = None,
                     predictions: Mapping[str, pd.DataFrame] | None = None, num_classes: int = 2,
                     experiment_name: str | None = None, device: str = "cpu",
                     already_legacy_columns: bool = False) -> Path:
    """Write one evaluation in the legacy per-run layout and return the run directory.

    ``fold_metrics`` uses the new column names unless ``already_legacy_columns``.
    ``thresholds`` maps fold -> {branch (new names): threshold}.
    ``predictions`` maps ``"calibration"``/``"validation"`` to long prediction frames (contracts.PRED_COLUMNS).
    """
    feature_model = str(cfg["feature_model"])
    legacy_root = Path(legacy_root)
    config = legacy_validation_config(cfg, fusion_alpha, experiment_name=experiment_name, device=device,
                                      output_dir=str(legacy_root))
    run_dir = legacy_root / config["experiment_name"] / alpha_dir(fusion_alpha)
    metrics = fold_metrics.copy() if already_legacy_columns else to_legacy_fold_metrics(fold_metrics, feature_model)
    _write_csv(run_dir / "validation_metrics.csv", metrics)
    _write_csv(run_dir / "validation_summary.csv", legacy_validation_summary(metrics))
    _dump_json(run_dir / "validation_config.json", config)
    for fold, values in sorted((thresholds or {}).items()):
        _dump_json(run_dir / f"fold_{int(fold)}" / "decision_thresholds.json",
                   {legacy_branch_name(b, feature_model): float(t) for b, t in values.items()})
    if threshold_metrics is not None:
        legacy_thr = to_legacy_threshold_metrics(threshold_metrics, feature_model)
        _write_csv(run_dir / "threshold_calibration_metrics.csv", legacy_thr)
        for fold, part in legacy_thr.groupby("fold", sort=True):
            _write_csv(run_dir / f"fold_{int(fold)}" / "threshold_calibration_metrics.csv", part)
    file_for_split = {"calibration": "threshold_calibration_predictions.csv",
                      "validation": "fusion_validation_predictions.csv"}
    for split, frame in (predictions or {}).items():
        for fold, part in frame.groupby("fold", sort=True):
            fold_thr = {legacy_branch_name(b, feature_model): t
                        for b, t in ((thresholds or {}).get(int(fold), {}) or {}).items()}
            _write_csv(run_dir / f"fold_{int(fold)}" / file_for_split[split],
                       to_legacy_predictions(part, feature_model, fold_thr, num_classes))
    return run_dir


# --------------------------------------------------------------------------- new eval dir adapter
def _read_thresholds(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "folds" in raw:
        raw = raw["folds"]
    out: dict[int, dict[str, float]] = {}
    if isinstance(raw, list):
        for item in raw:
            if "branch" in item:
                out.setdefault(int(item["fold"]), {})[str(item["branch"])] = item["threshold"]
            else:
                fold = int(item.pop("fold"))
                out[fold] = {k: v for k, v in item.items() if v is not None}
    elif isinstance(raw, dict):
        for fold, values in raw.items():
            if str(fold).isdigit() and isinstance(values, Mapping):
                out[int(fold)] = {k: v for k, v in values.items() if v is not None}
    return {f: {b: float(t) for b, t in v.items() if t is not None} for f, v in out.items()}


def _read_resolved_config(eval_dir: Path) -> dict[str, Any]:
    for name in ("evaluation_manifest.json", "validation_config.json"):
        path = eval_dir / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("resolved_config", "config"):
                if isinstance(data.get(key), Mapping) and "mri_model" in data[key]:
                    return dict(data[key])
            if "mri_model" in data:
                return dict(data)
    raise FileNotFoundError(f"no resolved configuration in {eval_dir}")


def export_evaluation_dir(eval_dir: str | Path, legacy_root: str | Path, *, device: str = "cpu",
                          experiment_id: str | None = None) -> Path:
    """Convert one new ``<validation_root>/<train_id>/<eval_id>`` directory into the legacy layout.

    ``experiment_id`` defaults to the new ``train_id`` (directory name one level
    up) so that two new trainings never collide in the legacy name space.
    """
    eval_dir = Path(eval_dir)
    cfg = _read_resolved_config(eval_dir)
    if experiment_id is None:
        experiment_id = eval_dir.parent.name
    cfg = {**cfg, "experiment_id": experiment_id}
    metrics = pd.read_csv(fs(eval_dir) / "validation_metrics.csv")
    thr_path = eval_dir / "threshold_calibration_metrics.csv"
    predictions = {}
    for split, name in (("calibration", "predictions_calibration.csv"), ("validation", "predictions_validation.csv")):
        if (eval_dir / name).is_file():
            predictions[split] = pd.read_csv(eval_dir / name, dtype={"patient_id": str, "study_id": str})
    num_classes = 3 if cfg.get("label_scheme") == "oncotype_3class_18_31" else 2
    return write_legacy_run(legacy_root, cfg, float(cfg.get("fusion_alpha", 0.5)), metrics,
                            thresholds=_read_thresholds(eval_dir / "decision_thresholds.json"),
                            threshold_metrics=pd.read_csv(thr_path) if thr_path.is_file() else None,
                            predictions=predictions, num_classes=num_classes, device=device)


# --------------------------------------------------------------------------- legacy summarizer (verbatim port)
def _json_value(value: Any) -> Any:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _read_config(run_dir: Path) -> dict[str, Any]:
    path = fs(run_dir) / "validation_config.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read validation configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Validation configuration must contain an object: {path}")
    return {key: _json_value(item) for key, item in value.items()}


def canonicalize_tabular_metrics(metrics: pd.DataFrame, feature_model: str | None) -> pd.DataFrame:
    """Append ``tabular_<metric>`` duplicates of ``<feature_model>_<metric>`` (legacy summarizer)."""
    output = metrics.copy()
    if not feature_model:
        return output
    for metric in CLASSIFICATION_METRICS:
        source = f"{feature_model}_{metric}"
        if source in output.columns:
            output[f"tabular_{metric}"] = output[source]
    return output


def discover_runs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Run directories below ``input_dir`` (extended-length paths on Windows), skipping ``summary``."""
    input_dir = fs(input_dir)
    summary_dir = input_dir / "summary"
    found = []
    for metrics_path in sorted(input_dir.rglob("validation_metrics.csv")):
        if summary_dir == metrics_path.parent or summary_dir in metrics_path.parents:
            continue
        found.append((metrics_path.parent, metrics_path))
    return found


def collect_results(input_dir: str | Path, fallback_expected_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Port of legacy ``collect_results``: fold rows + one configuration row per run."""
    input_dir = fs(input_dir)
    if fallback_expected_folds < 2:
        raise ValueError("expected_folds must be at least 2.")
    fold_frames: list[pd.DataFrame] = []
    configuration_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    runs = discover_runs(input_dir)
    if not runs:
        raise FileNotFoundError(f"No validation_metrics.csv files were found below {display(input_dir)}")
    for run_dir, metrics_path in runs:
        try:
            metrics = pd.read_csv(metrics_path)
            if metrics.empty:
                warnings.append(f"Skipped empty metrics file: {display(metrics_path)}")
                continue
            if "fold" not in metrics.columns:
                warnings.append(f"Skipped metrics file without a fold column: {display(metrics_path)}")
                continue
            config = _read_config(run_dir)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            warnings.append(f"Skipped unreadable run {display(run_dir)}: {error}")
            continue
        feature_model = str(config.get("feature_model", "")) or None
        metrics = canonicalize_tabular_metrics(metrics, feature_model)
        metric_columns = [c for c in metrics.select_dtypes(include="number").columns if c != "fold"]
        run_id = run_dir.relative_to(input_dir).as_posix()  # legacy (Linux) used '/' separators
        metrics.insert(0, "run_id", run_id)
        for key, value in config.items():
            if key not in metrics.columns:
                metrics[key] = value
        fold_frames.append(metrics)
        expected_folds = int(config.get("num_folds", fallback_expected_folds))
        numeric_folds = pd.to_numeric(metrics["fold"], errors="coerce").dropna()
        observed_folds = sorted({int(value) for value in numeric_folds})
        complete = observed_folds == list(range(1, expected_folds + 1))
        row: dict[str, Any] = {
            "run_id": run_id, **{key: config.get(key) for key in CONFIG_COLUMNS},
            "expected_folds": expected_folds, "completed_folds": len(observed_folds),
            "observed_fold_ids": ",".join(map(str, observed_folds)), "is_complete": complete,
        }
        for column in metric_columns:
            values = metrics[column].dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        configuration_rows.append(row)
    if not fold_frames:
        raise RuntimeError(f"No readable validation results found below {input_dir}\n" + "\n".join(warnings))
    return pd.concat(fold_frames, ignore_index=True, sort=False), pd.DataFrame(configuration_rows), warnings


def rank_configurations(configurations: pd.DataFrame, include_incomplete: bool = False) -> pd.DataFrame:
    """Legacy ranking: mean fusion ROC-AUC, balanced accuracy, sensitivity, specificity (desc, NaN last)."""
    candidates = configurations.copy()
    if not include_incomplete:
        candidates = candidates[candidates["is_complete"]].copy()
    sort_columns = [c for c in RANKING_COLUMNS if c in candidates]
    if not sort_columns:
        raise ValueError("Cannot rank configurations because no fusion performance metrics were found.")
    candidates = candidates.sort_values(sort_columns, ascending=[False] * len(sort_columns),
                                        na_position="last").reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _configuration_label(row: pd.Series) -> str:
    groups = str(row.get("feature_groups", "?"))
    if len(groups) > 34:
        groups = groups[:31] + "..."
    return (f"{row.get('mri_model', '?')} | {row.get('subtraction_mode', '?')} | {groups} | "
            f"{row.get('feature_model', '?')} | boost={row.get('positive_boost', '?')}")


def _bar_chart(ranked: pd.DataFrame, output_path: Path, top_n: int, specs: Sequence[tuple[str, str]], min_series: int,
               xlabel: str, title: str) -> None:
    available = [(c, label) for c, label in specs if c in ranked]
    if len(available) < min_series or ranked.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ranked.head(top_n).iloc[::-1]
    y = np.arange(len(shown))
    height = 0.8 / len(available)
    fig, ax = plt.subplots(figsize=(13, max(5.0, 0.55 * len(shown) + 2.0)))
    for index, (column, label) in enumerate(available):
        ax.barh(y + (index - (len(available) - 1) / 2) * height, shown[column], height=height, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([_configuration_label(row) for _, row in shown.iterrows()])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report_lines(configurations: pd.DataFrame, ranked: pd.DataFrame, warnings: list[str]) -> list[str]:
    completed = int(configurations["is_complete"].sum())
    lines = ["# Configurable fusion validation report", "",
             f"- Discovered configurations: {len(configurations)}",
             f"- Complete configurations: {completed}",
             f"- Incomplete configurations: {len(configurations) - completed}"]
    if not ranked.empty:
        best = ranked.iloc[0]
        lines += [f"- Best configuration: `{best['run_id']}`",
                  f"- Best mean fusion AUC ROC: {best.get('fusion_auc_roc_mean', float('nan')):.4f}"]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {item}" for item in warnings]
    return lines


def summarize_legacy(input_dir: str | Path, output_dir: str | Path | None = None, *, expected_folds: int = 5,
                     top_n: int = 20, include_incomplete: bool = False, plots: bool = True) -> Path:
    """Port of legacy ``generate_summary`` (same files, columns, ranking and report text)."""
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else input_dir / "summary"
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    output_dir = fs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, configurations, warnings = collect_results(input_dir, fallback_expected_folds=expected_folds)
    ranked = rank_configurations(configurations, include_incomplete)
    folds.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    configurations.sort_values("run_id").to_csv(output_dir / "configuration_summary.csv", index=False)
    ranked.to_csv(output_dir / "ranked_configurations.csv", index=False)
    if not ranked.empty:
        best = {key: _json_safe(value) for key, value in ranked.iloc[0].items()}
        (output_dir / "best_configuration.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n",
                                                             encoding="utf-8")
    if plots:
        _bar_chart(ranked, output_dir / "fusion_metric_comparison.png", top_n,
                   (("fusion_auc_roc_mean", "Fusion AUC ROC"), ("fusion_balanced_accuracy_mean", "Fusion balanced accuracy"),
                    ("fusion_sensitivity_mean", "Fusion sensitivity"), ("fusion_specificity_mean", "Fusion specificity")),
                   1, "Mean outer-fold score", "Top configurable fusion validation results")
        _bar_chart(ranked, output_dir / "branch_auc_comparison.png", top_n,
                   (("image_auc_roc_mean", "MRI"), ("tabular_auc_roc_mean", "Tabular"), ("fusion_auc_roc_mean", "Fusion")),
                   2, "Mean outer-fold AUC ROC", "MRI, tabular, and fusion branch comparison")
    (output_dir / "validation_report.md").write_text("\n".join(_report_lines(configurations, ranked, warnings)) + "\n",
                                                      encoding="utf-8")
    return Path(display(output_dir))


def export_many(eval_dirs: Iterable[str | Path], legacy_root: str | Path, **kwargs: Any) -> list[Path]:
    return [export_evaluation_dir(d, legacy_root, **kwargs) for d in eval_dirs]
