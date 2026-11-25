"""Cross-configuration summary of ``<validation_root>/<train_id>/<eval_id>/`` evaluations.

Port of legacy ``summarize_configurable_imaging_features_fusion.py``:

* per-run statistics: mean and population std (``ddof=0``) of every numeric
  per-fold column except ``fold``;
* completeness: observed fold ids == ``1..expected`` where ``expected`` comes from
  ``validation_config.json::num_folds`` (fallback ``expected_folds``); the new
  layout additionally requires the ``COMPLETE`` flag;
* ranking: ``sort_values([auc_roc, balanced_accuracy, sensitivity, specificity] means,
  ascending=False, na_position="last")`` - identical pandas call, so ties keep
  discovery order (sorted ``run_id``). The legacy ranking used the ``fusion_*``
  columns; here the ``primary_*`` columns (copies of the run's primary branch,
  ``fusion`` for fusion runs) are used, which is identical for fusion runs;
* incomplete runs stay visible in the CSVs but are ranked only with
  ``include_incomplete``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mribreast.artifacts import is_complete, write_json_atomic, write_text_atomic
from mribreast.reporting import plots
from mribreast.reporting.report import summary_report

CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc_roc", "decision_threshold")
RANKING_METRICS = ("auc_roc", "balanced_accuracy", "sensitivity", "specificity")
LEGACY_RANKING_COLUMNS = tuple(f"fusion_{m}_mean" for m in RANKING_METRICS)
RANKING_COLUMNS = tuple(f"primary_{m}_mean" for m in RANKING_METRICS)
CONFIG_COLUMNS = ("mri_model", "subtraction_mode", "feature_groups", "feature_model", "feature_selector", "batch_size",
                  "positive_boost", "sensitivity_lambda", "lr", "fusion_alpha")
NEW_CONFIG_COLUMNS = ("train_id", "eval_id", "profile", "label_scheme", "branches", "selection_split", "primary_branch",
                      "report_unit", "threshold_strategy", "image_dim", "mask_mode", "box_mode",
                      "selection_uses_outer_eval")


def _json_value(value: Any) -> Any:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonicalize_tabular(metrics: pd.DataFrame, feature_model: str | None) -> pd.DataFrame:
    """Legacy runs name the tabular branch after the model (``xgboost_*``); add ``tabular_*`` copies."""
    out = metrics.copy()
    if not feature_model:
        return out
    for metric in CLASSIFICATION_METRICS:
        source = f"{feature_model}_{metric}"
        if source in out.columns and f"tabular_{metric}" not in out.columns:
            out[f"tabular_{metric}"] = out[source]
    return out


def discover_runs(validation_root: Path, summary_root: Path | None = None) -> list[Path]:
    runs = []
    for metrics_path in sorted(Path(validation_root).glob("*/*/validation_metrics.csv"),
                               key=lambda p: (p.parent.parent.name, p.parent.name)):
        run_dir = metrics_path.parent
        if run_dir.name.startswith(".") or run_dir.parent.name.startswith("."):
            continue  # atomic staging / trash directories
        if summary_root is not None and Path(summary_root).resolve() in (run_dir.resolve(), *run_dir.resolve().parents):
            continue
        runs.append(run_dir)
    return runs


def _primary(config: dict[str, Any], metrics: pd.DataFrame) -> str:
    for key in ("primary_branch_resolved", "primary_branch"):
        value = config.get(key)
        if value and value != "auto":
            return str(value)
    for branch in ("fusion", "image", "tabular"):
        if f"{branch}_auc_roc" in metrics.columns:
            return branch
    return "fusion"


def collect_results(validation_root: Path, expected_folds: int, summary_root: Path | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if expected_folds < 2:
        raise ValueError("expected_folds must be at least 2.")
    validation_root = Path(validation_root)
    runs = discover_runs(validation_root, summary_root)
    if not runs:
        raise FileNotFoundError(f"No validation_metrics.csv files were found below {validation_root}/*/*/")
    fold_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_dir in runs:
        run_id = run_dir.relative_to(validation_root).as_posix()
        try:
            metrics = pd.read_csv(run_dir / "validation_metrics.csv")
            if metrics.empty:
                warnings.append(f"Skipped empty metrics file: {run_id}")
                continue
            if "fold" not in metrics.columns:
                warnings.append(f"Skipped metrics file without a fold column: {run_id}")
                continue
            config = {k: _json_value(v) for k, v in _read_json(run_dir / "validation_config.json").items()}
            manifest = _read_json(run_dir / "evaluation_manifest.json")
        except (OSError, ValueError, pd.errors.ParserError) as error:
            warnings.append(f"Skipped unreadable run {run_id}: {error}")
            continue
        has_flag = is_complete(run_dir)
        metrics = _canonicalize_tabular(metrics, str(config.get("feature_model", "")) or None)
        primary = _primary(config, metrics)
        metric_columns = [c for c in metrics.select_dtypes(include="number").columns if c != "fold"]
        metrics = metrics.drop(columns=["run_id"], errors="ignore")
        metrics.insert(0, "run_id", run_id)
        extra = {key: value for key, value in config.items() if key not in metrics.columns}
        if extra:
            metrics = pd.concat([metrics, pd.DataFrame([extra] * len(metrics), index=metrics.index)], axis=1)
        fold_frames.append(metrics)

        expected = int(config.get("num_folds", expected_folds))
        observed = sorted({int(v) for v in pd.to_numeric(metrics["fold"], errors="coerce").dropna()})
        complete = observed == list(range(1, expected + 1)) and has_flag
        if not has_flag:
            warnings.append(f"Run {run_id} has no COMPLETE flag (interrupted or imported); treated as incomplete")
        elif not complete:
            warnings.append(f"Run {run_id} has folds {observed}, expected 1..{expected}; treated as incomplete")
        config_view = dict(config)
        config_view["primary_branch"] = primary
        config_view.setdefault("train_id", run_dir.parent.name)
        config_view.setdefault("eval_id", run_dir.name)
        if "selection_uses_outer_eval" in manifest:
            config_view["selection_uses_outer_eval"] = bool(manifest["selection_uses_outer_eval"])
        row: dict[str, Any] = {
            "run_id": run_id,
            **{key: config_view.get(key) for key in CONFIG_COLUMNS + NEW_CONFIG_COLUMNS},
            "expected_folds": expected,
            "completed_folds": len(observed),
            "observed_fold_ids": ",".join(map(str, observed)),
            "has_complete_flag": has_flag,
            "is_complete": complete,
        }
        for column in metric_columns:
            values = metrics[column].dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        for metric in CLASSIFICATION_METRICS:
            for stat in ("mean", "std"):
                source = f"{primary}_{metric}_{stat}"
                if source in row:
                    row[f"primary_{metric}_{stat}"] = row[source]
        rows.append(row)
    if not fold_frames:
        raise RuntimeError(f"No readable validation results found below {validation_root}\n" + "\n".join(warnings))
    return pd.concat(fold_frames, ignore_index=True, sort=False), pd.DataFrame(rows), warnings


def rank_configurations(configurations: pd.DataFrame, include_incomplete: bool,
                        ranking_columns: tuple[str, ...] = RANKING_COLUMNS) -> pd.DataFrame:
    candidates = configurations.copy()
    if not include_incomplete:
        candidates = candidates[candidates["is_complete"]].copy()
    sort_columns = [c for c in ranking_columns if c in candidates]
    if not sort_columns:
        raise ValueError("Cannot rank configurations because no primary-branch performance metrics were found.")
    candidates = candidates.sort_values(sort_columns, ascending=[False] * len(sort_columns),
                                        na_position="last").reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None or (not isinstance(value, (list, dict, str, bool)) and pd.isna(value)):
        return None
    return value


def _csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n")


def summarize(validation_root: Path, summary_root: Path, *, expected_folds: int, top_n: int,
              include_incomplete: bool) -> dict[str, Any]:
    """Aggregate every evaluation, rank configurations, write the summary directory. Returns an overview dict."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    validation_root, summary_root = Path(validation_root), Path(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    folds, configurations, warnings = collect_results(validation_root, expected_folds, summary_root)
    ranked = rank_configurations(configurations, include_incomplete)

    write_text_atomic(summary_root / "all_validation_metrics.csv", _csv(folds))
    write_text_atomic(summary_root / "configuration_summary.csv", _csv(configurations.sort_values("run_id")))
    write_text_atomic(summary_root / "ranked_configurations.csv", _csv(ranked))
    best_path = summary_root / "best_configuration.json"
    best: dict[str, Any] | None = None
    if not ranked.empty:
        best = {key: _json_safe(value) for key, value in ranked.iloc[0].items()}
        write_json_atomic(best_path, best)
    elif best_path.exists():
        best_path.unlink()
    shown = ranked.head(top_n).iloc[::-1]
    labels = shown["run_id"].astype(str).tolist() if not shown.empty else []
    metric_series = {m.replace("primary_", "").replace("_mean", ""): shown[m].tolist()
                     for m in RANKING_COLUMNS if m in shown}
    if labels:
        plots.save_bars(summary_root / "fusion_metric_comparison.png", "Top configurations (primary branch)", labels,
                        metric_series, "Mean outer-fold score")
        branch_series = {b: shown[f"{b}_auc_roc_mean"].tolist() for b in ("image", "tabular", "fusion")
                         if f"{b}_auc_roc_mean" in shown}
        if len(branch_series) >= 2:
            plots.save_bars(summary_root / "branch_auc_comparison.png", "Branch ROC-AUC comparison", labels,
                            branch_series, "Mean outer-fold ROC-AUC")
    write_text_atomic(summary_root / "validation_report.md", summary_report(
        configurations=configurations, ranked=ranked, warnings=warnings, ranking_columns=RANKING_COLUMNS,
        include_incomplete=include_incomplete, top_n=top_n))
    complete = int(configurations["is_complete"].sum())
    return {
        "summary_dir": str(summary_root),
        "discovered": int(len(configurations)),
        "complete": complete,
        "incomplete": int(len(configurations) - complete),
        "ranked": int(len(ranked)),
        "ranked_run_ids": ranked["run_id"].tolist(),
        "best_run_id": best["run_id"] if best else None,
        "warnings": warnings,
    }
