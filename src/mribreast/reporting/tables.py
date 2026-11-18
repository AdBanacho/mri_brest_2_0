"""Validation tables: per-fold metric rows (legacy column names), pooled metrics, summaries.

Metric definitions live in :mod:`mribreast.reporting.metrics`; this module only
arranges them. Column naming follows the legacy ``<branch>_<metric>`` pattern
and the legacy per-branch order
``accuracy, balanced_accuracy, sensitivity, decision_threshold, specificity, auc_roc``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from mribreast.reporting.metrics import METRIC_NAMES, binary_metrics, confusion, multiclass_metrics

BRANCH_METRIC_ORDER = ("accuracy", "balanced_accuracy", "sensitivity", "decision_threshold", "specificity", "auc_roc")
BRANCH_ORDER = ("image", "tabular", "fusion")
#: Leading per-fold columns of validation_metrics.csv (legacy order).
FOLD_LEADING_COLUMNS = ("fold", "checkpoint_path", "inner_fit_rows", "inner_calibration_rows",
                        "effective_inner_calibration_folds", "outer_validation_rows")


def ordered_branches(branches: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(branches))
    return [b for b in BRANCH_ORDER if b in present] + [b for b in present if b not in BRANCH_ORDER]


def branch_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float | None, num_classes: int,
                   positive_class: int = 1) -> dict[str, float]:
    """Metrics for one branch on one evaluation set (binary threshold or multiclass argmax)."""
    if num_classes == 2:
        if threshold is None:
            raise ValueError("binary metrics need a decision threshold")
        return binary_metrics(labels, np.asarray(probs)[:, positive_class], threshold, positive_class)
    return multiclass_metrics(labels, probs)


def decision_metrics(labels: np.ndarray, decisions: np.ndarray, probs: np.ndarray, num_classes: int,
                     positive_class: int = 1) -> dict[str, float]:
    """Metrics from already-made decisions (pooled over folds with fold-specific thresholds).

    Same definitions as :func:`mribreast.reporting.metrics.binary_metrics` /
    ``multiclass_metrics``; ROC-AUC uses the pooled probabilities.
    """
    labels = np.asarray(labels, dtype=np.int64)
    decisions = np.asarray(decisions, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if num_classes != 2:
        if not np.array_equal(decisions, probs.argmax(axis=1)):
            raise ValueError("multiclass decisions must be argmax of the probabilities")
        return multiclass_metrics(labels, probs)
    y = (labels == positive_class).astype(np.int64)
    d = (decisions == positive_class).astype(np.int64)
    both = len(np.unique(y)) > 1
    return {
        "accuracy": float(np.mean(y == d)),
        "balanced_accuracy": float(balanced_accuracy_score(y, d)) if both else float("nan"),
        "sensitivity": float(recall_score(y, d, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, d, pos_label=0, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, probs[:, positive_class])) if both else float("nan"),
    }


def prefixed(branch: str, metrics: Mapping[str, float], num_classes: int) -> dict[str, float]:
    names = BRANCH_METRIC_ORDER if num_classes == 2 else tuple(m for m in BRANCH_METRIC_ORDER
                                                               if m != "decision_threshold")
    return {f"{branch}_{name}": metrics[name] for name in names if name in metrics}


def summary_table(fold_rows: pd.DataFrame, pooled: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Legacy ``metric, mean, std`` (population std, ddof=0) over numeric columns except ``fold`` + ``pooled``."""
    rows = []
    numeric = fold_rows.select_dtypes(include="number")
    for column in numeric.columns:
        if column == "fold":
            continue
        values = numeric[column].dropna().to_numpy()
        if not len(values):
            continue
        branch, _, metric = column.partition("_")
        pooled_value = pooled.get(branch, {}).get(metric, float("nan")) if metric in METRIC_NAMES else float("nan")
        rows.append({"metric": column, "mean": float(values.mean()), "std": float(values.std()),
                     "pooled": pooled_value})
    return pd.DataFrame(rows, columns=["metric", "mean", "std", "pooled"])


def branch_comparison(fold_rows: pd.DataFrame, pooled: Mapping[str, Mapping[str, float]], branches: Sequence[str],
                      primary: str) -> pd.DataFrame:
    rows = []
    for branch in branches:
        row: dict[str, Any] = {"branch": branch, "is_primary": branch == primary}
        for metric in METRIC_NAMES:
            column = f"{branch}_{metric}"
            values = fold_rows[column].dropna().to_numpy(dtype=float) if column in fold_rows else np.array([])
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std()) if len(values) else float("nan")
            row[f"{metric}_pooled"] = pooled.get(branch, {}).get(metric, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_payload(labels: np.ndarray, decisions: np.ndarray, num_classes: int) -> list[list[int]]:
    return confusion(labels, decisions, num_classes).astype(int).tolist()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"
