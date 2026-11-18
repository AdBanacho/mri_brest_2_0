"""Decision thresholds tuned on calibration data only.

* ``balanced_accuracy`` (legacy, ``threshold_tuning.py::select_balanced_accuracy_threshold``):
  candidates = ``unique({0, 0.5, 1} U scores)`` (ascending); maximise balanced
  accuracy, then sensitivity, then ``-|t - 0.5|``; a complete tie keeps the
  **first (lowest)** candidate because Python's ``max`` returns the first
  maximal element. Decision is ``p >= t``.
* ``fixed_0.5``: threshold 0.5 (binary) - the calibration table still reports
  the calibration metrics at 0.5.
* multiclass: no threshold, decision = ``argmax`` (``threshold = None``).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from mribreast.reporting.metrics import binary_decisions

STRATEGIES = ("balanced_accuracy", "fixed_0.5")


def predictions_at_threshold(probabilities, threshold) -> np.ndarray:
    """Binary predictions ``p >= threshold`` (legacy ``predictions_at_threshold``)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    return binary_decisions(probabilities, float(threshold))


def _record(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = predictions_at_threshold(probabilities, threshold)
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(labels, predictions, pos_label=0, zero_division=0)),
    }


def select_balanced_accuracy_threshold(labels, probabilities) -> dict[str, float]:
    """Legacy-identical threshold search (labels must be exactly {0, 1})."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be aligned 1-D arrays.")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Threshold tuning requires binary labels encoded as 0 and 1.")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    records = [_record(labels, probabilities, threshold) for threshold in candidates]
    return max(
        records,
        key=lambda record: (record["balanced_accuracy"], record["sensitivity"], -abs(record["threshold"] - 0.5)),
    )


def tune_threshold(strategy: str, labels, prob_positive, positive_class: int = 1) -> dict[str, Any]:
    """Tune one binary threshold. ``labels`` are class ids; ``prob_positive`` = p[:, positive_class]."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown threshold strategy {strategy!r}")
    y = (np.asarray(labels).astype(np.int64) == positive_class).astype(np.int64)
    p = np.asarray(prob_positive, dtype=np.float64)
    if strategy == "balanced_accuracy":
        result = select_balanced_accuracy_threshold(y, p)
    else:
        if len(np.unique(y)) < 2:
            raise ValueError("calibration data needs both classes")
        result = _record(y, p, 0.5)
    return {"strategy": strategy, **result}


def decide(probs: np.ndarray, threshold: float | None, positive_class: int = 1) -> np.ndarray:
    """Class decisions: binary ``p[:, pos] >= t`` -> pos else the other class; multiclass argmax."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be [n, C]")
    if probs.shape[1] != 2 or threshold is None:
        if probs.shape[1] == 2 and threshold is None:
            raise ValueError("binary decisions need a threshold (use 0.5 for fixed_0.5)")
        return probs.argmax(axis=1).astype(np.int64)
    positive = binary_decisions(probs[:, positive_class], float(threshold))
    negative_class = 1 - positive_class
    return np.where(positive == 1, positive_class, negative_class).astype(np.int64)


def tune_branch_thresholds(strategy: str, labels: np.ndarray, branch_probs: Mapping[str, np.ndarray],
                           num_classes: int, positive_class: int = 1) -> tuple[dict[str, float | None], pd.DataFrame]:
    """Independent threshold per branch (binary) or ``None`` (multiclass argmax).

    Returns ``(thresholds, table)``; ``table`` has legacy columns
    ``branch, threshold, balanced_accuracy, sensitivity, specificity`` + ``strategy``.
    """
    thresholds: dict[str, float | None] = {}
    rows = []
    for branch, probs in branch_probs.items():
        if num_classes == 2:
            result = tune_threshold(strategy, labels, np.asarray(probs)[:, positive_class], positive_class)
            thresholds[branch] = result["threshold"]
            rows.append({"branch": branch, "threshold": result["threshold"],
                         "balanced_accuracy": result["balanced_accuracy"], "sensitivity": result["sensitivity"],
                         "specificity": result["specificity"], "strategy": strategy})
        else:
            pred = np.asarray(probs).argmax(axis=1)
            y = np.asarray(labels).astype(np.int64)
            thresholds[branch] = None
            rows.append({"branch": branch, "threshold": np.nan,
                         "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                         "sensitivity": float(np.mean(recall_score(y, pred, labels=np.arange(num_classes),
                                                                   average=None, zero_division=0))),
                         "specificity": np.nan, "strategy": "argmax"})
    return thresholds, pd.DataFrame(rows, columns=["branch", "threshold", "balanced_accuracy", "sensitivity",
                                                   "specificity", "strategy"])
