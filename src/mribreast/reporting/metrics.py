"""Shared metric definitions (train-time selection, validation, summary).

Binary: decision = p[:, positive] >= threshold.  Sensitivity = recall of the
positive class, specificity = recall of the negative class, ROC-AUC on
p[:, positive] (NaN if a single class is present).
Multiclass: decision = argmax; sensitivity/specificity are macro one-vs-rest;
ROC-AUC is macro OvR.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

METRIC_NAMES = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc_roc")


def binary_decisions(prob_positive: np.ndarray, threshold: float) -> np.ndarray:
    p = np.asarray(prob_positive, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("prob_positive must be a finite 1-D array within [0, 1]")
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be within [0, 1]")
    return (p >= threshold).astype(np.int64)


def binary_metrics(labels: np.ndarray, prob_positive: np.ndarray, threshold: float, positive_class: int = 1) -> dict[str, float]:
    y = (np.asarray(labels) == positive_class).astype(np.int64)
    pred = binary_decisions(prob_positive, threshold)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, prob_positive)) if len(np.unique(y)) > 1 else float("nan"),
        "decision_threshold": float(threshold),
    }
    return out


def multiclass_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    classes = np.arange(probs.shape[1])
    sens = recall_score(y, pred, labels=classes, average=None, zero_division=0)
    spec = []
    for k in classes:
        neg = y != k
        spec.append(float(np.mean(pred[neg] != k)) if neg.any() else float("nan"))
    try:
        auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=classes))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(np.mean(sens)),
        "specificity": float(np.nanmean(spec)),
        "auc_roc": auc,
    }


def confusion(labels: np.ndarray, decisions: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(np.asarray(labels, dtype=np.int64), np.asarray(decisions, dtype=np.int64),
                            labels=np.arange(num_classes))
