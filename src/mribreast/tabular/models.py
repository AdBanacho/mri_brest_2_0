"""Tabular model factories (legacy ``make_feature_model``) + sample weights + grouped MLP early stopping.

Sample-weight support
---------------------
Legacy passes ``sample_weight`` to ``XGBClassifier.fit`` and ``MLPClassifier.fit``.
``MLPClassifier.fit(sample_weight=...)`` exists only from scikit-learn **1.7**
(legacy pins ``scikit-learn==1.7.2`` in ``requirementsCuda.txt``); on older versions
the legacy call raises ``TypeError``.  :func:`require_mlp_sample_weight` checks the
installed signature and fails with an explicit message instead.  Compatible range:
``scikit-learn>=1.7,<1.8`` for exact legacy LASSO parameters (1.8 changes the L1
spelling, handled in ``lasso._l1_parameters``); pyproject pins ``<1.8``.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

LEGACY_MLP_VALIDATION_FRACTION = 0.15
LEGACY_MLP_N_ITER_NO_CHANGE = 20
MLP_EARLY_STOPPING_MODES = ("internal", "grouped", "off")


def balanced_sample_weights(labels: Sequence[int], positive_boost: float = 1.0, positive_class: int = 1,
                            num_classes: int | None = None) -> np.ndarray:
    """``compute_sample_weight('balanced')`` x ``positive_boost`` on the positive class (binary only).

    Legacy (``n_fold_cv_run.py:360-366``) boosts label 1; ``positive_class`` defaults to 1.
    """
    labels = np.asarray(labels)
    weights = compute_sample_weight(class_weight="balanced", y=labels).astype(np.float64)
    n_classes = num_classes if num_classes is not None else len(np.unique(labels))
    if n_classes == 2 and positive_boost != 1.0:
        weights = weights.copy()
        weights[labels == positive_class] *= positive_boost
    return weights


def parse_hidden_layers(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        layers = tuple(int(w.strip()) for w in value.split(",") if w.strip())
    else:
        layers = tuple(int(w) for w in value)
    if not layers or any(w <= 0 for w in layers):
        raise ValueError("mlp_hidden_layers must contain comma-separated positive integers.")
    return layers


def make_xgboost(cfg: Mapping[str, Any], num_classes: int, seed: int):
    from xgboost import XGBClassifier

    params: dict[str, Any] = {
        "n_estimators": int(cfg.get("xgb_n_estimators", 300)),
        "max_depth": int(cfg.get("xgb_max_depth", 3)),
        "learning_rate": float(cfg.get("xgb_learning_rate", 0.03)),
        "subsample": float(cfg.get("xgb_subsample", 0.8)),
        "colsample_bytree": float(cfg.get("xgb_colsample_bytree", 0.8)),
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": int(cfg.get("xgb_n_jobs", 8)),
        "objective": "binary:logistic" if num_classes == 2 else "multi:softprob",
        "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
    }
    if num_classes > 2:
        params["num_class"] = num_classes
    params.update(dict(cfg.get("xgb_extra_json") or {}))
    return XGBClassifier(**params)


def mlp_parameters(cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    mode = cfg.get("mlp_early_stopping", "internal")
    if mode not in MLP_EARLY_STOPPING_MODES:
        raise ValueError(f"mlp_early_stopping must be one of {MLP_EARLY_STOPPING_MODES}")
    params: dict[str, Any] = dict(
        hidden_layer_sizes=parse_hidden_layers(cfg.get("mlp_hidden_layers", "128,64")),
        activation="relu", solver="adam", alpha=float(cfg.get("mlp_alpha", 1e-4)),
        learning_rate_init=float(cfg.get("mlp_learning_rate", 1e-3)), max_iter=int(cfg.get("mlp_max_iter", 500)),
        early_stopping=mode == "internal", validation_fraction=LEGACY_MLP_VALIDATION_FRACTION,
        n_iter_no_change=LEGACY_MLP_N_ITER_NO_CHANGE, random_state=int(seed),
    )
    extra = dict(cfg.get("mlp_extra_json") or {})
    if mode != "internal" and extra.get("early_stopping"):
        raise ValueError("mlp_extra_json.early_stopping=true conflicts with mlp_early_stopping="
                         f"{mode}; use mlp_early_stopping=internal for sklearn's random holdout")
    params.update(extra)
    return params


def make_mlp(cfg: Mapping[str, Any], seed: int):
    params = mlp_parameters(cfg, seed)
    if cfg.get("mlp_early_stopping", "internal") == "grouped":
        return GroupedEarlyStoppingMLP(**params)
    return MLPClassifier(**params)


def make_tabular_model(cfg: Mapping[str, Any], num_classes: int, seed: int):
    model = cfg.get("feature_model", "xgboost")
    if model == "xgboost":
        return make_xgboost(cfg, num_classes, seed)
    if model == "mlp":
        return make_mlp(cfg, seed)
    raise ValueError(f"unknown feature_model {model}")


def require_mlp_sample_weight() -> None:
    if "sample_weight" not in inspect.signature(MLPClassifier.fit).parameters:
        import sklearn

        raise RuntimeError(f"scikit-learn {sklearn.__version__} MLPClassifier.fit has no sample_weight; legacy weighting "
                           "needs scikit-learn>=1.7 (legacy pin 1.7.2)")


class GroupedEarlyStoppingMLP(MLPClassifier):
    """``MLPClassifier`` whose early-stopping holdout is a patient-grouped split.

    sklearn's ``early_stopping=True`` draws a random 15 % row holdout, so studies of one
    patient can sit on both sides.  Here ``fit(X, y, sample_weight, groups=...)`` holds
    out ~``validation_fraction`` of the *patients* (first fold of a shuffled
    ``StratifiedGroupKFold`` with ``round(1/validation_fraction)`` splits), trains on
    the rest one epoch at a time with ``partial_fit`` (adam state carried over),
    scores the holdout after each epoch with the same rule as sklearn (weighted
    accuracy; improvement must exceed ``tol``; stop after ``n_iter_no_change`` epochs
    without improvement) and restores the best-epoch weights.  Without ``groups`` it
    raises (use ``mlp_early_stopping=internal`` or ``off``).
    """

    def fit(self, X, y, sample_weight=None, groups=None):  # noqa: D401
        if groups is None:
            raise ValueError("GroupedEarlyStoppingMLP.fit needs groups (patient ids)")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        groups = np.asarray(groups)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        n_splits = max(2, int(round(1.0 / float(self.validation_fraction))))
        n_splits = min(n_splits, pd.Series(groups).nunique())
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        train_idx, val_idx = next(splitter.split(X, y, groups))
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise RuntimeError("grouped early stopping split leaked a patient")
        classes = np.unique(y)
        max_epochs = int(self.max_iter)
        saved = (self.early_stopping, self.max_iter)
        self.early_stopping = False
        best_score, best_state, no_improve = -np.inf, None, 0
        scores: list[float] = []
        try:
            for epoch in range(max_epochs):
                kwargs = {"classes": classes} if epoch == 0 else {}
                if sw is not None:
                    kwargs["sample_weight"] = sw[train_idx]
                super().partial_fit(X[train_idx], y[train_idx], **kwargs)
                score = float(np.average(self.predict(X[val_idx]) == y[val_idx],
                                         weights=None if sw is None else sw[val_idx]))
                scores.append(score)
                no_improve = no_improve + 1 if score < best_score + self.tol else 0
                if score > best_score:
                    best_score = score
                    best_state = (copy.deepcopy(self.coefs_), copy.deepcopy(self.intercepts_), epoch + 1)
                if no_improve > self.n_iter_no_change:
                    break
        finally:
            self.early_stopping, self.max_iter = saved
        self.coefs_, self.intercepts_, self.best_epoch_ = best_state
        self.validation_scores_ = scores
        self.best_validation_score_ = best_score
        self.n_epochs_run_ = len(scores)
        self.holdout_groups_ = np.unique(groups[val_idx])
        self.train_groups_ = np.unique(groups[train_idx])
        return self
