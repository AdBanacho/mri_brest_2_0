"""LASSO (L1-logistic) feature selection with patient-grouped inner CV + cross-fold stability.

Port of legacy ``mriBreastDuke/dataLoaders/lasso_feature_selection.py`` and the
stability aggregation of ``n_fold_cv_run.py`` (lines 676-723).

``selector_preprocessing``:

* ``outer`` (legacy): the caller preprocesses the whole fit set once and the
  selector runs ``LogisticRegressionCV(penalty=l1, solver=saga)`` on that matrix,
  so imputation medians / scaling / one-hot levels of every inner subfold were
  learnt with the inner validation rows included.
* ``fold_local``: C is chosen by a manual grouped CV in which every subfold runs
  ``Pipeline([preprocessor_factory(), LogisticRegression(l1, saga, C)])`` fitted on
  the subfold-train rows only (same splits, same C grid ``np.logspace(-4, 4, cs)``,
  same unweighted ROC-AUC scoring and "first max of summed fold scores" rule as
  ``LogisticRegressionCV``).  The final coefficients come from
  ``LogisticRegression(C=best)`` fitted on the outer-preprocessed fit matrix
  (LogisticRegressionCV instead warm-starts its refit from the mean fold
  coefficients; documented difference).
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.metrics import get_scorer  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

FEATURE_GROUP_COLORS = {
    "clinical": "#4C78A8",
    "kinetic": "#F58518",
    "morphology": "#54A24B",
    "heterogeneity": "#E45756",
    "density": "#B279A2",
    "mricore": "#9D755D",
}
_PREFIX_GROUPS = ("kinetic", "morphology", "heterogeneity", "density", "mricore")


def feature_group(feature_name: str) -> str:
    """Family of an encoded feature name (legacy rule + ``density_`` / ``mricore_``)."""
    source = str(feature_name).split("__", 1)[-1]
    for group in _PREFIX_GROUPS:
        if source.startswith(f"{group}_"):
            return group
    return "clinical"


def _l1_parameters(estimator_cls) -> dict:
    params = inspect.signature(estimator_cls).parameters
    out: dict = {}
    if "penalty" in params and params["penalty"].default != "deprecated":
        out["penalty"] = "l1"           # scikit-learn <= 1.7 (legacy pin 1.7.2)
    else:
        out["l1_ratios" if estimator_cls is LogisticRegressionCV else "l1_ratio"] = (1.0,) if estimator_cls is LogisticRegressionCV else 1.0
    if "use_legacy_attributes" in params:
        out["use_legacy_attributes"] = False
    return out


class LassoFeatureSelector(TransformerMixin, BaseEstimator):
    """Select encoded predictors with non-zero L1-logistic coefficients."""

    def __init__(self, cv_folds: int = 5, cs: int = 20, max_iter: int = 5000, tolerance: float = 1e-4,
                 min_features: int = 1, coefficient_threshold: float = 1e-8, n_jobs: int = 1, random_state: int = 42):
        self.cv_folds = cv_folds
        self.cs = cs
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.min_features = min_features
        self.coefficient_threshold = coefficient_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ validation
    def _check_params(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.cs < 1:
            raise ValueError("cs must be at least 1.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if self.min_features < 1:
            raise ValueError("min_features must be at least 1.")
        if self.coefficient_threshold < 0:
            raise ValueError("coefficient_threshold cannot be negative.")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

    @staticmethod
    def _validate(features, labels, sample_weight, groups):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("features must have shape (samples, nonzero_features).")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("labels must be one-dimensional and align with features.")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values.")
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("LASSO feature selection requires at least two classes.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (len(features),):
                raise ValueError("sample_weight must align with features.")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
                raise ValueError("sample_weight must be finite and non-negative.")
            if not np.any(sample_weight > 0):
                raise ValueError("sample_weight must contain a positive value.")
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape != (len(features),):
                raise ValueError("groups must align with features.")
            if pd.isna(groups).any():
                raise ValueError("groups cannot contain missing values.")
        return features, labels, sample_weight, groups, classes

    def make_cv_splits(self, labels, groups, classes=None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Legacy split rule: effective folds = min(cv_folds, groups (or rows) of the rarest class)."""
        labels = np.asarray(labels)
        classes = np.unique(labels) if classes is None else classes
        dummy = np.zeros((len(labels), 1))
        if groups is None:
            counts = np.array([np.sum(labels == c) for c in classes])
            effective = min(self.cv_folds, int(counts.min()))
            if effective < 2:
                raise ValueError("Every class needs at least two training rows for LASSO CV.")
            splits = list(StratifiedKFold(n_splits=effective, shuffle=True, random_state=self.random_state).split(dummy, labels))
        else:
            table = pd.DataFrame({"group": groups, "label": labels})
            if (table.groupby("group", sort=False)["label"].nunique() > 1).any():
                raise ValueError("Each patient group must have exactly one label.")
            unique = table.drop_duplicates()
            per_class = np.array([unique.loc[unique["label"] == c, "group"].nunique() for c in classes])
            effective = min(self.cv_folds, int(per_class.min()))
            if effective < 2:
                raise ValueError("Every class needs at least two patient groups for LASSO CV.")
            splits = list(StratifiedGroupKFold(n_splits=effective, shuffle=True, random_state=self.random_state)
                          .split(dummy, labels, groups))
        self.effective_cv_folds_ = effective
        return splits

    # ------------------------------------------------------------------ fitting
    def fit(self, features, labels, sample_weight=None, groups=None, *, raw: pd.DataFrame | None = None,
            preprocessor_factory: Callable[[], TransformerMixin] | None = None):
        """Fit on the (outer-)preprocessed fit matrix.

        With ``raw`` + ``preprocessor_factory`` (``selector_preprocessing=fold_local``) the C
        search refits the preprocessor inside each inner subfold on ``raw`` rows.
        """
        self._check_params()
        features, labels, sample_weight, groups, classes = self._validate(features, labels, sample_weight, groups)
        if self.min_features > features.shape[1]:
            raise ValueError("min_features cannot exceed the number of input features.")
        splits = self.make_cv_splits(labels, groups, classes)
        scoring = "roc_auc" if len(classes) == 2 else "roc_auc_ovr_weighted"
        common = dict(solver="saga", tol=self.tolerance, max_iter=self.max_iter, random_state=self.random_state)
        if raw is None:
            self.selector_preprocessing_ = "outer"
            model = LogisticRegressionCV(Cs=self.cs, cv=splits, scoring=scoring, n_jobs=self.n_jobs, refit=True,
                                         **common, **_l1_parameters(LogisticRegressionCV))
            model.fit(features, labels, sample_weight=sample_weight)
            self.model_ = model
            self.best_c_ = float(np.ravel(model.C_)[0])
            self.cv_details_ = []
        else:
            if preprocessor_factory is None or len(raw) != len(features):
                raise ValueError("fold_local selection needs raw rows aligned with features and a preprocessor_factory")
            self.selector_preprocessing_ = "fold_local"
            grid = np.logspace(-4, 4, self.cs)
            scorer = get_scorer(scoring)
            scores = np.zeros((len(splits), len(grid)))
            self.cv_details_ = []
            raw = raw.reset_index(drop=True)
            for k, (train, test) in enumerate(splits):
                fitted_pre = None
                for j, c in enumerate(grid):
                    pipe = Pipeline([("preprocess", preprocessor_factory()),
                                     ("lasso", LogisticRegression(C=c, **common, **_l1_parameters(LogisticRegression)))])
                    fit_kwargs = {} if sample_weight is None else {"lasso__sample_weight": sample_weight[train]}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        pipe.fit(raw.iloc[train], labels[train], **fit_kwargs)
                    scores[k, j] = scorer(pipe, raw.iloc[test], labels[test])
                    fitted_pre = pipe.named_steps["preprocess"]
                self.cv_details_.append({"subfold": k, "n_train": int(len(train)), "n_test": int(len(test)),
                                         "train_index": train, "preprocessor": fitted_pre})
            best = int(scores.sum(axis=0).argmax())
            self.cv_scores_ = scores
            self.Cs_ = grid
            self.best_c_ = float(grid[best])
            model = LogisticRegression(C=self.best_c_, **common, **_l1_parameters(LogisticRegression))
            model.fit(features, labels, sample_weight=sample_weight)
            self.model_ = model

        coefficients = np.asarray(self.model_.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        importances = np.max(np.abs(coefficients), axis=0)
        support = importances > self.coefficient_threshold
        fallback = max(0, self.min_features - int(support.sum()))
        if fallback:
            order = np.argsort(-importances, kind="stable")
            support[order[: self.min_features]] = True
        self.n_features_in_ = features.shape[1]
        self.classes_ = np.asarray(self.model_.classes_)
        self.coefficients_ = coefficients
        self.feature_importances_ = importances
        self.support_ = support
        self.selected_indices_ = np.flatnonzero(support)
        self.minimum_feature_fallback_used_ = bool(fallback)
        return self

    def transform(self, features):
        check_is_fitted(self, "support_")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("features must be two-dimensional with the same columns used to fit the LASSO selector.")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values.")
        return features[:, self.support_]

    def fit_transform(self, features, labels=None, sample_weight=None, groups=None, **kwargs):
        return self.fit(features, labels, sample_weight=sample_weight, groups=groups, **kwargs).transform(features)

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "support_")
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features):
        check_is_fitted(self, "support_")
        input_features = np.asarray(input_features, dtype=object)
        if input_features.shape != (self.n_features_in_,):
            raise ValueError("input_features must match the fitted feature count.")
        return input_features[self.support_].astype(str).tolist()

    def coefficient_columns(self) -> list[str]:
        classes = [self.classes_[-1]] if self.coefficients_.shape[0] == 1 else list(self.classes_)
        return [f"coefficient_class_{c}" for c in classes]

    def selection_report(self, feature_names: Sequence[str]) -> pd.DataFrame:
        """Ranked table of every encoded feature (legacy columns and sort)."""
        check_is_fitted(self, "support_")
        names = np.asarray(feature_names, dtype=object)
        if names.shape != (self.n_features_in_,):
            raise ValueError("feature_names must match the fitted feature count.")
        order = np.argsort(-self.feature_importances_, kind="stable")
        ranks = np.empty(self.n_features_in_, dtype=np.int64)
        ranks[order] = np.arange(1, self.n_features_in_ + 1)
        report = pd.DataFrame({
            "feature_name": names.astype(str),
            "feature_group": [feature_group(n) for n in names],
            "selected": self.support_,
            "lasso_importance": self.feature_importances_,
            "lasso_rank": ranks,
            "minimum_feature_fallback_used": self.minimum_feature_fallback_used_,
        })
        for row, column in enumerate(self.coefficient_columns()):
            report[column] = self.coefficients_[row]
        return report.sort_values(["selected", "lasso_importance", "feature_name"], ascending=[False, False, True],
                                  kind="stable").reset_index(drop=True)

    @property
    def output_dimension(self) -> int:
        check_is_fitted(self, "support_")
        return int(self.support_.sum())

    def __getstate__(self):
        state = self.__dict__.copy()
        # subfold preprocessors are audit-only; do not persist them with the selector
        if state.get("cv_details_"):
            state["cv_details_"] = [{k: v for k, v in d.items() if k not in ("preprocessor", "train_index")}
                                    for d in state["cv_details_"]]
        return state


# ---------------------------------------------------------------------- stability / charts
STABILITY_COLUMNS = ["feature_name", "feature_group", "feature_occurrence_count", "selection_count",
                     "mean_lasso_importance", "max_lasso_importance", "mean_lasso_rank", "selection_frequency"]


def lasso_stability(reports: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Cross-fold stability table (legacy ``n_fold_cv_run.py:676-707``).

    Each report needs a ``fold`` column (added by :class:`TabularBranch`).  The frequency
    denominator is the number of reports (folds), including folds where a rare
    one-hot level was absent.
    """
    if not reports:
        raise ValueError("no LASSO reports to aggregate")
    combined = pd.concat(list(reports), ignore_index=True)
    if "fold" not in combined:
        raise ValueError("LASSO reports need a 'fold' column")
    combined["selected"] = combined["selected"].astype(bool)
    return (combined.groupby(["feature_name", "feature_group"], as_index=False)
            .agg(feature_occurrence_count=("fold", "nunique"), selection_count=("selected", "sum"),
                 mean_lasso_importance=("lasso_importance", "mean"), max_lasso_importance=("lasso_importance", "max"),
                 mean_lasso_rank=("lasso_rank", "mean"))
            .assign(selection_frequency=lambda r: r["selection_count"] / len(reports))
            .sort_values(["selection_frequency", "mean_lasso_importance", "feature_name"],
                         ascending=[False, False, True], kind="stable")
            .reset_index(drop=True))[STABILITY_COLUMNS]


def save_lasso_feature_importance_chart(report: pd.DataFrame, output_path: str | Path, title: str,
                                        importance_column: str = "lasso_importance", top_n: int = 30,
                                        frequency_column: str | None = None) -> Path:
    """Horizontal bar chart of selected features coloured by family (legacy layout)."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    required = {"feature_name", "feature_group", importance_column} | ({frequency_column} if frequency_column else set())
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"LASSO chart report is missing columns: {sorted(missing)}")
    plotted = report.copy()
    if "selected" in plotted:
        plotted = plotted.loc[plotted["selected"].astype(bool)]
    if "selection_count" in plotted:
        plotted = plotted.loc[plotted["selection_count"] > 0]
    plotted = plotted.loc[np.isfinite(plotted[importance_column].to_numpy(dtype=np.float64))]
    if plotted.empty:
        raise ValueError("LASSO chart report contains no selected finite features.")
    plotted = (plotted.sort_values([importance_column, "feature_name"], ascending=[False, True], kind="stable")
               .head(top_n).sort_values(importance_column, ascending=True, kind="stable"))
    names = [str(n).split("__", 1)[-1] for n in plotted["feature_name"]]
    colors = [FEATURE_GROUP_COLORS.get(g, "#7F7F7F") for g in plotted["feature_group"]]
    fig, ax = plt.subplots(figsize=(12, min(15.0, max(4.5, 0.34 * len(plotted) + 2.0))))
    bars = ax.barh(np.arange(len(plotted)), plotted[importance_column].to_numpy(dtype=np.float64), color=colors)
    ax.set_yticks(np.arange(len(plotted)), labels=names)
    ax.set_xlabel("Absolute LASSO coefficient" if "lasso" in importance_column else importance_column)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    if frequency_column is not None:
        for bar, freq in zip(bars, plotted[frequency_column].to_numpy(dtype=np.float64)):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  selected in {freq:.0%} of folds",
                    va="center", fontsize=8)
    present = [g for g in FEATURE_GROUP_COLORS if g in set(plotted["feature_group"])]
    ax.legend(handles=[Patch(color=FEATURE_GROUP_COLORS[g], label=g) for g in present], title="Feature family",
              loc="lower right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_lasso_stability(reports: Sequence[pd.DataFrame], output_dir: str | Path, top_n: int = 30) -> dict[str, Path]:
    """Write ``lasso_feature_selection_by_fold.csv``, ``lasso_feature_stability.csv`` and ``.png``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"by_fold": output_dir / "lasso_feature_selection_by_fold.csv",
             "stability": output_dir / "lasso_feature_stability.csv",
             "chart": output_dir / "lasso_feature_stability.png"}
    pd.concat(list(reports), ignore_index=True).to_csv(paths["by_fold"], index=False)
    stability = lasso_stability(reports)
    stability.to_csv(paths["stability"], index=False)
    save_lasso_feature_importance_chart(stability, paths["chart"], title="Cross-fold LASSO feature importance and stability",
                                        importance_column="mean_lasso_importance", top_n=top_n,
                                        frequency_column="selection_frequency")
    return paths


def clone_preprocessor_factory(template) -> Callable[[], TransformerMixin]:
    return lambda: clone(template)
