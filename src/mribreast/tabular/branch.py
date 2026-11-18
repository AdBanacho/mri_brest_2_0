"""``TabularBranch`` — ``contracts.BranchTrainer`` for the tabular branch (legacy fold tabular path).

Per outer fold (legacy ``n_fold_cv_run.py:349-393``): fit the preprocessor on
``FoldData.fit`` only, compute balanced sample weights x ``positive_boost``,
optionally fit the LASSO selector (patient-grouped inner CV), fit XGBoost/MLP with
the same weights.  ``FoldData.selection`` is not used (legacy tabular ignores the
calibration rows).  Artifacts are written to ``FoldData.output_dir``.
"""

from __future__ import annotations

import importlib.metadata as md
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from mribreast import MANIFEST_SCHEMA_VERSION
from mribreast.artifacts.core import write_json_atomic, write_text_atomic, read_json
from mribreast.contracts import LABEL, PATIENT_ID, FoldData, check_probabilities
from mribreast.data.predictor_policy import PredictorPolicy, policy_from_cfg
from mribreast.tabular.lasso import (
    FEATURE_GROUP_COLORS, LassoFeatureSelector, feature_group, save_lasso_feature_importance_chart,
)
from mribreast.tabular.models import (
    GroupedEarlyStoppingMLP, balanced_sample_weights, make_tabular_model, require_mlp_sample_weight,
)
from mribreast.tabular.preprocessing import TabularPreprocessor

BRANCH = "tabular"
BRANCH_JSON = "tabular_branch.json"
FILES = {
    "preprocessor": "preprocessor.joblib",
    "selector": "selector.joblib",
    "model": "model.joblib",
    "feature_names": "feature_names.txt",
    "selected_feature_names": "selected_feature_names.txt",
    "coefficients": "coefficients.csv",
    "lasso_report": "lasso_feature_selection.csv",
    "lasso_chart": "lasso_feature_importance.png",
    "model_importance": "model_feature_importance.csv",
    "model_importance_chart": "model_feature_importance.png",
}


def _versions() -> dict[str, str | None]:
    out = {}
    for name in ("scikit-learn", "xgboost", "numpy", "pandas", "joblib"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def aligned_predict_proba(model, features: np.ndarray, num_classes: int) -> np.ndarray:
    """``predict_proba`` re-indexed to integer classes 0..C-1 (legacy ``xgboost_fusion.aligned_predict_proba``)."""
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    aligned = np.zeros((len(features), num_classes), dtype=np.float64)
    for source_index, class_label in enumerate(model.classes_):
        index = int(class_label)
        if not 0 <= index < num_classes:
            raise ValueError(f"Unexpected tabular class label: {class_label}")
        aligned[:, index] = probabilities[:, source_index]
    sums = aligned.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Tabular model produced a row without class probability mass.")
    return aligned / sums


@dataclass
class FittedTabular:
    """``contracts.FittedPredictor`` for one fold."""

    preprocessor: TabularPreprocessor
    model: Any
    num_classes: int
    selector: LassoFeatureSelector | None = None
    info: dict[str, Any] = field(default_factory=dict)
    lasso_report: pd.DataFrame | None = None
    branch: str = BRANCH

    @property
    def feature_names(self) -> list[str]:
        return self.preprocessor.get_feature_names_out()

    @property
    def selected_feature_names(self) -> list[str]:
        names = self.feature_names
        return self.selector.get_feature_names_out(names) if self.selector is not None else names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.preprocessor.transform(frame)
        return self.selector.transform(features) if self.selector is not None else features

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if len(frame) == 0:
            return np.zeros((0, self.num_classes))
        probs = aligned_predict_proba(self.model, self.transform(frame), self.num_classes)
        check_probabilities(probs, self.num_classes)
        return probs

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, directory / FILES["preprocessor"])
        joblib.dump(self.model, directory / FILES["model"])
        write_text_atomic(directory / FILES["feature_names"], "\n".join(self.feature_names) + "\n")
        write_text_atomic(directory / FILES["selected_feature_names"], "\n".join(self.selected_feature_names) + "\n")
        files = [FILES[k] for k in ("preprocessor", "model", "feature_names", "selected_feature_names")]
        top_n = int(self.info.get("lasso_plot_top_n", 30))
        fold = self.info.get("fold")
        if self.selector is not None:
            joblib.dump(self.selector, directory / FILES["selector"])
            report = self.lasso_report if self.lasso_report is not None else self.selector.selection_report(self.feature_names)
            report.to_csv(directory / FILES["lasso_report"], index=False)
            coef_cols = ["feature_name", "feature_group", "selected", *self.selector.coefficient_columns()]
            report[coef_cols].to_csv(directory / FILES["coefficients"], index=False)
            save_lasso_feature_importance_chart(report, directory / FILES["lasso_chart"],
                                                title=f"Fold {fold} selected LASSO feature importance", top_n=top_n)
            files += [FILES[k] for k in ("selector", "lasso_report", "coefficients", "lasso_chart")]
        importances = getattr(self.model, "feature_importances_", None)
        if importances is not None:
            table = pd.DataFrame({"feature_name": self.selected_feature_names,
                                  "feature_group": [feature_group(n) for n in self.selected_feature_names],
                                  "model_importance": np.asarray(importances, dtype=float)})
            table = table.sort_values(["model_importance", "feature_name"], ascending=[False, True], kind="stable")
            table.to_csv(directory / FILES["model_importance"], index=False)
            files.append(FILES["model_importance"])
            if (table["model_importance"] > 0).any():
                save_lasso_feature_importance_chart(table.loc[table["model_importance"] > 0],
                                                    directory / FILES["model_importance_chart"],
                                                    title=f"Fold {fold} {self.info.get('feature_model')} feature importance",
                                                    importance_column="model_importance", top_n=top_n)
                files.append(FILES["model_importance_chart"])
        info = {**self.info, "files": files, "num_classes": self.num_classes, "branch": BRANCH,
                "schema_version": MANIFEST_SCHEMA_VERSION}
        write_json_atomic(directory / BRANCH_JSON, info)


class TabularBranch:
    """``contracts.BranchTrainer`` (branch="tabular")."""

    branch = BRANCH

    def __init__(self, cfg: Mapping[str, Any], continuous_columns: Sequence[str], categorical_columns: Sequence[str],
                 num_classes: int, policy: PredictorPolicy | None = None) -> None:
        self.cfg = dict(cfg)
        self.continuous_columns = list(continuous_columns)
        self.categorical_columns = list(categorical_columns)
        self.num_classes = int(num_classes)
        self.policy = policy or policy_from_cfg(self.cfg)
        self.policy_groups = self.policy.check([*self.continuous_columns, *self.categorical_columns],
                                               prediction_time=self.cfg.get("prediction_time", "mri"),
                                               include_sensitive=bool(self.cfg.get("include_sensitive", False)))
        if self.cfg.get("feature_model", "xgboost") == "mlp":
            require_mlp_sample_weight()

    def _seed(self, data: FoldData) -> int:
        # legacy: every fold uses the global --seed for XGBoost/MLP/LASSO random_state
        return int(self.cfg.get("seed", 42)) if self.cfg.get("profile", "duke_legacy_3d") == "duke_legacy_3d" else int(data.seed)

    def make_selector(self, seed: int) -> LassoFeatureSelector | None:
        if self.cfg.get("feature_selector", "none") == "none":
            return None
        return LassoFeatureSelector(cv_folds=int(self.cfg.get("lasso_cv_folds", 5)), cs=int(self.cfg.get("lasso_cs", 20)),
                                    max_iter=int(self.cfg.get("lasso_max_iter", 5000)),
                                    tolerance=float(self.cfg.get("lasso_tolerance", 1e-4)),
                                    min_features=int(self.cfg.get("lasso_min_features", 1)),
                                    n_jobs=int(self.cfg.get("lasso_n_jobs", 8)), random_state=seed)

    def fit(self, data: FoldData) -> FittedTabular:
        frame = data.fit.reset_index(drop=True)
        labels = frame[LABEL].to_numpy()
        if not np.array_equal(labels, labels.astype(np.int64)):
            raise ValueError("labels must be integers 0..C-1")
        labels = labels.astype(np.int64)
        present = np.unique(labels)
        if not np.array_equal(present, np.arange(self.num_classes)):
            raise ValueError(f"fit rows of fold {data.fold} miss classes: present {present.tolist()}, "
                             f"expected 0..{self.num_classes - 1}")
        groups = frame[PATIENT_ID].astype(str).to_numpy()
        seed = self._seed(data)
        preprocessor = TabularPreprocessor(self.continuous_columns, self.categorical_columns).fit(frame)
        features = preprocessor.transform(frame)
        weights = balanced_sample_weights(labels, float(self.cfg.get("positive_boost", 1.0)),
                                          int(self.cfg.get("positive_class", 1)), self.num_classes)
        selector = self.make_selector(seed)
        mode = self.cfg.get("selector_preprocessing", "outer")
        if selector is not None:
            if mode == "fold_local":
                cont, cat = self.continuous_columns, self.categorical_columns
                selector.fit(features, labels, sample_weight=weights, groups=groups,
                             raw=frame[[*cont, *cat]], preprocessor_factory=lambda: TabularPreprocessor(cont, cat))
            elif mode == "outer":
                selector.fit(features, labels, sample_weight=weights, groups=groups)
            else:
                raise ValueError(f"unknown selector_preprocessing {mode}")
            model_input = selector.transform(features)
        else:
            model_input = features
        model = make_tabular_model(self.cfg, self.num_classes, seed)
        if isinstance(model, GroupedEarlyStoppingMLP):
            model.fit(model_input, labels, sample_weight=weights, groups=groups)
        else:
            model.fit(model_input, labels, sample_weight=weights)

        info: dict[str, Any] = {
            "fold": int(data.fold), "seed": seed, "fold_seed": int(data.seed),
            "feature_model": self.cfg.get("feature_model", "xgboost"),
            "feature_selector": self.cfg.get("feature_selector", "none"),
            "selector_preprocessing": mode if selector is not None else None,
            "mlp_early_stopping": self.cfg.get("mlp_early_stopping", "internal") if self.cfg.get("feature_model") == "mlp" else None,
            "continuous_columns": self.continuous_columns, "categorical_columns": self.categorical_columns,
            "predictor_policy": self.policy.describe(), "predictor_policy_groups": self.policy_groups,
            "n_fit_rows": int(len(frame)), "n_fit_patients": int(pd.Series(groups).nunique()),
            "class_counts": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "positive_boost": float(self.cfg.get("positive_boost", 1.0)),
            "sample_weight_by_class": {int(c): float(weights[labels == c][0]) for c in present},
            "preprocessed_dimension": int(features.shape[1]), "model_input_dimension": int(model_input.shape[1]),
            "model_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                             for k, v in model.get_params().items()},
            "lasso_plot_top_n": int(self.cfg.get("lasso_plot_top_n", 30)),
            "selection_rows_used": False, "library_versions": _versions(),
        }
        report = None
        if selector is not None:
            info.update(effective_cv_folds=int(selector.effective_cv_folds_), best_c=selector.best_c_,
                        minimum_feature_fallback_used=bool(selector.minimum_feature_fallback_used_),
                        n_selected=int(selector.output_dimension))
            report = selector.selection_report(preprocessor.get_feature_names_out())
            report.insert(0, "fold", int(data.fold))
        if isinstance(model, GroupedEarlyStoppingMLP):
            info.update(mlp_best_epoch=int(model.best_epoch_), mlp_epochs_run=int(model.n_epochs_run_),
                        mlp_holdout_patients=int(len(model.holdout_groups_)))
        elif hasattr(model, "n_iter_") and info["mlp_early_stopping"] is not None:
            info.update(mlp_epochs_run=int(model.n_iter_))
        fitted = FittedTabular(preprocessor, model, self.num_classes, selector, info, report)
        fitted.save(Path(data.output_dir))
        return fitted

    def load(self, directory: Path, fold: int) -> FittedTabular:
        directory = Path(directory)
        info = read_json(directory / BRANCH_JSON)
        if str(info.get("schema_version", "")).split(".")[0] != str(MANIFEST_SCHEMA_VERSION).split(".")[0]:
            raise ValueError(f"unsupported tabular artifact schema {info.get('schema_version')}")
        if info.get("branch") != BRANCH or int(info.get("fold", -1)) != int(fold):
            raise ValueError(f"{directory} holds branch={info.get('branch')} fold={info.get('fold')}, expected tabular fold {fold}")
        if int(info["num_classes"]) != self.num_classes:
            raise ValueError("saved tabular model has a different number of classes")
        if info.get("continuous_columns") != self.continuous_columns or info.get("categorical_columns") != self.categorical_columns:
            raise ValueError("saved tabular model was fitted on different predictor columns")
        selector_path = directory / FILES["selector"]
        selector = joblib.load(selector_path) if selector_path.is_file() else None
        report_path = directory / FILES["lasso_report"]
        report = pd.read_csv(report_path) if report_path.is_file() else None
        return FittedTabular(joblib.load(directory / FILES["preprocessor"]), joblib.load(directory / FILES["model"]),
                             self.num_classes, selector, info, report)


def build_tabular_branch(cfg: Mapping[str, Any], *, continuous_columns: Sequence[str], categorical_columns: Sequence[str],
                         num_classes: int) -> TabularBranch:
    return TabularBranch(cfg, continuous_columns, categorical_columns, num_classes)


__all__ = ["TabularBranch", "FittedTabular", "build_tabular_branch", "aligned_predict_proba", "FEATURE_GROUP_COLORS"]
