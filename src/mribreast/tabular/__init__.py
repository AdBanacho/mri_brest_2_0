"""mribreast.tabular — preprocessing, LASSO selection, XGBoost/MLP factories, ``TabularBranch``."""

from mribreast.tabular.branch import FittedTabular, TabularBranch, aligned_predict_proba, build_tabular_branch
from mribreast.tabular.lasso import (
    LassoFeatureSelector, feature_group, lasso_stability, save_lasso_feature_importance_chart, save_lasso_stability,
)
from mribreast.tabular.models import (
    GroupedEarlyStoppingMLP, balanced_sample_weights, make_mlp, make_tabular_model, make_xgboost,
)
from mribreast.tabular.preprocessing import TabularPreprocessor

__all__ = [
    "FittedTabular", "TabularBranch", "aligned_predict_proba", "build_tabular_branch", "LassoFeatureSelector",
    "feature_group", "lasso_stability", "save_lasso_feature_importance_chart", "save_lasso_stability",
    "GroupedEarlyStoppingMLP", "balanced_sample_weights", "make_mlp", "make_tabular_model", "make_xgboost",
    "TabularPreprocessor",
]
