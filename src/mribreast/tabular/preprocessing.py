"""Fold-local tabular preprocessing (legacy ``ClinicalFeaturePreprocessor``).

continuous : SimpleImputer(median, keep_empty_features) -> StandardScaler
categorical: SimpleImputer(most_frequent, keep_empty_features) -> OneHotEncoder(handle_unknown="ignore")
Feature names after encoding come from ``ColumnTransformer.get_feature_names_out``
(``continuous__<col>``, ``categorical__<col>_<level>``) exactly like legacy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _normalise_categorical(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """pd.NA/None -> np.nan and object dtype (sklearn's imputer rejects pd.NA comparisons)."""
    if not columns:
        return frame
    out = frame.copy()
    for column in columns:
        values = out[column].astype(object)
        out[column] = values.where(pd.notna(values), np.nan)
    return out


class TabularPreprocessor(TransformerMixin, BaseEstimator):
    """Fit numeric/categorical preprocessing on one training subset only."""

    def __init__(self, continuous_columns: Sequence[str] = (), categorical_columns: Sequence[str] = ()):
        self.continuous_columns = tuple(continuous_columns)
        self.categorical_columns = tuple(categorical_columns)

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.continuous_columns, *self.categorical_columns)

    def _validate(self, data: pd.DataFrame) -> None:
        overlap = set(self.continuous_columns) & set(self.categorical_columns)
        if overlap:
            raise ValueError(f"Columns cannot be both continuous and categorical: {sorted(overlap)}")
        if not self.columns:
            raise ValueError("At least one tabular feature column is required.")
        missing = set(self.columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing tabular feature columns: {sorted(missing)}")

    def fit(self, data: pd.DataFrame, y=None):
        self._validate(data)
        transformers = []
        if self.continuous_columns:
            transformers.append(("continuous", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler())]), list(self.continuous_columns)))
        if self.categorical_columns:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), list(self.categorical_columns)))
        self.transformer_ = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
        frame = _normalise_categorical(data[list(self.columns)], self.categorical_columns)
        for column in self.continuous_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        self.transformer_.fit(frame)
        self.n_fit_rows_ = int(len(data))
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "transformer_"):
            raise RuntimeError("Fit TabularPreprocessor before transform().")
        self._validate(data)
        frame = _normalise_categorical(data[list(self.columns)], self.categorical_columns)
        for column in self.continuous_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        out = np.asarray(self.transformer_.transform(frame), dtype=np.float32)
        if not np.all(np.isfinite(out)):
            raise ValueError("Tabular preprocessing produced non-finite values.")
        return out

    def get_feature_names_out(self, input_features=None) -> list[str]:
        if not hasattr(self, "transformer_"):
            raise RuntimeError("Fit TabularPreprocessor before requesting names.")
        return self.transformer_.get_feature_names_out().tolist()

    @property
    def output_dimension(self) -> int:
        return len(self.get_feature_names_out())
