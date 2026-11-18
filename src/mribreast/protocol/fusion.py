"""Late fusion: align branch predictions by ``study_id`` and mix probabilities.

``p_fusion = alpha * p_image + (1 - alpha) * p_tabular`` followed by the legacy
row renormalisation (``classificators/xgboost_fusion.py::fuse_probabilities``).
``alpha`` is an evaluation setting only; it never requires refitting.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.contracts import STUDY_ID, check_probabilities, prob_columns


class AlignmentError(ValueError):
    """Branch predictions cannot be aligned (missing / duplicate / extra studies, bad classes)."""


def branch_frame(study_ids: Sequence[str], probs: np.ndarray, num_classes: int, branch: str) -> pd.DataFrame:
    """Wrap a predictor output into ``study_id + prob_0..prob_{C-1}`` after shape/sum checks."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != len(study_ids):
        raise AlignmentError(f"{branch}: predictions have shape {probs.shape}, expected [{len(study_ids)}, {num_classes}]")
    try:
        check_probabilities(probs, num_classes)
    except ValueError as exc:
        raise AlignmentError(f"{branch}: {exc}") from exc
    out = pd.DataFrame({STUDY_ID: [str(s) for s in study_ids]})
    for k, column in enumerate(prob_columns(num_classes)):
        out[column] = probs[:, k]
    return out


def align_to_reference(reference_ids: Sequence[str], predictions: pd.DataFrame, num_classes: int,
                       branch: str) -> np.ndarray:
    """Return ``predictions`` probabilities reordered to ``reference_ids``.

    Rejects missing, duplicated and unexpected studies, class columns that are
    not exactly ``prob_0..prob_{C-1}``, and rows that are not distributions.
    """
    expected_cols = prob_columns(num_classes)
    have_cols = [c for c in predictions.columns if str(c).startswith("prob_")]
    if have_cols != expected_cols:
        raise AlignmentError(f"{branch}: class columns {have_cols} != expected {expected_cols} (fixed class order 0..C-1)")
    ids = predictions[STUDY_ID].astype(str)
    if ids.duplicated().any():
        raise AlignmentError(f"{branch}: duplicate study_id {ids[ids.duplicated()].tolist()[:5]}")
    reference = pd.Index([str(s) for s in reference_ids])
    if reference.duplicated().any():
        raise AlignmentError("reference study ids contain duplicates")
    missing = reference.difference(pd.Index(ids))
    if len(missing):
        raise AlignmentError(f"{branch}: missing predictions for {list(missing)[:5]}")
    extra = pd.Index(ids).difference(reference)
    if len(extra):
        raise AlignmentError(f"{branch}: unexpected studies {list(extra)[:5]}")
    probs = predictions.set_index(ids)[expected_cols].loc[reference].to_numpy(dtype=np.float64)
    try:
        check_probabilities(probs, num_classes)
    except ValueError as exc:
        raise AlignmentError(f"{branch}: {exc}") from exc
    return probs


def fuse_probabilities(image_probabilities, tabular_probabilities, alpha: float = 0.5) -> np.ndarray:
    """Legacy-identical weighted average (``alpha`` = image weight) with row renormalisation."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    image_probabilities = np.asarray(image_probabilities, dtype=np.float64)
    tabular_probabilities = np.asarray(tabular_probabilities, dtype=np.float64)
    if image_probabilities.shape != tabular_probabilities.shape:
        raise ValueError(f"image and tabular probability arrays differ in shape: "
                         f"{image_probabilities.shape} vs {tabular_probabilities.shape}")
    if image_probabilities.ndim != 2 or image_probabilities.shape[1] < 2:
        raise ValueError("Probability arrays must have shape (samples, classes).")
    if not (np.all(np.isfinite(image_probabilities)) and np.all(np.isfinite(tabular_probabilities))):
        raise ValueError("Probability arrays must contain only finite values.")
    fused = alpha * image_probabilities + (1.0 - alpha) * tabular_probabilities
    row_sums = fused.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Fused predictions contain a row without probability mass.")
    return fused / row_sums


def fuse_aligned(reference_ids: Sequence[str], branch_predictions: Mapping[str, pd.DataFrame], num_classes: int,
                 alpha: float) -> dict[str, np.ndarray]:
    """Align every branch to ``reference_ids``; add ``fusion`` when both image and tabular exist."""
    aligned = {branch: align_to_reference(reference_ids, frame, num_classes, branch)
               for branch, frame in branch_predictions.items()}
    if "image" in aligned and "tabular" in aligned:
        aligned["fusion"] = fuse_probabilities(aligned["image"], aligned["tabular"], alpha)
    return aligned
