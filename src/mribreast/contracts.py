"""Typed public contracts shared by data, imaging, tabular, protocol and reporting.

Rules:
* Column names below are the ONLY names that cross module boundaries.
* Probabilities are always ``np.ndarray`` of shape ``[n, num_classes]`` in the
  fixed class order ``0..C-1`` and indexed by ``study_id`` row order of the
  DataFrame passed in.
* The CV loop (``protocol``) depends on these Protocols only; swapping a model,
  selector or mask provider never edits the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Cohort table (one row per STUDY = one MRI examination)
# --------------------------------------------------------------------------
PATIENT_ID = "patient_id"      # grouping unit for every split
STUDY_ID = "study_id"          # StudyInstanceUID (or synthetic study key)
LABEL = "label"                # int 0..C-1
ONCOTYPE_SCORE = "oncotype_score"  # used ONLY to build LABEL and eligibility; never a predictor
SERIES_UIDS = "series_uids"    # tuple[str, ...] ordered phases (pre, post1, ...)
COHORT_REQUIRED_COLUMNS = (PATIENT_ID, STUDY_ID, LABEL, SERIES_UIDS)

# Image-level key (2D): one row per image/slice
SERIES_ID = "series_id"
SLICE_ID = "slice_id"          # slice index along slice_axis, or frame id for ready 2D images
IMAGE_KEY_COLUMNS = (PATIENT_ID, STUDY_ID, SERIES_ID, SLICE_ID)

# Split manifest (split_manifest.csv): one row per (fold, study)
FOLD = "fold"                  # 1-based outer fold
ROLE = "role"                  # see ROLES
ROLES = ("fit", "selection", "threshold", "calibration", "outer_eval")
# shared:   fit + calibration (calibration = selection = threshold)
# separate: fit + selection + threshold (disjoint patients)
# outer:    fit(=all outer-train) + outer_eval also used for checkpoint selection (historical image-only)


@dataclass(frozen=True)
class LabelDefinition:
    scheme: str                      # "oncotype_binary_18" | "oncotype_3class_18_31"
    cutoffs: tuple[float, ...]       # right-closed upper bounds: (18,) or (18, 31)
    class_names: tuple[str, ...]     # e.g. ("low_<=18", "high_>18")
    positive_class: int = 1

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def encode(self, score: float) -> int:
        for index, cutoff in enumerate(self.cutoffs):
            if score <= cutoff:
                return index
        return len(self.cutoffs)

    def to_json(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "cutoffs": list(self.cutoffs),
                "class_names": list(self.class_names), "positive_class": self.positive_class,
                "rule": "label = number of cutoffs strictly below score (score <= cutoff[i] -> i)"}


def label_definition(scheme: str, positive_class: int = 1) -> LabelDefinition:
    if scheme == "oncotype_binary_18":
        return LabelDefinition(scheme, (18.0,), ("low_<=18", "high_>18"), positive_class)
    if scheme == "oncotype_3class_18_31":
        return LabelDefinition(scheme, (18.0, 31.0), ("low_<=18", "mid_18-31", "high_>31"), positive_class)
    raise ValueError(f"unknown label scheme {scheme}")


# --------------------------------------------------------------------------
# Geometry / masks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """Array geometry of a loaded image: shape in loaded-array axis order, 4x4 affine (voxel->RAS mm)."""
    shape: tuple[int, ...]
    affine: tuple[tuple[float, ...], ...] | None = None   # None for plain 2D images
    spacing: tuple[float, ...] | None = None
    orientation: str | None = None                         # nibabel aff2axcodes, e.g. "LPS"

    def matches(self, other: "Geometry", atol: float = 1e-4) -> bool:
        if tuple(self.shape) != tuple(other.shape):
            return False
        if self.affine is None or other.affine is None:
            return self.affine is None and other.affine is None
        return bool(np.allclose(np.asarray(self.affine), np.asarray(other.affine), atol=atol))


@dataclass(frozen=True)
class ImageKey:
    patient_id: str
    study_id: str
    series_id: str
    slice_id: str          # "" for whole-volume (3D) masks

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.patient_id, self.study_id, self.series_id, self.slice_id)


@dataclass(frozen=True)
class MaskRecord:
    key: ImageKey
    array: np.ndarray                  # integer class codes, same shape as the image grid it refers to
    geometry: Geometry
    class_codes: Mapping[int, str]     # e.g. {0: "background", 1: "lesion"}
    source: str                        # "manual" | "predicted" | "synthetic"
    model_id: str = ""                 # external segmenter id
    model_version: str = ""
    fingerprint: str = ""              # sha256 of mask bytes
    grid_source: str = ""              # e.g. series uid the mask grid was defined on
    training_provenance: str = "unknown"  # unknown | frozen_external | fold_trained:<fold>
    known_overlap: str = ""            # free text: known overlap of segmenter training data with this cohort


@runtime_checkable
class MaskProvider(Protocol):
    """Narrow interface for any mask source (local files, synthetic, external model)."""

    name: str

    def get(self, key: ImageKey, fold: int | None = None) -> MaskRecord | None:
        """Return the mask for ``key`` or ``None`` when missing. ``fold`` selects fold-trained masks."""

    def describe(self) -> dict[str, Any]:
        """Provenance dict written into manifests (model id/version, training data, overlap)."""


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
@runtime_checkable
class CohortSource(Protocol):
    """Builds the study-level cohort table (COHORT_REQUIRED_COLUMNS + tabular columns)."""

    def load(self, cfg: Mapping[str, Any]) -> "CohortTable":
        ...


@dataclass
class CohortTable:
    frame: pd.DataFrame                      # one row per study
    continuous_columns: list[str]
    categorical_columns: list[str]
    label: LabelDefinition
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-study inclusion/exclusion reasons
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Branches / predictors
# --------------------------------------------------------------------------
@dataclass
class FoldData:
    """Everything a branch may see while fitting one outer fold."""
    fold: int
    fit: pd.DataFrame            # rows used to fit parameters / preprocessing
    selection: pd.DataFrame      # rows used for checkpoint / early-stopping selection
    seed: int
    output_dir: Path             # branch-private directory inside the fold directory


@runtime_checkable
class FittedPredictor(Protocol):
    """A fitted branch for one fold. Loading it back must not need training data."""

    branch: str                  # "image" | "tabular"

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Study-level probabilities [len(frame), C] in class order 0..C-1, row-aligned to ``frame``."""

    def save(self, directory: Path) -> None:
        ...


@runtime_checkable
class BranchTrainer(Protocol):
    branch: str

    def fit(self, data: FoldData) -> FittedPredictor:
        ...

    def load(self, directory: Path, fold: int) -> FittedPredictor:
        ...


# --------------------------------------------------------------------------
# Prediction tables written by validation (predictions_*.csv)
# --------------------------------------------------------------------------
PRED_COLUMNS = (FOLD, PATIENT_ID, STUDY_ID, LABEL, "branch", "split")  # + prob_0..prob_{C-1} + "decision"


def prob_columns(num_classes: int) -> list[str]:
    return [f"prob_{k}" for k in range(num_classes)]


def predictions_frame(fold: int, frame: pd.DataFrame, probs: np.ndarray, branch: str, split: str) -> pd.DataFrame:
    if probs.shape[0] != len(frame):
        raise ValueError("probabilities are not row-aligned with the frame")
    out = pd.DataFrame({FOLD: fold, PATIENT_ID: frame[PATIENT_ID].to_numpy(), STUDY_ID: frame[STUDY_ID].to_numpy(),
                        LABEL: frame[LABEL].to_numpy(), "branch": branch, "split": split})
    for k, column in enumerate(prob_columns(probs.shape[1])):
        out[column] = probs[:, k]
    return out


def check_probabilities(probs: np.ndarray, num_classes: int, atol: float = 1e-4) -> None:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise ValueError(f"probabilities must be [n, {num_classes}], got {probs.shape}")
    if not np.all(np.isfinite(probs)) or probs.min() < -atol or probs.max() > 1 + atol:
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=atol):
        raise ValueError("probability rows must sum to 1")


def as_str_tuple(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(v) for v in values)
