"""Optional feature group ``density`` from ``Breast_Radiologist_Density_Assessments.xlsx``.

Real schema (Duke release, verified on the workbook shipped with the legacy repo):

* sheet ``Assessments``; 50 data rows, one per patient, no duplicates;
* ``Subject_ID`` — Duke patient id (``Breast_MRI_NNN``), joins ``patient_id``;
* ``Radiologist A``, ``Radiologist B``, ``Radiologist C`` — BI-RADS breast density
  category letters ``a`` (almost entirely fat), ``b`` (scattered fibroglandular),
  ``c`` (heterogeneous fibroglandular), ``d`` (extreme fibroglandular);
* a fifth, empty column whose header is the category legend (ignored).

Raters disagree on 26/50 patients.  Only 18 of the 261 patients with a numeric
Oncotype score have an assessment, so the group is **off by default** and its
coverage is reported in cohort notes; patients without an assessment get NaN
(imputed fold-locally like any other predictor).

Derived predictors (both allowed at ``prediction_time=mri`` by policy v1 — the
assessments were read on the same pre-treatment MRI):

* ``density_consensus`` (categorical a..d): majority vote of the raters; with no
  majority the median ordinal category (lower median for even counts);
* ``density_ordinal_mean`` (continuous): mean of a=1, b=2, c=3, d=4 over raters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mribreast.contracts import PATIENT_ID

DENSITY_SHEET = "Assessments"
DENSITY_ID_COLUMN = "Subject_ID"
DENSITY_RATER_PREFIX = "Radiologist"
DENSITY_CATEGORIES = ("a", "b", "c", "d")
DENSITY_CATEGORICAL_COLUMNS = ("density_consensus",)
DENSITY_CONTINUOUS_COLUMNS = ("density_ordinal_mean",)


class DensityError(ValueError):
    pass


def read_density_assessments(path: str | Path) -> pd.DataFrame:
    """Return ``patient_id`` + one column per rater with normalised category letters (or NaN)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        sheets = pd.ExcelFile(path).sheet_names
        raw = pd.read_excel(path, sheet_name=DENSITY_SHEET if DENSITY_SHEET in sheets else 0)
    if DENSITY_ID_COLUMN not in raw.columns:
        raise DensityError(f"density workbook lacks '{DENSITY_ID_COLUMN}' (columns: {list(raw.columns)[:6]})")
    raters = [c for c in raw.columns if str(c).startswith(DENSITY_RATER_PREFIX)]
    if not raters:
        raise DensityError("density workbook has no 'Radiologist *' columns")
    out = pd.DataFrame({PATIENT_ID: raw[DENSITY_ID_COLUMN].astype("string").str.strip()})
    if out[PATIENT_ID].isna().any():
        raise DensityError("density workbook contains a missing Subject_ID")
    if out[PATIENT_ID].duplicated().any():
        raise DensityError(f"duplicate Subject_ID in density workbook: {sorted(set(out[PATIENT_ID][out[PATIENT_ID].duplicated()]))[:5]}")
    for rater in raters:
        values = raw[rater].map(lambda v: str(v).strip().lower() if pd.notna(v) and str(v).strip() else np.nan)
        bad = sorted(set(values.dropna()) - set(DENSITY_CATEGORIES))
        if bad:
            raise DensityError(f"unknown density categories in '{rater}': {bad}")
        out[str(rater)] = values.to_numpy()
    out[PATIENT_ID] = out[PATIENT_ID].astype(str)
    return out


def _consensus(row: pd.Series) -> Any:
    votes = [v for v in row if isinstance(v, str)]
    if not votes:
        return np.nan
    counts = pd.Series(votes).value_counts()
    if counts.iloc[0] > len(votes) / 2:
        return counts.index[0]
    ordinals = sorted(DENSITY_CATEGORIES.index(v) for v in votes)
    return DENSITY_CATEGORIES[ordinals[(len(ordinals) - 1) // 2]]


def density_features(assessments: pd.DataFrame) -> pd.DataFrame:
    raters = [c for c in assessments.columns if c != PATIENT_ID]
    ordinal = assessments[raters].apply(lambda col: col.map(lambda v: DENSITY_CATEGORIES.index(v) + 1 if isinstance(v, str) else np.nan))
    return pd.DataFrame({
        PATIENT_ID: assessments[PATIENT_ID].to_numpy(),
        "density_consensus": assessments[raters].apply(_consensus, axis=1).to_numpy(dtype=object),
        "density_ordinal_mean": ordinal.mean(axis=1, skipna=True).to_numpy(dtype=float),
    })


def merge_density(studies: pd.DataFrame, path: str | Path) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    """Left many-to-one merge; returns (frame, continuous, categorical, coverage report)."""
    features = density_features(read_density_assessments(path))
    merged = studies.merge(features, on=PATIENT_ID, how="left", validate="many_to_one")
    matched = merged["density_ordinal_mean"].notna()
    report = {"assessed_patients_in_file": int(len(features)),
              "cohort_patients": int(studies[PATIENT_ID].nunique()),
              "cohort_patients_with_density": int(merged.loc[matched, PATIENT_ID].nunique()),
              "studies_with_density": int(matched.sum()), "studies": int(len(merged))}
    report["coverage"] = report["studies_with_density"] / max(1, report["studies"])
    merged["density_consensus"] = merged["density_consensus"].astype(object).where(merged["density_consensus"].notna(), np.nan)
    return merged, list(DENSITY_CONTINUOUS_COLUMNS), list(DENSITY_CATEGORICAL_COLUMNS), report
