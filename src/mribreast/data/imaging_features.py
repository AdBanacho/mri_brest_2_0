"""Duke precomputed imaging features (``Imaging_Features.xlsx``) grouped by family.

Port of legacy ``mriBreastDuke/dataLoaders/precomputed_imaging_features.py``:
identical family classification (morphology exact names first, then
heterogeneity markers, then kinetic markers), ``<family>_<source column>``
prefixing, duplicate / missing patient-id validation, numeric-only check, and a
left many-to-one merge by patient id with an optional complete-match requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mribreast.contracts import PATIENT_ID

IMAGING_FEATURE_GROUPS = ("kinetic", "morphology", "heterogeneity")
IMAGING_SHEET = "Imaging Features"

MORPHOLOGY_COLUMNS = frozenset({
    "TumorMajorAxisLength_mm", "BEVR_Tumor", "BEDR1_Tumor", "BEDR2_Tumor", "MF_Tumor", "ASD_Tumor",
    "Volume_cu_mm_Tumor", "Median_solidity_Tumor", "Median_Elongation_Tumor", "Median_Euler_No_Tumor",
    "BreastVol", "tissueVol_T1", "tissueVol_PostCon", "breastDensity_T1", "breastDensity_PostCon",
})

HETEROGENEITY_MARKERS = (
    "autocorrelation", "contrast", "correlation1", "correlation2", "cluster_prominence", "cluster_shade",
    "dissimilarity", "energy", "entropy", "homogeneity1", "homogeneity2", "max_probability", "sum_of_squares",
    "sum_avg", "sum_average", "sum_variance", "sum_entropy", "diff_entropy", "difference_entropy",
    "inf_mea_of_corr", "information_measure_correlation", "inv_diff", "inverse_difference", "globalmoransi",
    "enhancementcluster", "dft_", "dhog_", "dlbp_", "variance_of_uptake", "change_in_variance_of_uptake",
    "margin_gradient", "variance_of_margin_gradient", "variance_of_rgh", "_map_mean", "_map_std_dev",
    "_map_skewness", "_map_kurtosis",
)

KINETIC_MARKERS = (
    "f1_dt_", "max_enhancement", "time_to_peak", "uptake_rate", "washout_rate", "ratio_tissue_vol_enhancing",
    "maximum_variance_of_enhancement", "peak_location_of_enhancement", "enhancement_variance_", "grouping_based_",
    "ser_total", "ser_partial", "ser_washout", "peak_ser", "peak_pe",
)


class ImagingFeatureError(ValueError):
    pass


def classify_imaging_feature(column: str) -> str:
    """Map one workbook column to a non-overlapping family (legacy order of checks)."""
    column = str(column)
    lowered = column.lower()
    if column in MORPHOLOGY_COLUMNS:
        return "morphology"
    if any(marker in lowered for marker in HETEROGENEITY_MARKERS):
        return "heterogeneity"
    if any(marker in lowered for marker in KINETIC_MARKERS):
        return "kinetic"
    raise ImagingFeatureError(
        f"Imaging feature '{column}' is not assigned to morphology, kinetic, or heterogeneity. "
        "Update the feature-family mapping explicitly.")


def classify_imaging_feature_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {group: [] for group in IMAGING_FEATURE_GROUPS}
    for column in columns:
        grouped[classify_imaging_feature(column)].append(column)
    return grouped


def read_imaging_feature_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=IMAGING_SHEET)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ImagingFeatureError("Precomputed imaging features must be supplied as .xlsx, .xls, or .csv.")


def load_imaging_features(path: str | Path, selected_groups: Sequence[str], source_patient_id_column: str = "Patient ID",
                          table: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Return (frame[patient_id + prefixed features], feature columns, family sizes)."""
    selected_groups = tuple(dict.fromkeys(selected_groups))
    unknown = set(selected_groups).difference(IMAGING_FEATURE_GROUPS)
    if unknown:
        raise ImagingFeatureError(f"Unknown imaging feature groups: {sorted(unknown)}")
    if not selected_groups:
        raise ImagingFeatureError("At least one imaging feature group must be selected.")
    table = read_imaging_feature_table(path) if table is None else table
    if source_patient_id_column not in table.columns:
        raise ImagingFeatureError(f"Imaging feature table is missing '{source_patient_id_column}'.")
    if table[source_patient_id_column].isna().any():
        raise ImagingFeatureError("Imaging feature table contains a missing patient ID.")
    if table[source_patient_id_column].duplicated().any():
        dups = table.loc[table[source_patient_id_column].duplicated(keep=False), source_patient_id_column]
        raise ImagingFeatureError(f"Imaging feature table contains duplicate patient IDs: {sorted(map(str, set(dups)))[:5]}")
    source_columns = [c for c in table.columns if c != source_patient_id_column]
    non_numeric = [c for c in source_columns if not pd.api.types.is_numeric_dtype(table[c])]
    if non_numeric:
        raise ImagingFeatureError(f"Imaging feature columns must be numeric; received non-numeric columns: {non_numeric[:10]}")
    grouped = classify_imaging_feature_columns(source_columns)
    selected_source = [c for g in selected_groups for c in grouped[g]]
    renamed = {c: f"{classify_imaging_feature(c)}_{c}" for c in selected_source}
    out = table[[source_patient_id_column, *selected_source]].rename(columns={source_patient_id_column: PATIENT_ID, **renamed})
    out[PATIENT_ID] = out[PATIENT_ID].astype(str)
    return out, [renamed[c] for c in selected_source], {g: len(grouped[g]) for g in selected_groups}


def merge_imaging_features(studies: pd.DataFrame, path: str | Path, selected_groups: Sequence[str], *,
                           source_patient_id_column: str = "Patient ID", require_complete_match: bool = True,
                           table: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Left many-to-one merge by patient id (legacy ``merge_precomputed_imaging_features``).

    Returns (merged frame with ``imaging_features_matched`` bool column, feature columns, report).
    """
    if PATIENT_ID not in studies.columns:
        raise ImagingFeatureError(f"Studies table is missing '{PATIENT_ID}'.")
    imaging, columns, sizes = load_imaging_features(path, selected_groups, source_patient_id_column, table=table)
    prepared = studies.copy()
    prepared[PATIENT_ID] = prepared[PATIENT_ID].astype(str)
    merged = prepared.merge(imaging, on=PATIENT_ID, how="left", validate="many_to_one")
    matched_mask = merged[columns].notna().any(axis=1) if columns else pd.Series(False, index=merged.index)
    matched = int(matched_mask.sum())
    if matched == 0:
        raise ImagingFeatureError("Imaging_Features did not match any studies by patient ID.")
    if require_complete_match and matched != len(merged):
        unmatched = merged.loc[~matched_mask, PATIENT_ID].astype(str)
        raise ImagingFeatureError(f"Imaging features are missing for {len(unmatched)} studies, including: "
                                  f"{', '.join(unmatched.head(5))} (set allow_missing_imaging_features=true to impute)")
    merged["imaging_features_matched"] = matched_mask.to_numpy()
    report = {"family_sizes": sizes, "selected": len(columns), "matched_studies": matched, "studies": len(merged)}
    return merged, columns, report
