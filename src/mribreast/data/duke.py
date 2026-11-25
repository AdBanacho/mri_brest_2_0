"""Duke Breast Cancer MRI sources and cohort building.

Data model
----------
* **patient** (``patient_id`` = Duke ``Breast_MRI_NNN``): clinical workbook row, Oncotype score, label.
  Every split groups by patient.
* **study** (``study_id`` = StudyInstanceUID): one MRI examination; one cohort row.
* **series / phase** (``SeriesInstanceUID``): one DCE acquisition; images are stored as
  ``<image_root>/<SeriesInstanceUID>.nii.gz``.  ``series_uids`` is the ordered phase tuple.
* **slice**: index along ``slice_axis`` of a loaded series volume (``contracts.SLICE_ID``);
  the metadata digest has no per-slice rows, only an optional ``Number of Images`` count.

Profiles
--------
``duke_legacy_3d`` reproduces legacy ``read_from_xlmx.get_oncotype_score_for_series_as_studyId_and_label_df``:
patients with a numeric score -> inner join with metadata rows (metadata row order kept) ->
studies with >= 2 *unique* series UIDs -> ``series_uids = first phase_count rows in metadata
file order`` (duplicates included, exactly like ``list(x)[:5]``) -> rows sorted by study UID
(``groupby`` order) -> inner many-to-one join with the clinical table.

**Ordering assumption (legacy):** the NBIA digest lists the series of a study in
acquisition order (pre, post1, ... post4).  Nothing in legacy checks this; it is
recorded in cohort notes.  ``phase_order=explicit`` / profile ``general`` replace
it by parsing the phase index from metadata.

``general`` parses a phase index for every series (``Phase Index`` column when present,
otherwise from ``Series Description``), verifies metadata consistency, and requires
pre-contrast + ``phase_count - 1`` consecutive post-contrast phases; anything that
prevents interpreting pre/post is reported in the audit (study excluded) or raised
(metadata lacking every column needed to interpret phases).

NBIA digest columns (file not distributed; inferred from legacy ``read_from_xlmx.py``):
sheet ``Metadata``, header row 0, required ``Patient ID``, ``Study Instance UID``,
``Series Instance UID``; optional ``Series Description``, ``Phase Index``,
``Series Number``, ``Number of Images``, ``Modality``, ``Study Date``.  CSV with the
same header is accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.contracts import (
    LABEL, ONCOTYPE_SCORE, PATIENT_ID, SERIES_UIDS, STUDY_ID, CohortTable, LabelDefinition, label_definition,
)
from mribreast.data.density import merge_density
from mribreast.data.imaging_features import IMAGING_FEATURE_GROUPS, merge_imaging_features
from mribreast.data.mricore import merge_mricore
from mribreast.data.predictor_policy import check_label_leakage, policy_from_cfg

# ---------------------------------------------------------------------------
# Clinical workbook (Clinical_and_Other_Features.xlsx, sheet "Data", 3 header rows)
# ---------------------------------------------------------------------------
CLINICAL_SHEET = "Data"
CLINICAL_HEADER_ROWS = 3  # feature group, feature name, coding definition
# Excel column letter -> canonical name (legacy read_from_xlmx._CLINICAL_EXCEL_COLUMNS)
CLINICAL_EXCEL_COLUMNS: dict[str, str] = {
    "A": "patient_id",
    "T": "date_of_birth_days",
    "U": "menopause",
    "V": "race_ethnicity",
    "AB": ONCOTYPE_SCORE,
    "AK": "tumor_laterality",
    "AW": "multicentric_multifocal",
    "AX": "contralateral_breast_involvement",
    "AY": "suspicious_lymph_nodes",
    "AZ": "skin_nipple_involvement",
    "BA": "pectoral_chest_involvement",
}
CLINICAL_PREDICTOR_COLUMNS = (
    "age_at_diagnosis_years", "menopause", "tumor_laterality", "multicentric_multifocal",
    "contralateral_breast_involvement", "suspicious_lymph_nodes", "skin_nipple_involvement",
    "pectoral_chest_involvement",
)
CLINICAL_CONTINUOUS_COLUMNS = ("age_at_diagnosis_years",)
CLINICAL_CATEGORICAL_COLUMNS = tuple(c for c in CLINICAL_PREDICTOR_COLUMNS if c not in CLINICAL_CONTINUOUS_COLUMNS)
SENSITIVE_CLINICAL_PREDICTOR_COLUMNS = ("race_ethnicity",)
MISSING_CLINICAL_TOKENS = frozenset({"", "NA", "N/A", "NC", "NP", "X", "NAN", "NONE"})
_NON_NUMERIC_CLINICAL = {"patient_id", "tumor_laterality"}

# ---------------------------------------------------------------------------
# NBIA metadata digest
# ---------------------------------------------------------------------------
METADATA_SHEET = "Metadata"
METADATA_REQUIRED = {"Patient ID": PATIENT_ID, "Study Instance UID": STUDY_ID, "Series Instance UID": "series_uid"}
METADATA_OPTIONAL = {"Series Description": "series_description", "Phase Index": "phase_index",
                     "Series Number": "series_number", "Number of Images": "num_images",
                     "Modality": "modality", "Study Date": "study_date"}
FEATURE_GROUPS = ("clinical", *IMAGING_FEATURE_GROUPS, "density", "mricore")
LEGACY_ORDER_NOTE = ("duke_legacy_3d: series_uids are the first phase_count series of each study in the CURRENT "
                     "metadata-file row order (assumed acquisition order pre, post1..; not verified by legacy)")


class CohortError(ValueError):
    """The inputs cannot produce a valid cohort."""


@dataclass(frozen=True)
class SeriesRecord:
    patient_id: str
    study_id: str
    series_uid: str
    row_index: int                    # 0-based data row in the metadata file
    description: str = ""
    phase_index: int | None = None    # 0 = pre-contrast, k = k-th post-contrast; None = not a DCE phase
    num_slices: int | None = None


@dataclass(frozen=True)
class StudyRecord:
    patient_id: str
    study_id: str
    series: tuple[SeriesRecord, ...]  # selected phases in order

    @property
    def series_uids(self) -> tuple[str, ...]:
        return tuple(s.series_uid for s in self.series)


# ---------------------------------------------------------------------------
def _clean_clinical_value(value: Any) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.upper() in MISSING_CLINICAL_TOKENS:
            return np.nan
    return value


def read_clinical_workbook(path: str | Path) -> pd.DataFrame:
    """Patient-level clinical table: ``patient_id``, cleaned source columns, ``age_at_diagnosis_years``.

    Legacy-identical parsing (``read_useful_clinical_predictors``): sheet ``Data``, three
    header rows skipped, explicit Excel column letters, missing tokens -> NaN, numeric
    coercion of every column except patient id and tumour laterality, age =
    ``-date_of_birth_days / 365.25`` (days counted back from diagnosis).
    """
    letters = ",".join(CLINICAL_EXCEL_COLUMNS)
    data = pd.read_excel(path, sheet_name=CLINICAL_SHEET, header=None, skiprows=CLINICAL_HEADER_ROWS,
                         usecols=letters, names=list(CLINICAL_EXCEL_COLUMNS.values()))
    data = data.apply(lambda column: column.map(_clean_clinical_value))
    data = data.dropna(subset=["patient_id"]).copy()
    data["patient_id"] = data["patient_id"].astype(str).str.strip()
    for column in data.columns.difference(list(_NON_NUMERIC_CLINICAL)):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["tumor_laterality"] = data["tumor_laterality"].astype(object).where(data["tumor_laterality"].notna(), np.nan)
    data["age_at_diagnosis_years"] = -data["date_of_birth_days"] / 365.25
    if data["patient_id"].duplicated().any():
        dups = sorted(set(data.loc[data["patient_id"].duplicated(), "patient_id"]))
        raise CohortError(f"duplicate patient ids in clinical workbook: {dups[:5]}")
    return data.reset_index(drop=True)


def read_metadata_digest(path: str | Path) -> pd.DataFrame:
    """Series-level metadata in file row order with canonical column names + ``row_index``."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    else:
        raw = pd.read_excel(path, sheet_name=METADATA_SHEET, header=0, dtype=object)
    missing = [c for c in METADATA_REQUIRED if c not in raw.columns]
    if missing:
        raise CohortError(f"metadata file {path.name} lacks required columns {missing}")
    rename = {**METADATA_REQUIRED, **{k: v for k, v in METADATA_OPTIONAL.items() if k in raw.columns}}
    out = raw[list(rename)].rename(columns=rename)
    for column in (PATIENT_ID, STUDY_ID, "series_uid"):
        out[column] = out[column].map(lambda v: np.nan if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v))
    if "series_description" in out:
        out["series_description"] = out["series_description"].map(lambda v: "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v))
    out.insert(0, "row_index", np.arange(len(out)))
    return out.reset_index(drop=True)


_ORDINAL_PASS = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\s*(?:pass|post|phase)")
_POST_NUMBER = re.compile(r"\bpost(?:contrast)?[\s_\-]*(\d+)\b")
_PHASE_NUMBER = re.compile(r"\bph(?:ase)?[\s_\-]*(\d+)\b")
_PRE = re.compile(r"\bpre(?:[\s_\-]*contrast)?\b|precontrast")


def parse_phase_index(description: str) -> int | None:
    """Phase index from a Duke-style DCE series description.

    ``... pre`` -> 0, ``... 1st pass`` / ``post1`` / ``ph1`` -> 1, etc.  Descriptions with no
    recognisable phase (e.g. ``ax t1``) return ``None`` (not a DCE phase).
    """
    text = str(description or "").strip().lower()
    if not text:
        return None
    for pattern in (_ORDINAL_PASS, _POST_NUMBER, _PHASE_NUMBER):
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    if _PRE.search(text):
        return 0
    return None


def assign_phase_indices(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add ``phase_index`` (nullable Int64): explicit column wins, else parsed from description."""
    out = metadata.copy()
    if "phase_index" in out.columns:
        out["phase_index"] = pd.to_numeric(out["phase_index"], errors="coerce").astype("Int64")
        return out
    if "series_description" not in out.columns:
        raise CohortError("phase_order=explicit needs 'Series Description' or 'Phase Index' in the metadata file "
                          "to interpret pre/post-contrast phases; neither is present")
    out["phase_index"] = pd.Series([parse_phase_index(d) for d in out["series_description"]], dtype="Int64")
    return _infer_unlabelled_pre_contrast(out)


_PHASE_PREFIX = re.compile(r"^ph(?:ase)?[\s_\-]*\d+\s*/\s*")
_TRAILING_PASS = re.compile(r"[\s_\-]*(?:\d+\s*(?:st|nd|rd|th)\s*(?:pass|post|phase)|post(?:contrast)?[\s_\-]*\d+)$")


def _normalise_description(description: Any) -> str:
    return " ".join(str(description or "").lower().split())


def _phase_base(description: Any) -> str:
    """Description without its post-contrast phase token (``Ph2/ax 3d dyn`` -> ``ax 3d dyn``)."""
    return _TRAILING_PASS.sub("", _PHASE_PREFIX.sub("", _normalise_description(description))).strip()


def _infer_unlabelled_pre_contrast(metadata: pd.DataFrame) -> pd.DataFrame:
    """Duke names the pre-contrast series like its post-contrast siblings minus the phase token
    (``ax 3d dyn`` next to ``Ph1/ax 3d dyn``; ``ax dyn`` next to ``ax dyn 1st pass``).

    In a study with parsed post-contrast phases but no phase 0, an unparsed series whose description
    equals the phase-stripped description of one of those posts becomes phase 0. Several such
    candidates are all marked and then excluded downstream as ``ambiguous_phase_index``.
    """
    if STUDY_ID not in metadata.columns:
        return metadata
    out = metadata.copy()
    for _, group in out.groupby(STUDY_ID, sort=False):
        phases = group["phase_index"]
        if (phases == 0).any() or not (phases > 0).any():
            continue
        bases = {_phase_base(d) for d, p in zip(group["series_description"], phases) if pd.notna(p) and p > 0}
        bases.discard("")
        candidates = [i for i, d, p in zip(group.index, group["series_description"], phases)
                      if pd.isna(p) and _normalise_description(d) in bases]
        if candidates:
            out.loc[candidates, "phase_index"] = 0
    return out


# ---------------------------------------------------------------------------
# Study selection
# ---------------------------------------------------------------------------
_AUDIT_COLUMNS = [PATIENT_ID, STUDY_ID, ONCOTYPE_SCORE, LABEL, "n_series_rows", "n_unique_series", "n_selected",
                  "included", "reason", "notes"]


def _audit_row(patient: str, study: str, score: float, label: Any, n_rows: int, n_unique: int, n_sel: int,
               included: bool, reason: str, notes: str = "") -> dict[str, Any]:
    return dict(zip(_AUDIT_COLUMNS, (patient, study, score, label, n_rows, n_unique, n_sel, included, reason, notes)))


def _scored_patients(clinical: pd.DataFrame, label_def: LabelDefinition) -> pd.DataFrame:
    scored = clinical.loc[clinical[ONCOTYPE_SCORE].notna(), ["patient_id", ONCOTYPE_SCORE]].copy()
    scored[LABEL] = scored[ONCOTYPE_SCORE].map(lambda s: label_def.encode(float(s))).astype("int64")
    return scored


def _unscored_audit(metadata: pd.DataFrame, clinical: pd.DataFrame, scored_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    clinical_ids = set(clinical["patient_id"])
    meta = metadata.dropna(subset=[STUDY_ID])
    for (patient, study), group in meta.groupby([PATIENT_ID, STUDY_ID], sort=True, dropna=True):
        if patient in scored_ids:
            continue
        reason = "missing_oncotype_score" if patient in clinical_ids else "no_clinical_record"
        rows.append(_audit_row(patient, study, np.nan, np.nan, len(group), group["series_uid"].nunique(), 0, False, reason))
    return rows


def _missing_metadata_audit(scored: pd.DataFrame, metadata: pd.DataFrame) -> list[dict[str, Any]]:
    present = set(metadata[PATIENT_ID].dropna())
    return [_audit_row(r.patient_id, "", getattr(r, ONCOTYPE_SCORE), getattr(r, LABEL), 0, 0, 0, False, "no_metadata_series")
            for r in scored.itertuples(index=False) if r.patient_id not in present]


def select_studies_legacy(metadata: pd.DataFrame, clinical: pd.DataFrame, label_def: LabelDefinition,
                          phase_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy eligibility (``read_from_xlmx.py:226-247``) -> (studies, audit)."""
    scored = _scored_patients(clinical, label_def)
    scored_ids = set(scored["patient_id"])
    labelled = metadata.merge(scored.rename(columns={"patient_id": PATIENT_ID}), on=PATIENT_ID, how="inner")
    audit = _unscored_audit(metadata, clinical, scored_ids) + _missing_metadata_audit(scored, metadata)
    rows = []
    for study, group in labelled.groupby(STUDY_ID, sort=True):  # groupby drops NaN study ids like legacy
        group = group.sort_values("row_index", kind="stable")
        patient = group[PATIENT_ID].iloc[0]
        n_unique = int(group["series_uid"].nunique())
        score, label = float(group[ONCOTYPE_SCORE].iloc[0]), int(group[LABEL].iloc[0])
        if n_unique < 2:
            audit.append(_audit_row(patient, study, score, label, len(group), n_unique, 0, False, "fewer_than_2_unique_series"))
            continue
        selected = tuple(group["series_uid"].tolist()[:phase_count])
        notes = []
        if len(group) > phase_count:
            notes.append(f"truncated_{len(group)}_to_{phase_count}")
        if len(selected) < phase_count:
            notes.append(f"only_{len(selected)}_series")
        if len(set(selected)) != len(selected):
            notes.append("duplicate_series_rows_selected")
        if group[PATIENT_ID].nunique() > 1:
            notes.append("study_spans_patients_first_taken")
        rows.append({PATIENT_ID: patient, STUDY_ID: study, LABEL: label, ONCOTYPE_SCORE: score, SERIES_UIDS: selected,
                     "series_descriptions": tuple(group["series_description"].tolist()[:phase_count]) if "series_description" in group else ()})
        audit.append(_audit_row(patient, study, score, label, len(group), n_unique, len(selected), True, "included", ";".join(notes)))
    studies = pd.DataFrame(rows, columns=[PATIENT_ID, STUDY_ID, LABEL, ONCOTYPE_SCORE, SERIES_UIDS, "series_descriptions"])
    return studies, pd.DataFrame(audit, columns=_AUDIT_COLUMNS)


def select_studies_general(metadata: pd.DataFrame, clinical: pd.DataFrame, label_def: LabelDefinition,
                           phase_count: int, phase_order: str = "explicit") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verified selection: explicit phase indices, pre + (phase_count-1) consecutive posts.

    ``phase_order=metadata`` keeps file row order but still requires that the parsed indices
    of the chosen series are 0..phase_count-1 in that order (otherwise excluded).
    """
    metadata = assign_phase_indices(metadata)
    missing_ids = metadata[[PATIENT_ID, STUDY_ID, "series_uid"]].isna().any(axis=1)
    if missing_ids.any():
        raise CohortError(f"metadata rows {metadata.loc[missing_ids, 'row_index'].tolist()[:10]} lack patient/study/series ids")
    metadata = metadata.copy()
    metadata[PATIENT_ID] = metadata[PATIENT_ID].str.strip()
    dup_series = metadata["series_uid"].duplicated(keep=False)
    series_in_studies = metadata.groupby("series_uid")[STUDY_ID].nunique()
    shared_series = set(series_in_studies[series_in_studies > 1].index)
    if shared_series:
        raise CohortError(f"series UIDs assigned to several studies: {sorted(shared_series)[:5]}")
    patients_per_study = metadata.groupby(STUDY_ID)[PATIENT_ID].nunique()

    scored = _scored_patients(clinical, label_def)
    score_map = scored.set_index("patient_id")
    audit = _unscored_audit(metadata, clinical, set(scored["patient_id"])) + _missing_metadata_audit(scored, metadata)
    rows = []
    for study, group in metadata.groupby(STUDY_ID, sort=True):
        group = group.sort_values("row_index", kind="stable")
        patient = group[PATIENT_ID].iloc[0]
        if patient not in score_map.index:
            continue  # already audited as unscored
        score, label = float(score_map.at[patient, ONCOTYPE_SCORE]), int(score_map.at[patient, LABEL])
        n_rows, n_unique = len(group), int(group["series_uid"].nunique())

        def exclude(reason: str, notes: str = "") -> None:
            audit.append(_audit_row(patient, study, score, label, n_rows, n_unique, 0, False, reason, notes))

        if patients_per_study[study] > 1:
            exclude("study_spans_patients"); continue
        if dup_series[group.index].any():
            exclude("duplicate_series_uid"); continue
        phased = group[group["phase_index"].notna()]
        ignored = n_rows - len(phased)
        notes = [f"ignored_{ignored}_non_phase_series"] if ignored else []
        if phased.empty:
            exclude("phase_unparseable", "no series description maps to a DCE phase"); continue
        counts = phased["phase_index"].value_counts()
        if (counts > 1).any():
            exclude("ambiguous_phase_index", f"repeated phases {sorted(int(i) for i in counts[counts > 1].index)}"); continue
        if 0 not in set(phased["phase_index"].astype(int)):
            exclude("missing_pre_contrast"); continue
        if len(phased) < 2:
            exclude("missing_post_contrast"); continue
        if phase_order == "explicit":
            ordered = phased.sort_values("phase_index", kind="stable")
        else:
            ordered = phased
        chosen = ordered.head(phase_count)
        indices = chosen["phase_index"].astype(int).tolist()
        if len(chosen) < phase_count:
            exclude("insufficient_phases", f"have phases {sorted(indices)}; need {phase_count}"); continue
        if indices != list(range(phase_count)):
            exclude("non_contiguous_or_unordered_phases", f"phases in selection order {indices}"); continue
        rows.append({PATIENT_ID: patient, STUDY_ID: study, LABEL: label, ONCOTYPE_SCORE: score,
                     SERIES_UIDS: tuple(chosen["series_uid"]),
                     "series_descriptions": tuple(chosen.get("series_description", pd.Series([""] * len(chosen))))})
        audit.append(_audit_row(patient, study, score, label, n_rows, n_unique, phase_count, True, "included", ";".join(notes)))
    studies = pd.DataFrame(rows, columns=[PATIENT_ID, STUDY_ID, LABEL, ONCOTYPE_SCORE, SERIES_UIDS, "series_descriptions"])
    return studies, pd.DataFrame(audit, columns=_AUDIT_COLUMNS)


def study_records(studies: pd.DataFrame, metadata: pd.DataFrame) -> list[StudyRecord]:
    """Explicit patient/study/series objects for the selected cohort (for audits/prepare)."""
    by_uid = {r.series_uid: r for r in metadata.itertuples(index=False)}
    records = []
    for row in studies.itertuples(index=False):
        series = []
        for uid in getattr(row, SERIES_UIDS):
            m = by_uid.get(uid)
            num = getattr(m, "num_images", None) if m is not None else None
            series.append(SeriesRecord(getattr(row, PATIENT_ID), getattr(row, STUDY_ID), uid,
                                       int(m.row_index) if m is not None else -1,
                                       str(getattr(m, "series_description", "") or "") if m is not None else "",
                                       None if m is None or pd.isna(getattr(m, "phase_index", np.nan)) else int(m.phase_index),
                                       int(num) if num is not None and not pd.isna(num) else None))
        records.append(StudyRecord(getattr(row, PATIENT_ID), getattr(row, STUDY_ID), tuple(series)))
    return records


# ---------------------------------------------------------------------------
# Tabular feature groups
# ---------------------------------------------------------------------------
def add_tabular_groups(studies: pd.DataFrame, clinical: pd.DataFrame, cfg: Mapping[str, Any]
                       ) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Attach the configured feature groups (legacy ``prepare_studies_and_features``)."""
    groups = tuple(dict.fromkeys(cfg.get("feature_groups", ("clinical",))))
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise CohortError(f"unknown feature groups {sorted(unknown)}")
    continuous: list[str] = []
    categorical: list[str] = []
    notes: list[str] = []
    frame = studies
    if "clinical" in groups:
        cols = list(CLINICAL_PREDICTOR_COLUMNS)
        if cfg.get("include_sensitive", False):
            cols += list(SENSITIVE_CLINICAL_PREDICTOR_COLUMNS)
        frame = frame.merge(clinical[["patient_id", *cols]], on=PATIENT_ID, how="left", validate="many_to_one")
        continuous += list(CLINICAL_CONTINUOUS_COLUMNS)
        categorical += [c for c in cols if c not in CLINICAL_CONTINUOUS_COLUMNS]
    imaging_groups = tuple(g for g in groups if g in IMAGING_FEATURE_GROUPS)
    if imaging_groups:
        path = cfg.get("imaging_features_file")
        if not path:
            raise CohortError("feature groups kinetic/morphology/heterogeneity need imaging_features_file")
        frame, columns, report = merge_imaging_features(
            frame, path, imaging_groups, source_patient_id_column=cfg.get("imaging_patient_id_column", "Patient ID"),
            require_complete_match=not cfg.get("allow_missing_imaging_features", False))
        continuous += columns
        notes.append(f"imaging features: {report}")
    if "density" in groups:
        path = cfg.get("density_file")
        if not path:
            raise CohortError("feature group density needs density_file")
        frame, cont, cat, report = merge_density(frame, path)
        continuous += cont
        categorical += cat
        notes.append(f"density coverage: {report}")
        if report["coverage"] < 0.5:
            notes.append("WARNING: density group covers <50% of studies; missing values are imputed fold-locally")
    if "mricore" in groups:
        path = cfg.get("mricore_features_file")
        if not path:
            raise CohortError("feature group mricore needs mricore_features_file")
        frame, cont, report = merge_mricore(frame, path)
        continuous += cont
        notes.append(f"mricore coverage: {report}")
        if report["coverage"] < 1.0:
            notes.append("WARNING: mricore embeddings are missing for some studies; missing values are imputed fold-locally")
    return frame, continuous, categorical, notes


# ---------------------------------------------------------------------------
class DukeCohortSource:
    """``contracts.CohortSource`` for the Duke workbooks + NBIA metadata digest."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = dict(cfg or {})

    def load(self, cfg: Mapping[str, Any] | None = None) -> CohortTable:
        cfg = {**self.cfg, **dict(cfg or {})}
        for key in ("clinical_features_file", "metadata_file"):
            if not cfg.get(key):
                raise CohortError(f"{key} is required to build the Duke cohort")
        profile = cfg.get("profile", "duke_legacy_3d")
        phase_count = int(cfg.get("phase_count", 5))
        phase_order = cfg.get("phase_order", "metadata")
        label_def = label_definition(cfg.get("label_scheme", "oncotype_binary_18"), int(cfg.get("positive_class", 1)))
        policy = policy_from_cfg(cfg)

        clinical = read_clinical_workbook(cfg["clinical_features_file"])
        metadata = read_metadata_digest(cfg["metadata_file"])
        notes = [f"profile={profile}", f"label={label_def.to_json()}", f"predictor_policy={policy.describe()}"]
        if profile == "duke_legacy_3d" and phase_order == "metadata":
            studies, audit = select_studies_legacy(metadata, clinical, label_def, phase_count)
            notes.append(LEGACY_ORDER_NOTE)
        elif profile == "duke_legacy_3d":
            raise CohortError("profile duke_legacy_3d reproduces metadata row order; use profile=general for phase_order=explicit")
        else:
            studies, audit = select_studies_general(metadata, clinical, label_def, phase_count, phase_order)
            notes.append(f"general: phases selected by parsed phase index (phase_order={phase_order}), pre + {phase_count - 1} posts required")
        if studies.empty:
            raise CohortError("no eligible studies (see audit)")
        # legacy final inner join with the clinical table (many_to_one)
        studies = studies.loc[studies[PATIENT_ID].isin(set(clinical["patient_id"]))].reset_index(drop=True)

        continuous: list[str] = []
        categorical: list[str] = []
        if "tabular" in tuple(cfg.get("branches", ("image", "tabular"))):
            studies, continuous, categorical, group_notes = add_tabular_groups(studies, clinical, cfg)
            notes += group_notes
            policy.check([*continuous, *categorical], prediction_time=cfg.get("prediction_time", "mri"),
                         include_sensitive=bool(cfg.get("include_sensitive", False)))
            check_label_leakage(studies, [*continuous, *categorical], score=studies[ONCOTYPE_SCORE])
            if "imaging_features_matched" in studies:
                matched = dict(zip(studies[STUDY_ID], studies["imaging_features_matched"]))
                audit["imaging_features_matched"] = audit[STUDY_ID].map(matched)
        studies = studies.drop(columns=[ONCOTYPE_SCORE, "imaging_features_matched"], errors="ignore")
        labels_present = sorted(studies[LABEL].unique())
        notes.append(f"studies={len(studies)} patients={studies[PATIENT_ID].nunique()} "
                     f"class_counts={studies[LABEL].value_counts().sort_index().to_dict()}")
        if labels_present != list(range(label_def.num_classes)):
            notes.append(f"WARNING: labels present {labels_present} do not cover all {label_def.num_classes} classes")
        frame = studies[[PATIENT_ID, STUDY_ID, LABEL, SERIES_UIDS, *continuous, *categorical,
                         *[c for c in studies.columns if c == "series_descriptions"]]].copy()
        frame[LABEL] = frame[LABEL].astype("int64")
        return CohortTable(frame=frame.reset_index(drop=True), continuous_columns=continuous,
                           categorical_columns=categorical, label=label_def, audit=audit.reset_index(drop=True), notes=notes)


def cohort_to_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """``series_uids``/``series_descriptions`` tuples joined with ``|`` (architecture section 4)."""
    out = frame.copy()
    for column in (SERIES_UIDS, "series_descriptions"):
        if column in out:
            out[column] = out[column].map(lambda t: "|".join(map(str, t)))
    return out


def cohort_from_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in (SERIES_UIDS, "series_descriptions"):
        if column in out:
            out[column] = out[column].map(lambda s: tuple(str(s).split("|")) if isinstance(s, str) and s else ())
    for column in (PATIENT_ID, STUDY_ID):
        out[column] = out[column].astype(str)
    return out
