"""Synthetic Duke-format tables (no real patient data).

``write_duke_tables`` writes workbooks with the same layout the readers expect:

* ``Clinical_and_Other_Features.xlsx`` — sheet ``Data`` with 3 header rows (group, name,
  coding) and 98 columns A..CT; values in the legacy-read letters A,T,U,V,AB,AK,AW..BA,
  plausible synthetic values elsewhere; ``NC``/``NP`` missing tokens sprinkled in.
* ``nbia_metadata.xlsx`` — sheet ``Metadata`` (NBIA digest columns, see ``data.duke``).
* ``Imaging_Features.xlsx`` — sheet ``Imaging Features``, ``Patient ID`` + a subset of the
  real Duke column NAMES (values synthetic) covering kinetic/morphology/heterogeneity.
* ``Annotation_Boxes.xlsx`` — sheet ``Sheet1``, legacy 7 columns, zero-based inclusive
  indices inside ``shape`` interpreted as (columns, rows, slices) = NIfTI axes (0, 1, 2).
* ``Breast_Radiologist_Density_Assessments.xlsx`` — sheet ``Assessments`` with the real schema.

Cohort design (deterministic for a seed): ~10 % patients without score, two patients
whose only study has 1 series (ineligible), two (+ ``studies_per_patient_extra``)
patients with two studies, ~30 % studies with a 6th non-DCE series appended after the
5 DCE phases (legacy truncation), two studies whose DCE rows are not in acquisition
order (legacy keeps row order; ``general`` re-orders by phase), one study with blank
series descriptions (``general`` excludes it as ``phase_unparseable``).  Scores 18 and 31
appear exactly to exercise label boundaries.  Tabular predictors carry a weak label
signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

from mribreast.contracts import LABEL, PATIENT_ID, SERIES_UIDS, STUDY_ID, label_definition

FILE_NAMES = {
    "clinical_features_file": "Clinical_and_Other_Features.xlsx",
    "metadata_file": "nbia_metadata.xlsx",
    "imaging_features_file": "Imaging_Features.xlsx",
    "annotation_boxes_file": "Annotation_Boxes.xlsx",
    "density_file": "Breast_Radiologist_Density_Assessments.xlsx",
}
DCE_DESCRIPTIONS = ("ax dyn pre", "ax dyn 1st pass", "ax dyn 2nd pass", "ax dyn 3rd pass", "ax dyn 4th pass")
EXTRA_DESCRIPTION = "ax t1 tse"
IMAGING_COLUMNS = {
    "kinetic": ["F1_DT_POSTCON (T11=0.05,T12=0.5)", "Max_Enhancement_from_char_curv", "Time_to_Peak_from_char_curv",
                "Uptake_rate_from_char_curv", "Washout_rate_from_char_curv", "Peak_SER_tumor"],
    "morphology": ["TumorMajorAxisLength_mm", "Volume_cu_mm_Tumor", "BEVR_Tumor", "Median_solidity_Tumor", "BreastVol"],
    "heterogeneity": ["Autocorrelation_Tumor", "Contrast_Tumor", "Entropy_Tumor", "Energy_Tumor", "SER_map_mean_tumor",
                      "Variance_of_Uptake_Tumor"],
}
DENSITY_LEGEND = ("BI‐RADS density categories           (a. Almost entirely fat                          "
                  "b. Scattered fibroglandular tissue\nc. Heterogeneous fibroglandular tissue\nd. Extreme fibroglandular tissue)")
# (letter, group header, name, coding) for columns read by legacy + a few realistic neighbours
_CLINICAL_LAYOUT = {
    "A": ("Patient Information", "Patient ID", None),
    "B": ("MRI Technical Information", "Days to MRI (From the Date of Diagnosis)", None),
    "T": ("Demographics", "Date of Birth (Days)", "(Taking date of diagnosis as day 0)"),
    "U": (None, "Menopause (at diagnosis)", "{0 = pre,\n1 = post,\n2 = N/A}"),
    "V": (None, "Race and Ethnicity", "{0 = N/A\n1 = white,\n2 = black,\n3 = asian,\n4 = native,\n5 = hispanic}"),
    "W": (None, "Metastatic at Presentation (Outside of Lymph Nodes)", "{0 = no\n1 = yes}"),
    "X": ("Tumor Characteristics", "ER", "{0 = neg,\n1 = pos}"),
    "Y": (None, "PR", "{0 = neg,\n1 = pos}"),
    "Z": (None, "HER2", "{0 = neg,\n1 = pos, 2 = borderline}"),
    "AB": (None, "Oncotype score", None),
    "AI": (None, "Nottingham grade", "1=low 2=intermediate 3=high"),
    "AK": (None, "Tumor Location", "Side of cancer L=left R=right"),
    "AW": ("MRI Findings", "Multicentric/Multifocal", "{0 = no, 1 = yes}"),
    "AX": (None, "Contralateral Breast Involvement", "{0 = no, 1 = yes}"),
    "AY": (None, "Lymphadenopathy or Suspicious Nodes", "{0 = no, 1 = yes}"),
    "AZ": (None, "Skin/Nipple Invovlement", "{0 = no, 1 = yes}"),
    "BA": (None, "Pec/Chest Involvement", "{0 = no, 1 = yes}"),
    "BB": ("SURGERY", "Surgery", "{0 = no,1 = yes}"),
    "BI": ("Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}"),
}
N_CLINICAL_COLUMNS = 98  # A..CT


def _uid(rng: np.random.Generator) -> str:
    return "2.25." + str(int(rng.integers(10**17, 10**18)))  # synthetic UUID-derived OID arc, not a Duke UID


def _score_for_class(rng: np.random.Generator, klass: int) -> float:
    lo, hi = [(3, 18), (19, 31), (32, 60)][klass]
    return float(rng.integers(lo, hi + 1))


def write_duke_tables(root: Path, n_patients: int = 60, seed: int = 0, shape: Sequence[int] = (32, 32, 16),
                      studies_per_patient_extra: int = 0,
                      label_scheme_balance: Sequence[float] = (0.5, 0.3, 0.2)) -> dict[str, Any]:
    """Write all synthetic Duke tables under ``root``.

    ``label_scheme_balance``: fractions of scored patients in the score bands <=18, (18,31], >31.
    ``studies_per_patient_extra``: additional patients (beyond the default two) that get a second study.
    Returns paths keyed by config field name plus ``studies`` (eligible under ``duke_legacy_3d``:
    patient_id, study_id, label (binary), label_3class, oncotype_score, series_uids),
    ``series`` (every metadata row), ``patients`` and ``boxes`` DataFrames.
    """
    if n_patients < 12:
        raise ValueError("n_patients must be >= 12 to include every edge case")
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    balance = np.asarray(label_scheme_balance, dtype=float)
    balance = balance / balance.sum()

    ids = [f"Breast_MRI_{i + 1:03d}" for i in range(n_patients)]
    order = rng.permutation(n_patients)
    n_unscored = max(2, n_patients // 10)
    unscored = set(order[:n_unscored].tolist())
    single_series = set(order[n_unscored:n_unscored + 2].tolist())
    n_multi = 2 + int(studies_per_patient_extra)
    multi = set(order[n_unscored + 2:n_unscored + 2 + n_multi].tolist())
    rest = order[n_unscored + 2 + n_multi:].tolist()
    shuffled_order = set(rest[:2])
    blank_desc = set(rest[2:3])

    patients = []
    for i, pid in enumerate(ids):
        klass = int(rng.choice(3, p=balance))
        score = np.nan if i in unscored else _score_for_class(rng, klass)
        patients.append({PATIENT_ID: pid, "oncotype_score": score})
    patients = pd.DataFrame(patients)
    scored_idx = [i for i in range(n_patients) if i not in unscored]
    patients.loc[scored_idx[0], "oncotype_score"] = 18.0   # boundary -> binary 0
    patients.loc[scored_idx[1], "oncotype_score"] = 31.0   # boundary -> 3-class 1
    patients.loc[scored_idx[2], "oncotype_score"] = 19.0
    # latent signal: higher score -> higher signal; unscored get random latent
    score = patients["oncotype_score"].to_numpy(dtype=float)
    latent = np.where(np.isnan(score), rng.normal(size=n_patients), np.clip((np.nan_to_num(score) - 18.0) / 10.0, -1.5, 2.0))
    signal = 0.6 * latent + rng.normal(scale=1.0, size=n_patients)  # weak label signal
    patients["signal"] = signal

    # ------------------------------------------------------------------ metadata
    series_rows = []
    for i, pid in enumerate(ids):
        n_studies = 2 if i in multi else 1
        for s in range(n_studies):
            study = _uid(rng)
            if i in single_series:
                descs = [DCE_DESCRIPTIONS[0]]
            else:
                descs = list(DCE_DESCRIPTIONS)
                if i in shuffled_order and s == 0:
                    descs = [descs[2], descs[0], descs[1], descs[4], descs[3]]
                if rng.random() < 0.3:
                    descs.append(EXTRA_DESCRIPTION)
            for k, desc in enumerate(descs):
                series_rows.append({"Collection": "Synthetic-Duke-Like", "Patient ID": pid, "Study Instance UID": study,
                                    "Study Date": f"2000-01-{1 + s:02d}", "Study Description": "BREAST MRI (synthetic)",
                                    "Series Instance UID": _uid(rng),
                                    "Series Description": "" if i in blank_desc else desc,
                                    "Series Number": 100 * (k + 1), "Modality": "MR", "Manufacturer": "SYNTHETIC",
                                    "Number of Images": int(shape[2])})
    series = pd.DataFrame(series_rows)
    paths = {key: root / name for key, name in FILE_NAMES.items()}
    with pd.ExcelWriter(paths["metadata_file"], engine="openpyxl") as writer:
        series.to_excel(writer, sheet_name="Metadata", index=False)

    # ------------------------------------------------------------------ clinical workbook
    age = np.clip(55 + 6 * signal + rng.normal(scale=8, size=n_patients), 25, 85)
    menopause = (age > 51).astype(int)
    tumour_side = rng.choice(["L", "R"], size=n_patients)
    prob = 1 / (1 + np.exp(-(signal - 0.5)))
    values: dict[str, list[Any]] = {
        "A": ids,
        "B": rng.integers(0, 30, size=n_patients).tolist(),
        "T": (-age * 365.25).round().astype(int).tolist(),
        "U": menopause.tolist(),
        "V": rng.integers(0, 6, size=n_patients).tolist(),
        "W": [0] * n_patients,
        "X": rng.integers(0, 2, size=n_patients).tolist(),
        "Y": rng.integers(0, 2, size=n_patients).tolist(),
        "Z": rng.integers(0, 2, size=n_patients).tolist(),
        "AB": [None if np.isnan(v) else (int(v) if float(v).is_integer() else float(v)) for v in score],
        "AI": rng.integers(1, 4, size=n_patients).tolist(),
        "AK": tumour_side.tolist(),
        "AW": (rng.random(n_patients) < prob).astype(int).tolist(),
        "AX": (rng.random(n_patients) < 0.1).astype(int).tolist(),
        "AY": (rng.random(n_patients) < prob * 0.8).astype(int).tolist(),
        "AZ": (rng.random(n_patients) < 0.08).astype(int).tolist(),
        "BA": (rng.random(n_patients) < 0.05).astype(int).tolist(),
        "BB": [1] * n_patients,
        "BI": (rng.random(n_patients) < 0.1).astype(int).tolist(),
    }
    for letter, token in (("AK", "NC"), ("U", "NP"), ("AY", "NC")):  # missing-value tokens
        for j in rng.choice(n_patients, size=max(1, n_patients // 20), replace=False):
            values[letter][j] = token
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    for col in range(1, N_CLINICAL_COLUMNS + 1):
        letter = get_column_letter(col)
        group, name, coding = _CLINICAL_LAYOUT.get(letter, (None, f"Synthetic column {letter}", None))
        ws.cell(row=1, column=col, value=group)
        ws.cell(row=2, column=col, value=name)
        ws.cell(row=3, column=col, value=coding)
        column_values = values.get(letter)
        for r in range(n_patients):
            value = column_values[r] if column_values is not None else int(rng.integers(0, 3))
            ws.cell(row=4 + r, column=col, value=value)
    legend = wb.create_sheet("Legend")
    legend["A1"] = "Synthetic workbook for tests; no real patient data."
    wb.create_sheet("Column Explanations")["A1"] = "See mribreast.data.duke.CLINICAL_EXCEL_COLUMNS"
    wb.save(paths["clinical_features_file"])

    # ------------------------------------------------------------------ imaging features
    imaging = {"Patient ID": ids}
    for group, columns in IMAGING_COLUMNS.items():
        for j, column in enumerate(columns):
            weight = 0.4 if j % 2 == 0 else 0.0
            imaging[column] = (weight * signal + rng.normal(size=n_patients)).round(6)
    imaging_frame = pd.DataFrame(imaging)
    for column in ("Contrast_Tumor", "BEVR_Tumor"):
        imaging_frame.loc[rng.choice(n_patients, size=2, replace=False), column] = np.nan
    with pd.ExcelWriter(paths["imaging_features_file"], engine="openpyxl") as writer:
        imaging_frame.to_excel(writer, sheet_name="Imaging Features", index=False)

    # ------------------------------------------------------------------ annotation boxes
    cols_n, rows_n, slices_n = (int(shape[0]), int(shape[1]), int(shape[2]))
    boxes = []
    for pid in ids:
        def span(n: int) -> tuple[int, int]:
            length = max(1, n // 4)
            start = int(rng.integers(0, n - length + 1))
            return start, start + length - 1
        r0, r1 = span(rows_n)
        c0, c1 = span(cols_n)
        s0, s1 = span(slices_n)
        boxes.append({"Patient ID": pid, "Start Row": r0, "End Row": r1, "Start Column": c0, "End Column": c1,
                      "Start Slice": s0, "End Slice": s1})
    boxes = pd.DataFrame(boxes)
    with pd.ExcelWriter(paths["annotation_boxes_file"], engine="openpyxl") as writer:
        boxes.to_excel(writer, sheet_name="Sheet1", index=False)

    # ------------------------------------------------------------------ density
    assessed = sorted(rng.choice(n_patients, size=max(4, int(0.3 * n_patients)), replace=False).tolist())
    letters = "abcd"
    density_rows = []
    for j in assessed:
        base = int(np.clip(round(1.5 + 0.5 * signal[j] + rng.normal(scale=0.7)), 0, 3))
        raters = [letters[int(np.clip(base + rng.integers(-1, 2), 0, 3))] if rng.random() < 0.4 else letters[base]
                  for _ in range(3)]
        density_rows.append({"Subject_ID": ids[j], "Radiologist A": raters[0], "Radiologist B": raters[1],
                             "Radiologist C": raters[2], DENSITY_LEGEND: np.nan})
    with pd.ExcelWriter(paths["density_file"], engine="openpyxl") as writer:
        pd.DataFrame(density_rows).to_excel(writer, sheet_name="Assessments", index=False)

    # ------------------------------------------------------------------ eligible studies (legacy rule)
    binary, three = label_definition("oncotype_binary_18"), label_definition("oncotype_3class_18_31")
    meta = series.rename(columns={"Patient ID": PATIENT_ID, "Study Instance UID": STUDY_ID,
                                  "Series Instance UID": "series_uid"})
    score_map = dict(zip(patients[PATIENT_ID], patients["oncotype_score"]))
    eligible = []
    for study, group in meta.groupby(STUDY_ID, sort=True):
        pid = group[PATIENT_ID].iloc[0]
        s = score_map[pid]
        if np.isnan(s) or group["series_uid"].nunique() < 2:
            continue
        eligible.append({PATIENT_ID: pid, STUDY_ID: study, LABEL: binary.encode(s), "label_3class": three.encode(s),
                         "oncotype_score": s, SERIES_UIDS: tuple(group["series_uid"].tolist()[:5])})
    studies = pd.DataFrame(eligible)
    return {**paths, "root": root, "studies": studies, "series": series, "patients": patients, "boxes": boxes}
