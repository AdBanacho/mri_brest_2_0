"""Optional feature group ``mricore``: frozen MRI-CORE encoder embeddings per study.

The embeddings are produced by the explicit preparation job
``python -m mribreast.prepare mricore-embed`` (see :mod:`mribreast.prepare.mricore_embed`)
and read here from its CSV output (``mricore_features_file``):

* ``patient_id``, ``study_id`` - join keys (one row per study, no duplicates);
* ``mricore_*`` - float predictors (e.g. ``mricore_roi_000`` .. ``mricore_roi_255``);
* any other column (``series_uid``, ``phase``, ``n_slices``, ``pool`` ...) is provenance
  and is **not** used as a predictor.

The encoder is a frozen, label-free foundation model applied to each study
independently, so the features carry no fold information and can be computed
once for the whole cohort.  Studies without a row (missing NIfTI or annotation
box at extraction time) get NaN and are imputed fold-locally like any predictor;
coverage is reported in the cohort notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mribreast.contracts import PATIENT_ID, STUDY_ID

MRICORE_PREFIX = "mricore_"


class MriCoreFeaturesError(ValueError):
    pass


def read_mricore_features(path: str | Path) -> pd.DataFrame:
    """Return ``patient_id, study_id`` + the ``mricore_*`` columns as float."""
    path = Path(path)
    raw = pd.read_csv(path, dtype={PATIENT_ID: str, STUDY_ID: str})
    missing = [c for c in (PATIENT_ID, STUDY_ID) if c not in raw.columns]
    if missing:
        raise MriCoreFeaturesError(f"{path}: lacks columns {missing}")
    features = [c for c in raw.columns if str(c).startswith(MRICORE_PREFIX)]
    if not features:
        raise MriCoreFeaturesError(f"{path}: no '{MRICORE_PREFIX}*' feature columns")
    out = raw[[PATIENT_ID, STUDY_ID]].astype(str).apply(lambda col: col.str.strip())
    dup = out.duplicated([PATIENT_ID, STUDY_ID])
    if dup.any():
        raise MriCoreFeaturesError(f"{path}: duplicate studies {out.loc[dup].values[:3].tolist()}")
    values = raw[features].apply(pd.to_numeric, errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise MriCoreFeaturesError(f"{path}: non-numeric or non-finite values in feature columns")
    return pd.concat([out, values], axis=1)


def merge_mricore(studies: pd.DataFrame, path: str | Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Left one-to-one merge on (patient_id, study_id); returns (frame, continuous columns, coverage report)."""
    features = read_mricore_features(path)
    columns = [c for c in features.columns if c.startswith(MRICORE_PREFIX)]
    clash = sorted(set(columns) & set(studies.columns))
    if clash:
        raise MriCoreFeaturesError(f"mricore columns already present in the cohort: {clash[:5]}")
    merged = studies.merge(features, on=[PATIENT_ID, STUDY_ID], how="left", validate="one_to_one")
    matched = merged[columns[0]].notna()
    report = {"studies_in_file": int(len(features)), "studies": int(len(merged)),
              "studies_with_mricore": int(matched.sum()), "features": len(columns)}
    report["coverage"] = report["studies_with_mricore"] / max(1, report["studies"])
    return merged, columns, report
