"""MAMA-MIA resource import and mapping onto Duke series/patients (isolated ``prepare`` job).

Inventory of the legacy working scripts (``C:\\code\\mri_brest\\mama_mia``, SHA 127a2604):

``downloadMamamia.py``
    Module-level script: ``load_dotenv()``, reads the token from env ``SYNAPSE``,
    ``synapseclient.Synapse().login(authToken=...)`` and ``syn.get("syn64871114")``
    - all at import time; the result is not saved anywhere explicit (default
    Synapse cache).  Needs ``synapseclient`` and ``python-dotenv`` (neither is in
    the legacy requirements/experiment.conf).
``readFromXlmx.py``
    Reads ``features/41597_2025_4707_MOESM2_ESM.xlsx`` (Scientific Data
    supplement of the MAMA-MIA paper, see legacy ``urls.txt``), sheet
    ``dataset_info``, column ``tcia_series_uid``; intersects it with the Duke
    series of patients that have a numeric Oncotype score
    (``get_oncotype_score_for_series(False)``, 3-class labels) and prints the
    count at import time.  ``mama_mia/features`` is git-ignored, so the table is
    not in the repository.

This module keeps both functions as an explicit, side-effect-free job:

* :func:`download_resource` - Synapse download through an **injected** client
  (:class:`SynapseClient`); :func:`make_synapse_client` creates the real one
  lazily from a token held in an environment variable (never logged, never
  written to artifacts);
* :func:`read_mama_mia_table` + :func:`map_to_duke` - the UID intersection,
  extended to a per-series mapping table with patient/study ids, outcome
  cohort membership and coverage counts.

Nothing here is used by train/validate; MAMA-MIA masks enter the classifier
only through the mask manifest (``mask_provider=local_files``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import pandas as pd

LEGACY_ENTITY_ID = "syn64871114"
LEGACY_TABLE_NAME = "41597_2025_4707_MOESM2_ESM.xlsx"
LEGACY_SHEET = "dataset_info"
LEGACY_UID_COLUMN = "tcia_series_uid"
TOKEN_ENV = "SYNAPSE"
MAPPING_COLUMNS = ("patient_id", "study_id", "series_uid", "in_mama_mia", "in_outcome_cohort")

LEGACY_SCRIPTS = (
    {"file": "mama_mia/downloadMamamia.py", "function": "<module>", "status": "replaced",
     "new": "mribreast.prepare.mama_mia.download_resource / python -m mribreast.prepare mama-mia-download"},
    {"file": "mama_mia/readFromXlmx.py", "function": "read_series_uids", "status": "replaced",
     "new": "mribreast.prepare.mama_mia.read_mama_mia_table"},
    {"file": "mama_mia/readFromXlmx.py", "function": "get_mamamia_images_for_oncotype", "status": "extended",
     "new": "mribreast.prepare.mama_mia.map_to_duke (+ summary counts) / python -m mribreast.prepare mama-mia-map"},
)


class SynapseClient(Protocol):
    def get(self, entity: str, downloadLocation: str | None = None) -> Any:  # noqa: N803 - synapseclient API
        ...


def make_synapse_client(token_env: str = TOKEN_ENV) -> SynapseClient:  # pragma: no cover - network
    """Real client (lazy import, explicit call only). Token is read from ``token_env``."""
    import synapseclient

    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"set the Synapse personal access token in ${token_env}")
    client = synapseclient.Synapse()
    client.login(authToken=token, silent=True)
    return client


def download_resource(client: SynapseClient, dest_dir: str | os.PathLike, entity_id: str = LEGACY_ENTITY_ID) -> Path:
    """Download ``entity_id`` into ``dest_dir``; returns the local file path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    entity = client.get(entity_id, downloadLocation=str(dest_dir))
    path = getattr(entity, "path", None) or (entity.get("path") if isinstance(entity, Mapping) else None)
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Synapse entity {entity_id} did not produce a local file")
    return Path(path)


def read_mama_mia_table(path: str | os.PathLike, *, sheet: str = LEGACY_SHEET,
                        uid_column: str = LEGACY_UID_COLUMN) -> pd.DataFrame:
    """Read the MAMA-MIA dataset table (xlsx sheet or CSV); ``series_uid`` column added, UIDs stripped."""
    path = Path(path)
    frame = pd.read_csv(path, dtype=str) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=sheet,
                                                                                            header=0, dtype=str)
    if uid_column not in frame.columns:
        raise ValueError(f"{path}: column {uid_column!r} not found (columns: {list(frame.columns)[:12]})")
    frame = frame.copy()
    frame["series_uid"] = frame[uid_column].astype(str).str.strip()
    frame = frame[frame["series_uid"].ne("") & frame["series_uid"].ne("nan")]
    duplicated = frame["series_uid"].duplicated()
    if duplicated.any():
        raise ValueError(f"{path}: duplicate {uid_column} values, e.g. {frame.loc[duplicated, 'series_uid'].iloc[0]}")
    return frame.reset_index(drop=True)


def map_to_duke(mama: pd.DataFrame, duke_metadata: pd.DataFrame, outcome_patients: Iterable[str] | None = None,
                *, keep_columns: Iterable[str] = ()) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per-Duke-series mapping + coverage summary.

    ``duke_metadata`` needs ``patient_id, study_id, series_uid`` (see
    :func:`mribreast.prepare.nbia.read_nbia_metadata`).  ``summary['legacy_intersection']``
    equals the number printed by legacy ``get_mamamia_images_for_oncotype``
    (Duke series of outcome patients that appear in MAMA-MIA).
    """
    for column in ("patient_id", "study_id", "series_uid"):
        if column not in duke_metadata.columns:
            raise ValueError(f"duke metadata lacks {column}")
    duke = duke_metadata[["patient_id", "study_id", "series_uid"]].astype(str).drop_duplicates("series_uid")
    mama_uids = set(mama["series_uid"])
    out = duke.copy()
    out["in_mama_mia"] = out["series_uid"].isin(mama_uids)
    outcome = set(outcome_patients) if outcome_patients is not None else None
    out["in_outcome_cohort"] = out["patient_id"].isin(outcome) if outcome is not None else pd.NA
    extra = [c for c in keep_columns if c in mama.columns and c != "series_uid"]
    if extra:
        out = out.merge(mama[["series_uid", *extra]], on="series_uid", how="left", validate="one_to_one")
    summary = {
        "duke_series": int(len(out)),
        "mama_mia_series": int(len(mama_uids)),
        "mama_mia_series_in_duke": int(len(mama_uids & set(out["series_uid"]))),
        "mama_mia_series_not_in_duke": int(len(mama_uids - set(out["series_uid"]))),
        "duke_patients_with_mama_mia": int(out.loc[out["in_mama_mia"], "patient_id"].nunique()),
    }
    if outcome is not None:
        hit = out["in_mama_mia"] & out["in_outcome_cohort"].astype(bool)
        summary["legacy_intersection"] = int(out.loc[hit, "series_uid"].nunique())
        summary["outcome_patients_with_mama_mia"] = int(out.loc[hit, "patient_id"].nunique())
    return out.reset_index(drop=True), summary
