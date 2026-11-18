"""NBIA/TCIA download of Duke DICOM series (explicit ``prepare`` stage).

Replaces legacy ``mriBreastDuke/dataDownloaders/downloadDicom.py`` (``tcia_utils.nbia``
``getSeries`` + ``downloadSeries``; ``download_all`` for every study of the NBIA
metadata sheet, ``download_oncotype`` for studies of patients with a numeric
Oncotype score).  Differences, all deliberate:

* the HTTP client is injected (:class:`HttpGet`); the default uses only the
  standard library and is created lazily - nothing touches the network at import;
* each series is extracted into ``<dest>/<SeriesInstanceUID>/`` via a temporary
  directory and an atomic rename, with a ``.download.json`` completion marker;
  completed series are skipped on re-runs (resume);
* zip members with absolute paths or ``..`` are rejected;
* ``download_manifest.csv`` records study, series, patient, modality, image
  count, file count and status for every series.

The DICOM -> NIfTI conversion (with UID/geometry verification) is a separate
step: :mod:`mribreast.prepare.dicom_to_nifti`.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import pandas as pd

DEFAULT_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DUKE_COLLECTION = "Duke-Breast-Cancer-MRI"
COMPLETE_MARKER = ".download.json"
COHORTS = ("all", "outcome")
METADATA_COLUMNS = {"Patient ID": "patient_id", "Study Instance UID": "study_id", "Series Instance UID": "series_uid"}
MANIFEST_COLUMNS = ("study_uid", "series_uid", "patient_id", "modality", "series_description", "image_count",
                    "files", "status", "error")


class HttpGet(Protocol):
    def __call__(self, url: str, params: Mapping[str, str], timeout: float) -> bytes:
        ...


def urllib_get(url: str, params: Mapping[str, str], timeout: float = 300.0) -> bytes:
    """Stdlib GET (no third-party dependency); raises on HTTP errors."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    query = urlencode(dict(params))
    request = Request(f"{url}?{query}" if query else url, headers={"User-Agent": "mribreast-prepare"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit https endpoint from config
        return response.read()


class NbiaClient(Protocol):
    def get_series(self, collection: str, study_uid: str) -> list[dict[str, Any]]:
        ...

    def get_series_zip(self, series_uid: str) -> bytes:
        ...


@dataclass
class RestNbiaClient:
    """Minimal NBIA REST v1 client (public collections need no token)."""

    base_url: str = DEFAULT_BASE_URL
    http_get: HttpGet | Callable[..., bytes] | None = None
    timeout: float = 300.0

    def _get(self, endpoint: str, params: Mapping[str, str]) -> bytes:
        getter = self.http_get or urllib_get
        return getter(f"{self.base_url.rstrip('/')}/{endpoint}", params, self.timeout)

    def get_series(self, collection: str, study_uid: str) -> list[dict[str, Any]]:
        payload = self._get("getSeries", {"Collection": collection, "StudyInstanceUID": study_uid, "format": "json"})
        if not payload.strip():
            return []
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"unexpected getSeries payload for {study_uid}: {type(data).__name__}")
        return data

    def get_series_zip(self, series_uid: str) -> bytes:
        return self._get("getImage", {"SeriesInstanceUID": series_uid})


# --------------------------------------------------------------------------- cohort selection
def read_nbia_metadata(path: str | os.PathLike) -> pd.DataFrame:
    """NBIA digest (sheet ``Metadata``) or CSV -> ``patient_id, study_id, series_uid`` (+ other columns), file order kept."""
    path = Path(path)
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name="Metadata", header=0)
    missing = [c for c in METADATA_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: metadata lacks columns {missing}")
    frame = frame.rename(columns=METADATA_COLUMNS)
    for column in METADATA_COLUMNS.values():
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def read_outcome_patients(path: str | os.PathLike) -> set[str]:
    """Patients with a numeric Oncotype score (legacy ``read_patient_id_for_oncotype_score_not_na``).

    Workbook: sheet ``Data``, three header rows, column A = patient, AB = score.
    CSV (tests/exports): columns ``patient_id`` and ``oncotype_score``.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"patient_id": str})
    else:
        frame = pd.read_excel(path, sheet_name="Data", header=None, skiprows=3, usecols="A,AB",
                              names=["patient_id", "oncotype_score"])
    scores = pd.to_numeric(frame["oncotype_score"], errors="coerce")
    ids = frame.loc[scores.notna(), "patient_id"].dropna().astype(str).str.strip()
    return set(ids[ids != ""])


def select_study_uids(metadata: pd.DataFrame, cohort: str = "outcome", outcome_patients: Iterable[str] | None = None) -> list[str]:
    """Unique study UIDs in metadata order; ``outcome`` keeps patients with a numeric score."""
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}")
    frame = metadata
    if cohort == "outcome":
        if outcome_patients is None:
            raise ValueError("cohort=outcome needs the outcome patient set (clinical workbook)")
        frame = frame[frame["patient_id"].isin(set(outcome_patients))]
    return list(dict.fromkeys(frame["study_id"]))


# --------------------------------------------------------------------------- download
def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        name = PurePosixPath(info.filename.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts or (name.parts and ":" in name.parts[0]):
            raise ValueError(f"unsafe zip member {info.filename!r}")
        if not info.is_dir():
            members.append(info)
    return members


def is_complete(series_dir: Path) -> bool:
    return (series_dir / COMPLETE_MARKER).is_file()


def download_series(client: NbiaClient, series: Mapping[str, Any], dest_root: str | os.PathLike, *,
                    overwrite: bool = False) -> dict[str, Any]:
    """Download one series zip and publish it atomically as ``<dest_root>/<SeriesInstanceUID>/``."""
    series_uid = str(series["SeriesInstanceUID"])
    dest_root = Path(dest_root)
    target = dest_root / series_uid
    row = {"study_uid": str(series.get("StudyInstanceUID", "")), "series_uid": series_uid,
           "patient_id": str(series.get("PatientID", "")), "modality": str(series.get("Modality", "")),
           "series_description": str(series.get("SeriesDescription", "")),
           "image_count": series.get("ImageCount", ""), "files": 0, "status": "", "error": ""}
    if is_complete(target) and not overwrite:
        row["files"] = sum(1 for p in target.rglob("*") if p.is_file() and p.name != COMPLETE_MARKER)
        row["status"] = "skipped_existing"
        return row
    dest_root.mkdir(parents=True, exist_ok=True)
    payload = client.get_series_zip(series_uid)
    tmp = Path(tempfile.mkdtemp(prefix=f".{series_uid}.", dir=dest_root))
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = _safe_members(archive)
            for info in members:
                archive.extract(info, tmp)
        files = sum(1 for p in tmp.rglob("*") if p.is_file())
        expected = series.get("ImageCount")
        if expected not in (None, "") and int(expected) != files:
            raise ValueError(f"series {series_uid}: {files} files extracted, NBIA ImageCount={expected}")
        (tmp / COMPLETE_MARKER).write_text(json.dumps({k: row[k] for k in MANIFEST_COLUMNS[:6]} | {"files": files},
                                                      sort_keys=True), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
        row.update(files=files, status="downloaded")
        return row
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def download_studies(client: NbiaClient, study_uids: Sequence[str], dest_root: str | os.PathLike, *,
                     collection: str = DUKE_COLLECTION, overwrite: bool = False, stop_on_error: bool = False,
                     progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Download every series of ``study_uids``; returns (and does not write) the manifest frame."""
    rows: list[dict[str, Any]] = []
    for index, study_uid in enumerate(study_uids, start=1):
        try:
            series_list = client.get_series(collection, study_uid)
        except Exception as error:  # noqa: BLE001 - recorded in the manifest
            if stop_on_error:
                raise
            rows.append({"study_uid": study_uid, "series_uid": "", "status": "failed_listing", "error": repr(error)})
            continue
        if not series_list:
            rows.append({"study_uid": study_uid, "series_uid": "", "status": "no_series", "error": ""})
        for series in series_list:
            if str(series.get("StudyInstanceUID", study_uid)) != str(study_uid):
                rows.append({"study_uid": study_uid, "series_uid": str(series.get("SeriesInstanceUID", "")),
                             "status": "failed", "error": "series listed under a different StudyInstanceUID"})
                continue
            try:
                rows.append(download_series(client, {"StudyInstanceUID": study_uid, **series}, dest_root,
                                            overwrite=overwrite))
            except Exception as error:  # noqa: BLE001
                if stop_on_error:
                    raise
                rows.append({"study_uid": study_uid, "series_uid": str(series.get("SeriesInstanceUID", "")),
                             "status": "failed", "error": repr(error)})
        if progress:
            progress(f"[{index}/{len(study_uids)}] {study_uid}")
    frame = pd.DataFrame(rows)
    for column in MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[list(MANIFEST_COLUMNS)]


def write_manifest(frame: pd.DataFrame, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path
