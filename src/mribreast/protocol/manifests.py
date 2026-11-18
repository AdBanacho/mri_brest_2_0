"""Cohort serialisation, fingerprints and manifest verification shared by train and validate."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from mribreast import CONFIG_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION
from mribreast.artifacts import ArtifactMismatch, file_fingerprint, read_json
from mribreast.contracts import (COHORT_REQUIRED_COLUMNS, LABEL, ONCOTYPE_SCORE, PATIENT_ID, SERIES_UIDS, STUDY_ID,
                                 CohortTable)

TRAIN_MANIFEST = "train_manifest.json"
FOLD_MANIFEST = "fold_manifest.json"
SPLIT_MANIFEST = "split_manifest.csv"
EVALUATION_MANIFEST = "evaluation_manifest.json"


def prepared_frame(cohort: CohortTable, num_classes: int) -> pd.DataFrame:
    """Validated copy of the cohort frame (string ids, int labels, original row order)."""
    frame = cohort.frame.reset_index(drop=True).copy()
    missing = [c for c in COHORT_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"cohort frame is missing required columns {missing}")
    if cohort.label.num_classes != num_classes:
        raise ValueError(f"cohort label definition has {cohort.label.num_classes} classes, config expects {num_classes}")
    frame[PATIENT_ID] = frame[PATIENT_ID].astype(str)
    frame[STUDY_ID] = frame[STUDY_ID].astype(str)
    if frame[STUDY_ID].duplicated().any():
        raise ValueError("cohort has duplicate study_id values")
    labels = frame[LABEL]
    if labels.isna().any() or not (labels.astype(int) == labels).all():
        raise ValueError("cohort labels must be integers")
    frame[LABEL] = labels.astype(int)
    if frame[LABEL].min() < 0 or frame[LABEL].max() >= num_classes:
        raise ValueError(f"cohort labels must be within 0..{num_classes - 1}")
    return frame


def cohort_csv_text(frame: pd.DataFrame) -> str:
    """Canonical ``cohort.csv`` (``oncotype_score`` excluded, ``series_uids`` ``|``-joined)."""
    out = frame.drop(columns=[ONCOTYPE_SCORE], errors="ignore").copy()
    if SERIES_UIDS in out.columns:
        out[SERIES_UIDS] = out[SERIES_UIDS].map(
            lambda v: "|".join(str(x) for x in v) if isinstance(v, (list, tuple)) else ("" if v is None else str(v)))
    buffer = io.StringIO()
    out.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def text_fingerprint(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def directory_fingerprints(directory: Path) -> dict[str, str]:
    """``{relative posix path: sha256}`` for every file below ``directory``."""
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): file_fingerprint(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def check_schema(manifest: Mapping[str, Any], path: Path) -> None:
    version = manifest.get("schema_version")
    if not isinstance(version, int) or version != MANIFEST_SCHEMA_VERSION:
        raise ArtifactMismatch(f"{path}: unsupported manifest schema_version {version!r} "
                               f"(this code reads {MANIFEST_SCHEMA_VERSION})")
    cfg_version = manifest.get("config_schema_version", CONFIG_SCHEMA_VERSION)
    if cfg_version != CONFIG_SCHEMA_VERSION:
        raise ArtifactMismatch(f"{path}: config_schema_version {cfg_version!r} != {CONFIG_SCHEMA_VERSION}")


def verify_train_manifest(path: Path, *, train_id: str, label_definition: Mapping[str, Any],
                          input_fingerprints: Mapping[str, str], cohort_fingerprint: str,
                          split_hash: str | None) -> dict[str, Any]:
    """Raise :class:`ArtifactMismatch` listing every identity difference."""
    manifest = read_json(path)
    check_schema(manifest, path)
    problems = []
    if manifest.get("train_id") != train_id:
        problems.append(f"train_id {manifest.get('train_id')!r} != {train_id!r}")
    if manifest.get("label_definition") != dict(label_definition):
        problems.append("label definition differs")
    if dict(manifest.get("input_fingerprints") or {}) != dict(input_fingerprints):
        changed = sorted(k for k in set(manifest.get("input_fingerprints") or {}) | set(input_fingerprints)
                         if (manifest.get("input_fingerprints") or {}).get(k) != input_fingerprints.get(k))
        problems.append(f"input data fingerprints differ: {changed}")
    if manifest.get("cohort_fingerprint") != cohort_fingerprint:
        problems.append("cohort fingerprint differs (cohort rows/labels/columns changed)")
    if split_hash is not None and manifest.get("split_hash") != split_hash:
        problems.append(f"split_hash {manifest.get('split_hash')!r} != {split_hash!r}")
    if problems:
        raise ArtifactMismatch(f"{path} belongs to another configuration/data: " + "; ".join(problems))
    return manifest


def verify_fold(fold_dir: Path, *, train_id: str, split_hash: str, fold: int, check_files: bool) -> dict[str, Any]:
    path = fold_dir / FOLD_MANIFEST
    if not path.is_file():
        raise ArtifactMismatch(f"{fold_dir}: COMPLETE without {FOLD_MANIFEST}")
    manifest = read_json(path)
    check_schema(manifest, path)
    if manifest.get("train_id") != train_id or manifest.get("split_hash") != split_hash or manifest.get("fold") != fold:
        raise ArtifactMismatch(f"{path}: fold artifacts belong to train_id={manifest.get('train_id')!r}, "
                               f"split_hash={manifest.get('split_hash')!r}, fold={manifest.get('fold')!r}")
    if check_files:
        for branch, files in (manifest.get("branch_artifacts") or {}).items():
            actual = directory_fingerprints(fold_dir / branch)
            if actual != files:
                diff = sorted(k for k in set(actual) | set(files) if actual.get(k) != files.get(k))
                raise ArtifactMismatch(f"{fold_dir / branch}: artifacts changed since training: {diff[:5]}")
    return manifest
