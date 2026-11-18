"""Versioned allowed-predictor policy (which tabular columns may be used at a prediction time).

The policy is a JSON document (packaged default: ``allowed_predictors_v1.json``;
``cfg["allowed_predictors_file"]`` overrides it).  Every tabular predictor that
reaches a model must

1. be allowed for the configured ``prediction_time`` by its feature group
   (exact names for clinical/density, a name prefix for imaging families), and
2. not match any forbidden name / token / substring (oncotype score, label,
   pathology, receptors, stage, grade, therapy, outcome ...).

A column that fails either rule raises :class:`PredictorPolicyError`.  The
policy id + version are written into cohort notes and ``tabular_branch.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from mribreast.contracts import LABEL, ONCOTYPE_SCORE, PATIENT_ID, SERIES_UIDS, STUDY_ID

DEFAULT_POLICY_FILE = "allowed_predictors_v1.json"
RESERVED_COLUMNS = (PATIENT_ID, STUDY_ID, LABEL, SERIES_UIDS, ONCOTYPE_SCORE)


class PredictorPolicyError(ValueError):
    """A configured predictor is not allowed (post-outcome, label-leaking or unknown)."""


@dataclass(frozen=True)
class GroupRule:
    columns: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    sensitive: bool = False

    def allows(self, column: str) -> bool:
        return column in self.columns or any(column.startswith(p) for p in self.prefixes)


@dataclass(frozen=True)
class PredictorPolicy:
    policy_id: str
    version: str
    prediction_times: Mapping[str, Mapping[str, GroupRule]]
    forbidden_exact: tuple[str, ...]
    forbidden_tokens: tuple[str, ...]
    forbidden_substrings: tuple[str, ...]
    source: str
    sha256: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def describe(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "version": self.version, "source": self.source, "sha256": self.sha256}

    # ------------------------------------------------------------------
    def forbidden_reason(self, column: str) -> str | None:
        lowered = str(column).lower()
        if column in RESERVED_COLUMNS or lowered in self.forbidden_exact:
            return "reserved/forbidden column name"
        tokens = [t for t in re.split(r"[^a-z0-9]+", lowered) if t]
        hit = sorted(set(tokens).intersection(self.forbidden_tokens))
        if hit:
            return f"forbidden token {hit}"
        for sub in self.forbidden_substrings:
            if sub in lowered:
                return f"forbidden substring '{sub}'"
        return None

    def group_of(self, column: str, prediction_time: str, include_sensitive: bool) -> str | None:
        groups = self.prediction_times.get(prediction_time)
        if groups is None:
            raise PredictorPolicyError(f"policy {self.policy_id} v{self.version} has no prediction_time '{prediction_time}'")
        for name, rule in groups.items():
            if rule.sensitive and not include_sensitive:
                continue
            if rule.allows(column):
                return name
        return None

    def check(self, columns: Iterable[str], *, prediction_time: str = "mri", include_sensitive: bool = False) -> dict[str, str]:
        """Return ``{column: policy group}`` or raise listing every rejected column."""
        problems: list[str] = []
        mapping: dict[str, str] = {}
        for column in columns:
            reason = self.forbidden_reason(column)
            if reason:
                problems.append(f"{column}: {reason}")
                continue
            group = self.group_of(column, prediction_time, include_sensitive)
            if group is None:
                sensitive_group = self.group_of(column, prediction_time, True)
                if sensitive_group is not None:
                    problems.append(f"{column}: sensitive predictor requires include_sensitive=true")
                else:
                    problems.append(f"{column}: not in the allowed list for prediction_time={prediction_time}")
                continue
            mapping[column] = group
        if problems:
            raise PredictorPolicyError(
                f"allowed-predictor policy {self.policy_id} v{self.version} rejected: " + "; ".join(problems))
        return mapping


def _rule(obj: Mapping[str, Any]) -> GroupRule:
    return GroupRule(columns=tuple(obj.get("columns", ())), prefixes=tuple(obj.get("prefixes", ())),
                     sensitive=bool(obj.get("sensitive", False)))


def parse_policy(payload: Mapping[str, Any], source: str = "<memory>") -> PredictorPolicy:
    for key in ("policy_id", "version", "prediction_times", "forbidden"):
        if key not in payload:
            raise PredictorPolicyError(f"policy {source} lacks '{key}'")
    major = str(payload["version"]).split(".")[0]
    if major != "1":
        raise PredictorPolicyError(f"unsupported policy major version {payload['version']} in {source}")
    times = {t: {g: _rule(r) for g, r in groups.items()} for t, groups in payload["prediction_times"].items()}
    forbidden = payload["forbidden"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return PredictorPolicy(
        policy_id=str(payload["policy_id"]), version=str(payload["version"]), prediction_times=times,
        forbidden_exact=tuple(s.lower() for s in forbidden.get("exact", ())),
        forbidden_tokens=tuple(s.lower() for s in forbidden.get("tokens", ())),
        forbidden_substrings=tuple(s.lower() for s in forbidden.get("substrings", ())),
        source=source, sha256="sha256:" + hashlib.sha256(canonical.encode()).hexdigest(), raw=payload)


def load_policy(path: str | Path | None = None) -> PredictorPolicy:
    """Load ``path`` or the packaged default policy."""
    if path:
        path = Path(path)
        with open(path, encoding="utf-8") as handle:
            return parse_policy(json.load(handle), source=str(path))
    text = resources.files("mribreast.data").joinpath(DEFAULT_POLICY_FILE).read_text(encoding="utf-8")
    return parse_policy(json.loads(text), source=f"package:mribreast.data/{DEFAULT_POLICY_FILE}")


def policy_from_cfg(cfg: Mapping[str, Any]) -> PredictorPolicy:
    return load_policy(cfg.get("allowed_predictors_file") or None)


def check_label_leakage(frame: pd.DataFrame, columns: Sequence[str], *, label: str = LABEL,
                        score: pd.Series | None = None, min_rows: int = 5) -> None:
    """Data-based guard: reject a predictor that reproduces the label or the score exactly."""
    targets: dict[str, pd.Series] = {}
    if label in frame:
        targets["label"] = frame[label]
    if score is not None:
        targets["oncotype_score"] = pd.Series(np.asarray(score), index=frame.index)
    bad = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        for name, target in targets.items():
            both = values.notna() & target.notna()
            if both.sum() >= min_rows and np.allclose(values[both].to_numpy(float), target[both].to_numpy(float)):
                bad.append(f"{column} == {name}")
    if bad:
        raise PredictorPolicyError("predictors reproduce the target: " + ", ".join(bad))
