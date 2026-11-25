"""Per-study box decisions for ``box_mode`` (missing-box policy + audit), mirroring the mask policies.

A study is *missing* a box whenever its :func:`~mribreast.imaging.annotations.annotation_coverage` status
is not ``eligible`` (missing_box, ambiguous_patient_studies, missing_series, missing_series_file,
incompatible_series_geometry, out_of_volume_box). ``box_missing_policy`` then decides:

* ``error``       raise :class:`BoxPolicyError` (audit-only callers get action ``error`` instead);
* ``full_image``  keep the study; the mode builder gets ``box=None`` (roi_crop -> full volume,
                  box_channel -> all-zero channel, dual_stream -> local stream = full volume);
* ``drop``        the study gets no items, so prediction returns a NaN row (as mask ``drop``).

Boxes are resolved for EVERY frame the branch sees (fit, selection and prediction frames):
box modes need a box at inference time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from mribreast.contracts import PATIENT_ID, STUDY_ID
from mribreast.imaging.annotations import Box, annotation_coverage

BOX_MODES = ("none", "roi_crop", "box_channel", "dual_stream")
BOX_POLICIES = ("error", "full_image", "drop")
BOX_AUDIT_COLUMNS = ("patient_id", "study_id", "box_mode", "status", "action", "box_start", "box_end",
                     "source_shape", "detail")


class BoxPolicyError(ValueError):
    """A study has no eligible annotation box and ``box_missing_policy`` is ``error``."""


@dataclass
class BoxDecision:
    patient_id: str
    study_id: str
    status: str                              # annotation_coverage status (eligible = box available)
    action: str                              # use | full_image | drop | error
    box: Box | None = None
    source_shape: tuple[int, ...] | None = None
    detail: str = ""
    box_mode: str = ""

    def audit_row(self) -> dict[str, Any]:
        return {"patient_id": self.patient_id, "study_id": self.study_id, "box_mode": self.box_mode,
                "status": self.status, "action": self.action,
                "box_start": " ".join(str(v) for v in self.box[0]) if self.box else "",
                "box_end": " ".join(str(v) for v in self.box[1]) if self.box else "",
                "source_shape": " ".join(str(v) for v in self.source_shape) if self.source_shape else "",
                "detail": self.detail}


def decide_boxes(frame: pd.DataFrame, boxes: Mapping[str, Box], image_root: str | os.PathLike, *,
                 policy: str, box_mode: str = "", raise_errors: bool = True
                 ) -> tuple[dict[str, BoxDecision], pd.DataFrame]:
    """Coverage for ``frame`` + one :class:`BoxDecision` per study (keyed by ``study_id``).

    Returns ``(decisions, coverage_report)``. With ``policy='error'`` the first missing study raises
    :class:`BoxPolicyError` unless ``raise_errors=False`` (cohort audits), which reports action ``error``.
    """
    if policy not in BOX_POLICIES:
        raise ValueError(f"unknown box_missing_policy {policy!r}")
    matched, report = annotation_coverage(frame, boxes, image_root)
    decisions: dict[str, BoxDecision] = {}
    for row in report.itertuples(index=False):
        study = str(getattr(row, STUDY_ID))
        if not study:                       # unmatched_annotation rows: patients not in this frame
            continue
        patient, status, detail = str(getattr(row, PATIENT_ID)), str(row.status), str(row.detail)
        if status == "eligible":
            box, source_shape = matched[study]
            decisions[study] = BoxDecision(patient, study, status, "use", box,
                                           tuple(int(v) for v in source_shape), detail, box_mode)
            continue
        decision = BoxDecision(patient, study, status, policy, None, None, detail, box_mode)
        if policy == "error":
            if raise_errors:
                raise BoxPolicyError(f"no eligible annotation box for study {study} (patient {patient}): "
                                     f"{status} {detail}".rstrip())
            decision.action = "error"
        decisions[study] = decision
    return decisions, report


def box_audit_frame(decisions: Sequence[BoxDecision]) -> pd.DataFrame:
    return pd.DataFrame([d.audit_row() for d in decisions], columns=list(BOX_AUDIT_COLUMNS))


def coverage_counts(report: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in report["status"].value_counts().items()} if len(report) else {}


__all__ = ["BOX_MODES", "BOX_POLICIES", "BOX_AUDIT_COLUMNS", "BoxPolicyError", "BoxDecision", "decide_boxes",
           "box_audit_frame", "coverage_counts"]
