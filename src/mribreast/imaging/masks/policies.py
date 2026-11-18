"""Mask modes, missing/empty/invalid policies and the mask audit.

Modes (``mask_mode``):

* ``none``          image only; masks may still drive slice selection.
* ``roi_crop``      crop the ROI bbox (+``roi_margin`` x extent per side) from image
                    and mask with shared params, then resize both to the target grid.
* ``mask_channel``  append the binary ROI mask (resized nearest) as an extra channel.
* ``foreground_only`` zero image voxels outside the ROI (after the joint resize).

``full_image`` fallback (missing/empty mask with policy ``full_image``): the whole
image is treated as ROI -> no crop, no zeroing, and for ``mask_channel`` an all-ones
channel, so the expected channel count is preserved.

Invalid masks (key/geometry mismatch, undeclared class codes, wrong source kind)
always follow ``invalid_policy`` (default ``error``; there is no config field yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mribreast.contracts import Geometry, ImageKey, MaskProvider, MaskRecord
from mribreast.imaging.transforms import TransformLog, joint_crop, joint_resize, roi_bbox

MASK_MODES = ("none", "roi_crop", "mask_channel", "foreground_only")
MASK_POLICIES = ("error", "full_image", "drop")
AUDIT_COLUMNS = ("patient_id", "study_id", "series_id", "slice_id", "purpose", "status", "action", "source",
                 "model_id", "model_version", "training_provenance", "fingerprint", "detail")


class MaskPolicyError(ValueError):
    """A mask is missing/empty/invalid and the configured policy is ``error``."""


@dataclass
class MaskDecision:
    key: ImageKey
    status: str          # ok | missing | empty | invalid_key | invalid_geometry | invalid_codes | source_mismatch
    action: str          # use | full_image | drop
    record: MaskRecord | None = None
    detail: str = ""
    purpose: str = "mask_mode"

    def audit_row(self) -> dict[str, Any]:
        rec = self.record
        return {"patient_id": self.key.patient_id, "study_id": self.key.study_id, "series_id": self.key.series_id,
                "slice_id": self.key.slice_id, "purpose": self.purpose, "status": self.status, "action": self.action,
                "source": rec.source if rec else "", "model_id": rec.model_id if rec else "",
                "model_version": rec.model_version if rec else "",
                "training_provenance": rec.training_provenance if rec else "",
                "fingerprint": rec.fingerprint if rec else "", "detail": self.detail}


def _apply_policy(decision: MaskDecision, policy: str) -> MaskDecision:
    if policy not in MASK_POLICIES:
        raise ValueError(f"unknown mask policy {policy!r}")
    if policy == "error":
        raise MaskPolicyError(f"mask {decision.status} for {decision.key.as_tuple()}: {decision.detail}")
    decision.action = policy
    return decision


def check_mask(record: MaskRecord, key: ImageKey, expected: Geometry | None,
               mask_kind: str | None = None) -> tuple[str, str]:
    """Return ``(status, detail)`` for a fetched record (``ok`` or an ``invalid_*`` status)."""
    if record.key.as_tuple() != key.as_tuple():
        return "invalid_key", f"mask key {record.key.as_tuple()} != image key {key.as_tuple()}"
    if expected is not None and not record.geometry.matches(expected):
        return "invalid_geometry", f"mask grid {record.geometry.shape} / affine differ from image grid {expected.shape}"
    if tuple(np.asarray(record.array).shape) != tuple(record.geometry.shape):
        return "invalid_geometry", "mask array shape differs from its declared geometry"
    codes = set(np.unique(np.asarray(record.array)).astype(int).tolist())
    undeclared = codes.difference(int(c) for c in record.class_codes)
    if undeclared:
        return "invalid_codes", f"undeclared class codes {sorted(undeclared)}"
    if mask_kind is not None and record.source not in (mask_kind, "synthetic"):
        return "source_mismatch", f"mask source {record.source!r} but mask_kind={mask_kind!r}"
    return "ok", ""


def resolve_mask(provider: MaskProvider, key: ImageKey, *, fold: int | None, expected: Geometry | None,
                 roi_classes: Sequence[int], missing_policy: str, empty_policy: str,
                 invalid_policy: str = "error", mask_kind: str | None = None,
                 purpose: str = "mask_mode") -> MaskDecision:
    """Fetch, validate and apply the configured policy (raises ``MaskPolicyError`` for ``error``)."""
    record = provider.get(key, fold)
    if record is None:
        return _apply_policy(MaskDecision(key, "missing", "", None, "no mask for key", purpose), missing_policy)
    status, detail = check_mask(record, key, expected, mask_kind)
    if status != "ok":
        return _apply_policy(MaskDecision(key, status, "", record, detail, purpose), invalid_policy)
    if not np.isin(np.asarray(record.array), np.asarray(list(roi_classes))).any():
        return _apply_policy(MaskDecision(key, "empty", "", record, f"no voxels of ROI classes {list(roi_classes)}",
                                          purpose), empty_policy)
    return MaskDecision(key, "ok", "use", record, "", purpose)


def roi_array(mask: np.ndarray, roi_classes: Sequence[int]) -> np.ndarray:
    return np.isin(np.asarray(mask), np.asarray(list(roi_classes))).astype(np.float32)


def apply_mask_mode(image: np.ndarray, mask: np.ndarray | None, mode: str, *, roi_classes: Sequence[int],
                    margin: float, out_size: Sequence[int], log: TransformLog | None = None) -> np.ndarray:
    """Apply ``mode`` to ``image [C, *S]`` with ``mask [*S]`` (``None`` = full-image fallback).

    Output is ``[C (+1 for mask_channel), *out_size]``.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"unknown mask_mode {mode!r}")
    log = log if log is not None else []
    if mode == "none" or mask is None:
        image, _ = joint_resize(image, None, out_size, log)
        if mode == "mask_channel":
            image = np.concatenate([image, np.ones((1,) + image.shape[1:], dtype=np.float32)], axis=0)
            log.append({"op": "mask_channel", "fallback": "full_image_ones"})
        elif mode != "none":
            log.append({"op": mode, "fallback": "full_image"})
        return image
    if mode == "roi_crop":
        bbox = roi_bbox(mask, roi_classes, margin)
        if bbox is not None:
            image, mask = joint_crop(image, mask, bbox[0], bbox[1], log)
        image, mask = joint_resize(image, mask, out_size, log)
        return image
    image, mask = joint_resize(image, mask, out_size, log)
    roi = roi_array(mask, roi_classes)
    if mode == "mask_channel":
        log.append({"op": "mask_channel", "roi_classes": [int(c) for c in roi_classes]})
        return np.concatenate([image, roi[None]], axis=0).astype(np.float32)
    log.append({"op": "foreground_only", "roi_classes": [int(c) for c in roi_classes]})
    return (image * roi[None]).astype(np.float32)


def audit_frame(decisions: Sequence[MaskDecision]) -> pd.DataFrame:
    return pd.DataFrame([d.audit_row() for d in decisions], columns=list(AUDIT_COLUMNS))


def summarize_mask_audit(audit: pd.DataFrame, n_studies_in: int, n_studies_out: int, n_images_in: int,
                         n_images_out: int) -> pd.DataFrame:
    """One-row cohort-change summary (denominators for metrics when ``drop`` removes samples)."""
    row: dict[str, Any] = {"n_studies_in": n_studies_in, "n_studies_out": n_studies_out,
                           "n_studies_dropped": n_studies_in - n_studies_out, "n_images_in": n_images_in,
                           "n_images_out": n_images_out, "n_images_dropped": n_images_in - n_images_out}
    for status, count in audit["status"].value_counts().items() if len(audit) else []:
        row[f"status_{status}"] = int(count)
    for action, count in audit["action"].value_counts().items() if len(audit) else []:
        row[f"action_{action}"] = int(count)
    return pd.DataFrame([row])

