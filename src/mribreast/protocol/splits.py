"""Patient-grouped split plan: outer StratifiedGroupKFold + inner fit/calibration split.

Compatibility guarantees (see docs/protocol.md):

* :func:`make_inner_calibration_split` is a verbatim port of the legacy
  ``mriBreastDuke/threshold_tuning.py::make_inner_calibration_split`` and returns
  bit-identical indices for identical inputs.
* :func:`make_split_plan` reproduces the legacy fold loop
  (``n_fold_cv_run.py::run_5fold_cv`` and
  ``validate_configurable_imaging_features_fusion.py::run_validation``):
  ``StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)``
  over the cohort rows in their given order, grouped by ``patient_id``; the
  inner split of fold ``k`` uses ``random_state = seed + k`` on the outer-train
  rows (kept in cohort order).

The plan is a long table (one row per (fold, study)) with columns
:data:`SPLIT_COLUMNS`. It is frozen on disk (``split_manifest.csv``) and its
content hash (:func:`split_hash`) identifies it in every manifest.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from mribreast.artifacts import write_text_atomic
from mribreast.contracts import FOLD, LABEL, PATIENT_ID, ROLE, ROLES, STUDY_ID

SPLIT_META_COLUMNS = ("requested_inner_folds", "effective_inner_folds", "outer_seed", "inner_seed")
SPLIT_COLUMNS = (FOLD, PATIENT_ID, STUDY_ID, LABEL, ROLE) + SPLIT_META_COLUMNS
SELECTION_SPLITS = ("shared", "separate", "outer")
#: Offset added to ``seed + fold`` for the second (selection/threshold) split in ``separate`` mode.
SEPARATE_SEED_OFFSET = 1000
#: Role order inside one fold of the written manifest.
_ROLE_ORDER = {role: index for index, role in enumerate(("fit", "calibration", "selection", "threshold", "outer_eval"))}


class SplitError(ValueError):
    """The cohort cannot be split under the requested protocol, or a plan is invalid."""


# --------------------------------------------------------------------------- legacy port
def make_inner_calibration_split(labels, groups=None, requested_folds=5, random_state=42):
    """Split outer-training rows into fit and calibration indices (legacy, bit-identical).

    The first deterministic fold is held out. With ``groups`` no group crosses
    the split. Returns ``(fit_indices, calibration_indices, effective_folds)``
    where ``effective_folds = min(requested_folds, smallest #groups per class)``.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least 2.")

    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("Inner calibration requires at least two classes.")

    placeholder = np.zeros(labels.size, dtype=np.uint8)
    if groups is None:
        class_counts = np.array([np.sum(labels == value) for value in classes])
        effective_folds = min(int(requested_folds), int(class_counts.min()))
        if effective_folds < 2:
            raise ValueError("Every class needs at least two rows for inner calibration.")
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        fit_indices, calibration_indices = next(splitter.split(placeholder, labels))
    else:
        groups = np.asarray(groups)
        if groups.shape != labels.shape:
            raise ValueError("groups must align with labels.")
        if pd.isna(groups).any():
            raise ValueError("groups cannot contain missing values.")

        group_labels = pd.DataFrame({"group": groups, "label": labels})
        labels_per_group = group_labels.groupby("group", sort=False)["label"].nunique()
        if (labels_per_group > 1).any():
            raise ValueError("Each group must have exactly one label.")
        unique_groups = group_labels.drop_duplicates()
        groups_per_class = np.array(
            [unique_groups.loc[unique_groups["label"] == value, "group"].nunique() for value in classes]
        )
        effective_folds = min(int(requested_folds), int(groups_per_class.min()))
        if effective_folds < 2:
            raise ValueError("Every class needs at least two patient groups for inner calibration.")
        splitter = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        fit_indices, calibration_indices = next(splitter.split(placeholder, labels, groups))
        overlap = set(groups[fit_indices]).intersection(groups[calibration_indices])
        if overlap:
            raise RuntimeError("Patient-group leakage detected in inner calibration.")

    for name, indices in (("fit", fit_indices), ("calibration", calibration_indices)):
        present_classes = np.unique(labels[indices])
        if not np.array_equal(present_classes, classes):
            raise ValueError(
                f"The inner {name} split is missing a class: "
                f"received {present_classes.tolist()}, expected {classes.tolist()}."
            )

    return fit_indices, calibration_indices, effective_folds


# --------------------------------------------------------------------------- plan
def _check_cohort(frame: pd.DataFrame, num_folds: int) -> np.ndarray:
    missing = [c for c in (PATIENT_ID, STUDY_ID, LABEL) if c not in frame.columns]
    if missing:
        raise SplitError(f"cohort frame is missing columns {missing}")
    if len(frame) == 0:
        raise SplitError("cohort frame is empty")
    for column in (PATIENT_ID, STUDY_ID, LABEL):
        if frame[column].isna().any():
            raise SplitError(f"cohort column {column!r} contains missing values")
    if frame[STUDY_ID].astype(str).duplicated().any():
        dup = frame.loc[frame[STUDY_ID].astype(str).duplicated(), STUDY_ID].astype(str).tolist()[:5]
        raise SplitError(f"duplicate study_id values in cohort: {dup}")
    labels = frame[LABEL].to_numpy()
    integer = labels.astype(np.int64)
    if not np.array_equal(labels, integer):
        raise SplitError("labels must be integers 0..C-1")
    classes = np.unique(integer)
    if classes.size < 2 or not np.array_equal(classes, np.arange(classes.size)):
        raise SplitError(f"labels must be contiguous 0..C-1 with >= 2 classes, got {classes.tolist()}")
    per_patient = frame.groupby(frame[PATIENT_ID].astype(str), sort=False)[LABEL].nunique()
    if (per_patient > 1).any():
        raise SplitError(f"patients with more than one label: {per_patient[per_patient > 1].index.tolist()[:5]}")
    groups_per_class = (frame.assign(_p=frame[PATIENT_ID].astype(str))[["_p", LABEL]].drop_duplicates()
                        .groupby(LABEL)["_p"].nunique().reindex(classes, fill_value=0).to_numpy())
    if np.any(groups_per_class < num_folds):
        raise SplitError(f"every class needs >= num_folds distinct patients; counts={groups_per_class.tolist()}, "
                         f"num_folds={num_folds}")
    return integer


def make_split_plan(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Build the frozen split plan (architecture §4) for ``frame`` (one row per study).

    Uses ``cfg`` keys ``num_folds``, ``threshold_calibration_folds``, ``seed``,
    ``selection_split``. Row order of ``frame`` matters (legacy splitters are
    order dependent); grouping is always by ``patient_id`` so that every study,
    phase, slice and variant of one person stays on one side of every split.
    """
    num_folds = int(cfg["num_folds"])
    requested = int(cfg["threshold_calibration_folds"])
    seed = int(cfg["seed"])
    mode = cfg.get("selection_split", "shared")
    if mode not in SELECTION_SPLITS:
        raise SplitError(f"unknown selection_split {mode!r}")
    if num_folds < 2:
        raise SplitError("num_folds must be at least 2")
    frame = frame.reset_index(drop=True)
    labels = _check_cohort(frame, num_folds)
    groups = frame[PATIENT_ID].astype(str).to_numpy()
    studies = frame[STUDY_ID].astype(str).to_numpy()

    outer = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    parts: list[pd.DataFrame] = []
    for fold, (train_idx, eval_idx) in enumerate(outer.split(frame, labels, groups), start=1):
        roles: list[tuple[str, np.ndarray]] = []
        inner_seed = seed + fold
        if mode == "outer":
            roles.append(("fit", train_idx))
            req, eff, inner_seed = 0, 0, -1
        else:
            try:
                fit_local, cal_local, eff = make_inner_calibration_split(
                    labels[train_idx], groups=groups[train_idx], requested_folds=requested, random_state=inner_seed)
            except ValueError as exc:
                raise SplitError(f"fold {fold}: inner fit/calibration split failed: {exc}") from exc
            req = requested
            fit_idx, cal_idx = train_idx[fit_local], train_idx[cal_local]
            roles.append(("fit", fit_idx))
            if mode == "shared":
                roles.append(("calibration", cal_idx))
            else:
                try:
                    sel_local, thr_local, _ = make_inner_calibration_split(
                        labels[cal_idx], groups=groups[cal_idx], requested_folds=2,
                        random_state=inner_seed + SEPARATE_SEED_OFFSET)
                except ValueError as exc:
                    raise SplitError(
                        f"fold {fold}: selection_split=separate needs >= 2 calibration patients per class "
                        f"to split selection/threshold ({exc}); lower threshold_calibration_folds or use shared"
                    ) from exc
                roles.append(("selection", cal_idx[sel_local]))
                roles.append(("threshold", cal_idx[thr_local]))
        roles.append(("outer_eval", eval_idx))
        for role, idx in roles:
            parts.append(pd.DataFrame({
                FOLD: fold, PATIENT_ID: groups[idx], STUDY_ID: studies[idx], LABEL: labels[idx], ROLE: role,
                "requested_inner_folds": req, "effective_inner_folds": eff, "outer_seed": seed,
                "inner_seed": inner_seed,
            }))
    plan = pd.concat(parts, ignore_index=True)[list(SPLIT_COLUMNS)]
    validate_split_plan(plan, frame)
    return plan


# --------------------------------------------------------------------------- validation
def validate_split_plan(plan: pd.DataFrame, frame: pd.DataFrame | None = None, *,
                        check_labels: bool = True) -> None:
    """Raise :class:`SplitError` unless the plan is leakage-free and complete.

    * columns present, roles known, each (fold, study) exactly once;
    * patients disjoint across roles inside each fold;
    * each study is ``outer_eval`` in exactly one fold, outer-eval folds disjoint by patient;
    * every class present in ``fit`` and in each inner held-out role
      (``calibration``/``selection``/``threshold``) - required by threshold tuning;
    * one label per patient;
    * with ``frame``: the plan covers exactly the cohort studies with the same
      patient ids (and the same labels when ``check_labels``).
    """
    missing = [c for c in SPLIT_COLUMNS if c not in plan.columns]
    if missing:
        raise SplitError(f"split plan is missing columns {missing}")
    bad_roles = set(plan[ROLE]).difference(ROLES)
    if bad_roles:
        raise SplitError(f"unknown roles {sorted(bad_roles)}")
    pid = plan[PATIENT_ID].astype(str)
    sid = plan[STUDY_ID].astype(str)
    if pd.DataFrame({"f": plan[FOLD], "s": sid}).duplicated().any():
        raise SplitError("a study appears twice in one fold")
    per_patient = plan.groupby(pid)[LABEL].nunique()
    if (per_patient > 1).any():
        raise SplitError(f"patients with more than one label: {per_patient[per_patient > 1].index.tolist()[:5]}")
    classes = np.sort(plan[LABEL].astype(int).unique())
    all_studies = set(sid)
    folds = sorted(plan[FOLD].unique())
    if folds != list(range(1, len(folds) + 1)):
        raise SplitError(f"folds must be 1..K, got {folds}")
    eval_seen: dict[str, int] = {}
    for fold in folds:
        part = plan[plan[FOLD] == fold]
        if set(part[STUDY_ID].astype(str)) != all_studies:
            raise SplitError(f"fold {fold} does not assign every study exactly once")
        role_patients = {role: set(g[PATIENT_ID].astype(str)) for role, g in part.groupby(ROLE)}
        names = sorted(role_patients)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = role_patients[a] & role_patients[b]
                if overlap:
                    raise SplitError(f"fold {fold}: patients {sorted(overlap)[:5]} in both {a} and {b}")
        if "fit" not in role_patients or "outer_eval" not in role_patients:
            raise SplitError(f"fold {fold} needs fit and outer_eval roles")
        for role in ("fit", "calibration", "selection", "threshold"):
            if role not in role_patients:
                continue
            present = np.sort(part.loc[part[ROLE] == role, LABEL].astype(int).unique())
            if not np.array_equal(present, classes):
                raise SplitError(f"fold {fold}: role {role} has classes {present.tolist()}, expected {classes.tolist()}")
        for study in part.loc[part[ROLE] == "outer_eval", STUDY_ID].astype(str):
            if study in eval_seen:
                raise SplitError(f"study {study} is outer_eval in folds {eval_seen[study]} and {fold}")
            eval_seen[study] = fold
    if set(eval_seen) != all_studies:
        raise SplitError("some studies are never in outer_eval")
    if frame is not None:
        cohort = pd.DataFrame({"s": frame[STUDY_ID].astype(str), "p": frame[PATIENT_ID].astype(str),
                               "y": frame[LABEL].astype(int)}).set_index("s")
        if set(cohort.index) != all_studies:
            extra = sorted(all_studies - set(cohort.index))[:5]
            absent = sorted(set(cohort.index) - all_studies)[:5]
            raise SplitError(f"split plan does not match cohort studies (not in cohort: {extra}; not in plan: {absent})")
        # Every fold row is checked (not only fold 1): a hand-edited/stale plan could otherwise relabel a
        # patient in a later fold and put the same person into fit and outer_eval.
        joined = cohort.loc[plan[STUDY_ID].astype(str)]
        wrong_patient = joined["p"].to_numpy() != plan[PATIENT_ID].astype(str).to_numpy()
        if wrong_patient.any():
            bad = plan.loc[wrong_patient, [FOLD, STUDY_ID]].head(5).to_dict("records")
            raise SplitError(f"split plan patient ids differ from cohort: {bad}")
        if check_labels and not np.array_equal(joined["y"].to_numpy(), plan[LABEL].astype(int).to_numpy()):
            raise SplitError("split plan labels differ from cohort labels")


def check_plan_classes(plan: pd.DataFrame, frame: pd.DataFrame) -> None:
    """Re-run the class-presence checks with the *current* cohort labels (frozen plans)."""
    relabelled = plan.drop(columns=[LABEL]).merge(
        pd.DataFrame({STUDY_ID: frame[STUDY_ID].astype(str), LABEL: frame[LABEL].astype(int)}),
        on=STUDY_ID, how="left", validate="many_to_one")
    validate_split_plan(relabelled[list(SPLIT_COLUMNS)], frame, check_labels=True)


# --------------------------------------------------------------------------- persistence
def _canonical_csv(plan: pd.DataFrame) -> str:
    plan = plan[list(SPLIT_COLUMNS)].copy()
    for column in (FOLD, LABEL) + SPLIT_META_COLUMNS:
        plan[column] = plan[column].astype(np.int64)
    plan[PATIENT_ID] = plan[PATIENT_ID].astype(str)
    plan[STUDY_ID] = plan[STUDY_ID].astype(str)
    buffer = io.StringIO()
    plan.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def split_hash(plan: pd.DataFrame) -> str:
    """``sha256:<hex>`` of the canonical CSV; equals ``file_fingerprint(split_manifest.csv)``."""
    return "sha256:" + hashlib.sha256(_canonical_csv(plan).encode("utf-8")).hexdigest()


def save_split_plan(plan: pd.DataFrame, path: str | Path) -> str:
    """Write the canonical CSV atomically and return its :func:`split_hash`."""
    text = _canonical_csv(plan)
    write_text_atomic(path, text)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split_plan(path: str | Path) -> pd.DataFrame:
    plan = pd.read_csv(path, dtype={PATIENT_ID: str, STUDY_ID: str, ROLE: str}, keep_default_na=False)
    missing = [c for c in SPLIT_COLUMNS if c not in plan.columns]
    if missing:
        raise SplitError(f"{path}: split manifest is missing columns {missing}")
    for column in (FOLD, LABEL) + SPLIT_META_COLUMNS:
        plan[column] = plan[column].astype(np.int64)
    return plan[list(SPLIT_COLUMNS)]


# --------------------------------------------------------------------------- helpers
def fold_numbers(plan: pd.DataFrame) -> list[int]:
    return [int(f) for f in sorted(plan[FOLD].unique())]


def role_rows(plan: pd.DataFrame, fold: int, roles: Iterable[str]) -> pd.DataFrame:
    roles = tuple(roles)
    part = plan[(plan[FOLD] == fold) & plan[ROLE].isin(roles)]
    return part


def role_frame(plan: pd.DataFrame, frame: pd.DataFrame, fold: int, roles: Iterable[str]) -> pd.DataFrame:
    """Cohort rows (all columns) of ``fold`` with one of ``roles``, in cohort order."""
    wanted = set(role_rows(plan, fold, roles)[STUDY_ID].astype(str))
    mask = frame[STUDY_ID].astype(str).isin(wanted).to_numpy()
    return frame.loc[mask].reset_index(drop=True)


def selection_roles(mode: str) -> tuple[str, ...]:
    """Roles a branch may use for checkpoint selection / early stopping."""
    return {"shared": ("calibration",), "separate": ("selection",), "outer": ("outer_eval",)}[mode]


def threshold_roles(mode: str) -> tuple[str, ...]:
    """Roles on which decision thresholds are tuned."""
    return {"shared": ("calibration",), "separate": ("threshold",), "outer": ("outer_eval",)}[mode]


def inner_heldout_roles() -> tuple[str, ...]:
    return ("calibration", "selection", "threshold")


def fold_meta(plan: pd.DataFrame, fold: int) -> dict[str, int]:
    part = plan[plan[FOLD] == fold]
    row = part.iloc[0]
    counts = part[ROLE].value_counts().to_dict()
    return {"requested_inner_folds": int(row["requested_inner_folds"]),
            "effective_inner_folds": int(row["effective_inner_folds"]),
            "outer_seed": int(row["outer_seed"]), "inner_seed": int(row["inner_seed"]),
            **{f"{role}_rows": int(counts.get(role, 0)) for role in ROLES}}


def sort_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Stable fold/role ordering (keeps within-role order)."""
    key = plan[ROLE].map(_ROLE_ORDER)
    return plan.assign(_k=key).sort_values([FOLD, "_k"], kind="stable").drop(columns="_k").reset_index(drop=True)
