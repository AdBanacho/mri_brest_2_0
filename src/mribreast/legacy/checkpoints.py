"""Legacy checkpoint filenames, ranking and fold-directory inventory.

Reproduces ``mriBreastDuke/checkpoint_selection.py`` (reference SHA
127a2604) bit-for-bit: the same regular expressions and the same 7-tuple sort
key.  Three filename eras exist:

* ``auc`` (current legacy fusion workflow, ``ModelCheckpoint`` filename
  ``best-{epoch:02d}-{val_auc_roc:.4f}-{val_balanced_accuracy:.4f}-{val_sensitivity:.4f}``
  with Lightning's ``auto_insert_metric_name``)::

      best-epoch=07-val_auc_roc=0.7500-val_balanced_accuracy=0.6058-val_sensitivity=0.2500.ckpt

* ``balanced_accuracy`` era::

      best-epoch=03-val_balanced_accuracy=0.61-val_sensitivity=0.70[-val_auc_roc=0.66].ckpt

* ``sensitivity`` era::

      best-epoch=03-val_sensitivity=0.70[-val_auc_roc=0.66].ckpt

An optional Lightning version suffix ``-v<N>`` is accepted.  Any other
``best-*.ckpt`` file ranks last.  Within an era, higher primary metric wins,
then the secondary metrics, then the epoch, then the file mtime.  An AUC-era
file always ranks above older eras so a stale checkpoint never overrides a
newer model trained in the same directory.

No torch import happens at module import; :func:`load_legacy_network_state`
imports it lazily.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

METRIC_VALUE_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
AUC_CHECKPOINT_PATTERN = re.compile(
    r"best-epoch=(?P<epoch>\d+)-"
    rf"val_auc_roc=(?P<auc>{METRIC_VALUE_PATTERN})-"
    rf"val_balanced_accuracy=(?P<balanced_accuracy>{METRIC_VALUE_PATTERN})-"
    rf"val_sensitivity=(?P<sensitivity>{METRIC_VALUE_PATTERN})"
    r"(?:-v\d+)?\.ckpt$"
)
BALANCED_CHECKPOINT_PATTERN = re.compile(
    r"best-epoch=(?P<epoch>\d+)-"
    rf"(?:val_balanced_accuracy=(?P<balanced_accuracy>{METRIC_VALUE_PATTERN})-)?"
    rf"val_sensitivity=(?P<sensitivity>{METRIC_VALUE_PATTERN})"
    rf"(?:-val_auc_roc=(?P<auc>{METRIC_VALUE_PATTERN}))?"
    r"(?:-v\d+)?\.ckpt$"
)
VERSION_SUFFIX = re.compile(r"-v(?P<version>\d+)\.ckpt$")

FORMAT_AUC = "auc"
FORMAT_BALANCED = "balanced_accuracy"
FORMAT_SENSITIVITY = "sensitivity"
FORMAT_UNKNOWN = "unknown"
CHECKPOINT_FORMATS = (FORMAT_AUC, FORMAT_BALANCED, FORMAT_SENSITIVITY, FORMAT_UNKNOWN)

#: Selection metric that produced the "best" file in each era (for migration notes).
FORMAT_SELECTION_METRIC = {
    FORMAT_AUC: "val_auc_roc (max)",
    FORMAT_BALANCED: "val_balanced_accuracy (max) - historical",
    FORMAT_SENSITIVITY: "val_sensitivity (max) - historical",
    FORMAT_UNKNOWN: "unknown",
}

NEG_INF = float("-inf")


@dataclass(frozen=True)
class LegacyCheckpointName:
    """Parsed legacy checkpoint filename."""

    name: str
    format: str
    epoch: int | None = None
    auc_roc: float | None = None
    balanced_accuracy: float | None = None
    sensitivity: float | None = None
    version: int | None = None

    @property
    def is_current_format(self) -> bool:
        return self.format == FORMAT_AUC


def parse_checkpoint_name(name: str) -> LegacyCheckpointName:
    """Parse a ``best-*.ckpt`` file name (basename or path) into its legacy era and metrics."""
    base = Path(name).name
    version_match = VERSION_SUFFIX.search(base)
    version = int(version_match.group("version")) if version_match else None
    match = AUC_CHECKPOINT_PATTERN.match(base)
    if match is not None:
        return LegacyCheckpointName(
            base, FORMAT_AUC, int(match.group("epoch")), float(match.group("auc")),
            float(match.group("balanced_accuracy")), float(match.group("sensitivity")), version)
    match = BALANCED_CHECKPOINT_PATTERN.match(base)
    if match is None:
        return LegacyCheckpointName(base, FORMAT_UNKNOWN, version=version)
    auc = match.group("auc")
    balanced = match.group("balanced_accuracy")
    return LegacyCheckpointName(
        base,
        FORMAT_BALANCED if balanced is not None else FORMAT_SENSITIVITY,
        int(match.group("epoch")),
        float(auc) if auc is not None else None,
        float(balanced) if balanced is not None else None,
        float(match.group("sensitivity")),
        version,
    )


def score_from_name(name: str, mtime: float) -> tuple:
    """Legacy ``checkpoint_score`` computed from a file name and an explicit mtime."""
    parsed = parse_checkpoint_name(name)
    if parsed.format == FORMAT_AUC:
        return (1, 2, parsed.auc_roc, parsed.balanced_accuracy, parsed.sensitivity, parsed.epoch, mtime)
    if parsed.format == FORMAT_UNKNOWN:
        return (0, 0, NEG_INF, NEG_INF, NEG_INF, -1, mtime)
    auc = parsed.auc_roc if parsed.auc_roc is not None else NEG_INF
    if parsed.format == FORMAT_SENSITIVITY:
        return (1, 0, parsed.sensitivity, auc, NEG_INF, parsed.epoch, mtime)
    return (1, 1, parsed.balanced_accuracy, parsed.sensitivity, auc, parsed.epoch, mtime)


def checkpoint_score(path: str | os.PathLike) -> tuple:
    """Exact equivalent of legacy ``checkpoint_selection.checkpoint_score(path)`` (reads mtime)."""
    path = Path(path)
    return score_from_name(path.name, path.stat().st_mtime)


def rank_checkpoints(paths: Iterable[str | os.PathLike]) -> list[Path]:
    """Best-to-worst order identical to legacy ``rank_checkpoints`` (stable ``sorted(reverse=True)``)."""
    return sorted((Path(p) for p in paths), key=checkpoint_score, reverse=True)


def find_best_checkpoint(checkpoint_dir: str | os.PathLike) -> Path:
    """Legacy ``validate_configurable_imaging_features_fusion.find_best_checkpoint``."""
    checkpoint_dir = Path(checkpoint_dir)
    candidates = list(checkpoint_dir.glob("best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best-*.ckpt checkpoint found in: {checkpoint_dir}")
    return rank_checkpoints(candidates)[0]


def legacy_fold_checkpoint_dir(checkpoint_root: str | os.PathLike, experiment_name: str, fold: int) -> Path:
    """``<checkpoint_root>/<experiment_name>/fold_<k>/checkpoints`` (both legacy training workflows)."""
    return Path(checkpoint_root) / experiment_name / f"fold_{fold}" / "checkpoints"


def resolve_historical_fold_dir(checkpoint_root: str | os.PathLike, model_name: str, fold: int) -> Path:
    """Candidate search of historical ``validate_best_checkpoints.resolve_fold_checkpoint_dir``."""
    root = Path(checkpoint_root)
    candidates = [root / model_name / f"fold_{fold}" / "checkpoints", root / f"fold_{fold}" / "checkpoints",
                  root / f"fold_{fold}"]
    if root.name == "checkpoints" and root.parent.name.startswith("fold_"):
        candidates.append(root.parent.parent / f"fold_{fold}" / "checkpoints")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# --------------------------------------------------------------------------- fold inventory
#: Files written by legacy ``n_fold_cv_run.run_5fold_cv`` into ``fold_k/checkpoints``.
LEGACY_FOLD_FILES = {
    "split_manifest": "inner_calibration_split.csv",
    "tabular_preprocessor": "tabular_preprocessor.joblib",
    "tabular_selector": "tabular_feature_selector.joblib",
    "feature_names": "tabular_feature_names.txt",
    "selected_feature_names": "tabular_selected_feature_names.txt",
    "lasso_report": "lasso_feature_selection.csv",
    "lasso_chart": "lasso_feature_importance.png",
    "annotation_coverage": "annotation_coverage.csv",
    "last_checkpoint": "last.ckpt",
}
#: Legacy file -> new path relative to ``<checkpoint_root>/<train_id>/fold_<k>/`` (docs/migration.md).
NEW_FOLD_PATHS = {
    "best_checkpoint": "image/best.ckpt",
    "last_checkpoint": "image/last.ckpt",
    "split_manifest": "../split_manifest.csv (fold rows, role fit/calibration; outer_eval rebuilt)",
    "tabular_preprocessor": "tabular/preprocessor.joblib",
    "tabular_selector": "tabular/selector.joblib",
    "tabular_model": "tabular/model.joblib",
    "feature_names": "tabular/feature_names.txt",
    "selected_feature_names": "tabular/selected_feature_names.txt",
    "lasso_report": "tabular/lasso_feature_selection.csv",
    "lasso_chart": "tabular/lasso_feature_importance.png",
    "annotation_coverage": "../annotation_coverage.csv",
}


@dataclass
class LegacyFoldInventory:
    """What a legacy ``fold_k/checkpoints`` directory contains and whether it can be migrated."""

    fold: int
    directory: Path
    exists: bool
    best_checkpoint: Path | None = None
    checkpoint_format: str = FORMAT_UNKNOWN
    files: dict[str, Path] = field(default_factory=dict)
    tabular_models: dict[str, Path] = field(default_factory=dict)  # feature_model -> <model>_model.joblib
    split_columns: tuple[str, ...] = ()
    reasons: list[str] = field(default_factory=list)

    @property
    def has_split_metadata(self) -> bool:
        return "split_manifest" in self.files and not any(r.startswith("split_manifest") for r in self.reasons)

    @property
    def requires_retraining(self) -> bool:
        return bool(self.reasons)

    def as_row(self) -> dict[str, Any]:
        return {
            "fold": self.fold, "directory": str(self.directory), "exists": self.exists,
            "best_checkpoint": _display(self.best_checkpoint) if self.best_checkpoint else "",
            "checkpoint_format": self.checkpoint_format,
            "has_split_metadata": self.has_split_metadata,
            "tabular_models": ",".join(sorted(self.tabular_models)),
            "requires_retraining": self.requires_retraining, "reasons": ";".join(self.reasons),
        }


def _display(path: Path) -> str:
    from mribreast.legacy._fs import display

    return display(path)


REQUIRED_SPLIT_COLUMNS = ("studyId", "split", "requested_inner_folds", "effective_inner_folds")


def inventory_fold(directory: str | os.PathLike, fold: int, *, expect_tabular: str | None = None,
                   expect_selector: bool = False) -> LegacyFoldInventory:
    """Inspect one legacy ``fold_k/checkpoints`` directory (read-only).

    ``reasons`` lists every condition that prevents importing the fold into the
    new protocol without retraining (see docs/migration.md, section 4).
    """
    from mribreast.legacy._fs import display, fs

    directory = fs(directory)
    inv = LegacyFoldInventory(fold=fold, directory=Path(display(directory)), exists=directory.is_dir())
    if not inv.exists:
        inv.reasons.append("fold_directory_missing")
        return inv
    for key, filename in LEGACY_FOLD_FILES.items():
        if (directory / filename).is_file():
            inv.files[key] = directory / filename
    for model_file in directory.glob("*_model.joblib"):
        inv.tabular_models[model_file.name[: -len("_model.joblib")]] = model_file
    candidates = list(directory.glob("best-*.ckpt"))
    if candidates:
        inv.best_checkpoint = rank_checkpoints(candidates)[0]
        inv.checkpoint_format = parse_checkpoint_name(inv.best_checkpoint.name).format
        if inv.checkpoint_format != FORMAT_AUC:
            inv.reasons.append(f"checkpoint_format_{inv.checkpoint_format}")
    else:
        inv.reasons.append("best_checkpoint_missing")
    if "split_manifest" not in inv.files:
        inv.reasons.append("split_metadata_missing")
    else:
        import pandas as pd  # local: keep import-time cheap

        manifest = pd.read_csv(inv.files["split_manifest"], dtype={"studyId": str}, nrows=5)
        inv.split_columns = tuple(manifest.columns)
        missing = [c for c in REQUIRED_SPLIT_COLUMNS if c not in manifest.columns]
        if missing:
            inv.reasons.append(f"split_manifest_missing_columns:{','.join(missing)}")
        if "patientId" not in manifest.columns:
            inv.reasons.append("split_manifest_without_patientId")
    if expect_tabular is not None:
        if expect_tabular not in inv.tabular_models:
            inv.reasons.append(f"tabular_model_missing:{expect_tabular}")
        if "tabular_preprocessor" not in inv.files:
            inv.reasons.append("tabular_preprocessor_missing")
        if expect_selector and "tabular_selector" not in inv.files:
            inv.reasons.append("tabular_selector_missing")
    return inv


def inventory_experiment(checkpoint_root: str | os.PathLike, experiment_name: str, num_folds: int, *,
                         expect_tabular: str | None = None, expect_selector: bool = False) -> list[LegacyFoldInventory]:
    return [inventory_fold(legacy_fold_checkpoint_dir(checkpoint_root, experiment_name, k), k,
                           expect_tabular=expect_tabular, expect_selector=expect_selector)
            for k in range(1, num_folds + 1)]


# --------------------------------------------------------------------------- weights
def legacy_network_state(state_dict: Mapping[str, Any], prefix: str = "model.") -> dict[str, Any]:
    """Strip the ``NiftiClassifier`` wrapper from a Lightning ``state_dict``.

    Legacy keys are ``model.<network param>`` plus the ``class_weights`` buffer
    (``sensitivity_class_weights`` is ``None`` and not saved; torchmetrics
    states are non-persistent).  The returned dict loads into the bare MONAI
    network (``DenseNet121``/``resnet10``/``resnet18``) with ``strict=True``.
    """
    out = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not out:
        raise ValueError(f"no '{prefix}*' keys in legacy state_dict (keys: {list(state_dict)[:5]})")
    return out


def load_legacy_network_state(path: str | os.PathLike) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a legacy Lightning checkpoint (lazy torch import).

    Returns ``(network_state_dict, extras)`` where ``extras`` carries
    ``class_weights`` (if saved), ``epoch``, ``global_step`` and the parsed
    filename.  Legacy files need ``weights_only=False`` (they pickle
    Lightning/numpy objects); only load files from a trusted checkpoint root.
    """
    import torch

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    if state is None:
        raise RuntimeError(f"Checkpoint does not contain a model state_dict: {path}")
    extras = {"epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step"),
              "class_weights": state.get("class_weights"), "name": parse_checkpoint_name(Path(path).name)}
    return legacy_network_state(state), extras


def ranking_table(paths: Sequence[str | os.PathLike]) -> list[dict[str, Any]]:
    """Ranked rows (best first) with parsed metrics - used by docs and the importer."""
    rows = []
    for rank, path in enumerate(rank_checkpoints(paths), start=1):
        parsed = parse_checkpoint_name(path.name)
        rows.append({"rank": rank, "path": str(path), "format": parsed.format, "epoch": parsed.epoch,
                     "auc_roc": parsed.auc_roc, "balanced_accuracy": parsed.balanced_accuracy,
                     "sensitivity": parsed.sensitivity, "version": parsed.version,
                     "selection_metric": FORMAT_SELECTION_METRIC[parsed.format]})
    return rows
