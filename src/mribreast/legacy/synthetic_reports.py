"""Synthetic equivalents of legacy validation reports (same headers/layout, random numbers).

Used by tests and parity checks; the numbers are drawn from a seeded RNG and
never come from the real ``validationSummary`` files.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from mribreast.legacy._fs import fs
from mribreast.legacy.exporter import LEGACY_VALIDATE_ARGS, write_legacy_run
from mribreast.legacy.names import CLINICAL_SAFE_SUFFIX, alpha_dir, build_experiment_name

#: keys missing from validation_config.json files written before the Slurm-controller era (stored summary)
STORED_ERA_DROPPED_KEYS = ("seed", "target_size", "experiment_id", "use_annotation_boxes", "annotation_boxes_file",
                           "clinical_features_file", "metadata_file", "image_root", "prepared_root",
                           "subtraction_root", "mri_extra_json")

BASE_CFG: dict[str, Any] = {
    "subtraction_mode": "none", "feature_selector": "lasso", "lasso_cv_folds": 5, "lasso_cs": 20,
    "lasso_min_features": 1, "threshold_calibration_folds": 5, "lr": 1e-4, "batch_size": 4, "num_folds": 5,
    "seed": 42, "target_size": (256, 256, 64), "use_annotation_boxes": False, "experiment_id": "",
    "imaging_patient_id_column": "Patient ID", "validate_num_workers": 4,
}


def grid(mri_models: Iterable[str] = ("densenet121", "resnet18"), sens: Iterable[float] = (0.05, 0.3),
         boosts: Iterable[float] = (1.0, 2.0), groups: Iterable[str] = ("clinical", "clinical,kinetic,morphology,heterogeneity"),
         feature_models: Iterable[str] = ("xgboost",)) -> list[dict[str, Any]]:
    return [{**BASE_CFG, "mri_model": m, "sensitivity_lambda": s, "positive_boost": b, "feature_groups": g,
             "feature_model": f}
            for m, g, f, s, b in itertools.product(mri_models, groups, feature_models, sens, boosts)]


def synthetic_fold_metrics(rng: np.random.Generator, num_folds: int = 5, *, folds: Iterable[int] | None = None,
                           experiment_name: str = "exp", nan_fusion_auc_fold: int | None = None,
                           tie_value: float | None = None) -> pd.DataFrame:
    """Per-fold metrics in NEW column names (image/tabular/fusion) with legacy-like row counts."""
    rows = []
    for fold in (folds or range(1, num_folds + 1)):
        row: dict[str, Any] = {
            "fold": fold,
            "checkpoint_path": (f"/synthetic/check_points/{experiment_name}/fold_{fold}/checkpoints/"
                                f"best-epoch={int(rng.integers(0, 30)):02d}-val_auc_roc={rng.uniform(0.4, 0.9):.4f}-"
                                f"val_balanced_accuracy={rng.uniform(0.4, 0.8):.4f}-val_sensitivity={rng.uniform(0, 1):.4f}.ckpt"),
            "inner_fit_rows": int(rng.integers(160, 170)), "inner_calibration_rows": 42,
            "effective_inner_calibration_folds": 5, "outer_validation_rows": int(rng.integers(51, 54)),
        }
        for branch in ("image", "tabular", "fusion"):
            row[f"{branch}_accuracy"] = rng.uniform(0.3, 0.8)
            row[f"{branch}_balanced_accuracy"] = rng.uniform(0.3, 0.8)
            row[f"{branch}_sensitivity"] = rng.uniform(0.0, 1.0)
            row[f"{branch}_decision_threshold"] = rng.uniform(0.0, 1.0)
            row[f"{branch}_specificity"] = rng.uniform(0.0, 1.0)
            row[f"{branch}_auc_roc"] = rng.uniform(0.3, 0.9) if tie_value is None else tie_value
        if nan_fusion_auc_fold == fold:
            row["fusion_auc_roc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_synthetic_validation_root(root: str | Path, configs: list[Mapping[str, Any]], *, seed: int = 0,
                                    alphas: Iterable[float] = (0.5,), stored_era: bool = False,
                                    incomplete: Mapping[int, Iterable[int]] | None = None,
                                    tie_groups: Mapping[int, float] | None = None) -> list[Path]:
    """Write one legacy run directory per (config, alpha).

    ``stored_era=True`` mimics the tracked validationSummary: experiment names
    without ``_clinical-safe-v2`` and validation_config.json without the keys
    in :data:`STORED_ERA_DROPPED_KEYS`.  ``incomplete[i]`` gives the folds
    written for config ``i``; ``tie_groups[i]`` forces all AUCs of config ``i``.
    """
    rng = np.random.default_rng(seed)
    run_dirs = []
    for index, cfg in enumerate(configs):
        name = build_experiment_name(cfg)
        if stored_era:
            name = name.replace(CLINICAL_SAFE_SUFFIX, "")
        for alpha in alphas:
            folds = (incomplete or {}).get(index)
            metrics = synthetic_fold_metrics(rng, int(cfg.get("num_folds", 5)), folds=folds, experiment_name=name,
                                             tie_value=(tie_groups or {}).get(index))
            thresholds = {int(f): {"image": float(r["image_decision_threshold"]),
                                   "tabular": float(r["tabular_decision_threshold"]),
                                   "fusion": float(r["fusion_decision_threshold"])}
                          for f, r in metrics.set_index("fold").iterrows()}
            run_dir = write_legacy_run(root, cfg, alpha, metrics, thresholds=thresholds, experiment_name=name,
                                       device="cpu")
            if stored_era:
                path = fs(run_dir) / "validation_config.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                for key in STORED_ERA_DROPPED_KEYS:
                    data.pop(key, None)
                path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run_dirs.append(run_dir)
    return run_dirs


def run_id_of(cfg: Mapping[str, Any], alpha: float = 0.5, stored_era: bool = False) -> str:
    name = build_experiment_name(cfg)
    if stored_era:
        name = name.replace(CLINICAL_SAFE_SUFFIX, "")
    return f"{name}/{alpha_dir(alpha)}"


__all__ = ["BASE_CFG", "LEGACY_VALIDATE_ARGS", "STORED_ERA_DROPPED_KEYS", "grid", "run_id_of",
           "synthetic_fold_metrics", "write_synthetic_validation_root"]
