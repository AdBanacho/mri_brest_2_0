"""Deterministic fake branches and cohorts for protocol / end-to-end tests (no torch, no xgboost).

Usage::

    from tests.fakes import FakeTrainer, make_fake_cohort, make_cfg
    cohort = make_fake_cohort(n_patients=40, multi_study=True)
    trainers = {"image": FakeTrainer("image", "signal_image"), "tabular": FakeTrainer("tabular", "signal_tabular")}
    cfg = make_cfg("validate", num_folds=3)

``FakeTrainer`` fits a tiny multinomial logistic regression on one synthetic
column of the *fit* rows, records every call in ``.calls`` and saves/loads its
parameters as JSON (byte-identical for identical inputs).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mribreast.config import schema
from mribreast.contracts import (LABEL, ONCOTYPE_SCORE, PATIENT_ID, SERIES_UIDS, STUDY_ID, CohortTable, FoldData,
                                 label_definition)


def _unit_noise(key: str) -> float:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 2**32 - 0.5


class FakePredictor:
    def __init__(self, branch: str, params: Mapping[str, Any]):
        self.branch = branch
        self.params = dict(params)
        self.checkpoint_path: str | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.params
        x = frame[p["column"]].to_numpy(dtype=float)
        coef = np.asarray(p["coef"], dtype=float)          # [C, 1]
        intercept = np.asarray(p["intercept"], dtype=float)  # [C]
        logits = x[:, None] * coef[:, 0][None, :] + intercept[None, :]
        if p["noise"]:
            logits = logits + p["noise"] * np.array(
                [[_unit_noise(f"{s}|{self.branch}|{k}") for k in range(len(intercept))]
                 for s in frame[STUDY_ID].astype(str)])
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fake_{self.branch}.json"
        path.write_text(json.dumps(self.params, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self.checkpoint_path = str(path)


class FakeTrainer:
    """``BranchTrainer`` stand-in. ``fail_on_fold`` raises inside ``fit`` (simulated crash)."""

    def __init__(self, branch: str, column: str, noise: float = 0.0, num_classes: int = 2,
                 fail_on_fold: int | None = None, with_hook: bool = False):
        self.branch = branch
        self.column = column
        self.noise = float(noise)
        self.num_classes = int(num_classes)
        self.fail_on_fold = fail_on_fold
        self.calls: list[dict[str, Any]] = []
        self.hook_calls: list[Path] = []
        if with_hook:
            self.after_all_folds = self._after_all_folds  # duck-typed optional hook

    def fit(self, data: FoldData) -> FakePredictor:
        self.calls.append({"fold": data.fold, "fit": data.fit[STUDY_ID].astype(str).tolist(),
                           "selection": data.selection[STUDY_ID].astype(str).tolist(),
                           "output_dir": str(data.output_dir)})
        if self.fail_on_fold is not None and data.fold == self.fail_on_fold:
            raise RuntimeError(f"simulated crash in fold {data.fold}")
        x = data.fit[[self.column]].to_numpy(dtype=float)
        y = data.fit[LABEL].to_numpy(dtype=int)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(x, y)
        if model.coef_.shape[0] == 1:  # binary sklearn -> two-class logits
            coef = [[0.0], [float(model.coef_[0, 0])]]
            intercept = [0.0, float(model.intercept_[0])]
        else:
            coef = [[float(c)] for c in model.coef_[:, 0]]
            intercept = [float(v) for v in model.intercept_]
        if [int(c) for c in model.classes_] != list(range(self.num_classes)):
            raise ValueError(f"fit rows miss classes: {model.classes_}")
        params = {"branch": self.branch, "column": self.column, "noise": self.noise, "fold": data.fold,
                  "coef": [[round(c[0], 12)] for c in coef], "intercept": [round(v, 12) for v in intercept],
                  "fit_studies": data.fit[STUDY_ID].astype(str).tolist(),
                  "selection_studies": data.selection[STUDY_ID].astype(str).tolist(), "seed": data.seed}
        (Path(data.output_dir) / "history.csv").write_text("epoch,score\n0,0.5\n", encoding="utf-8")
        return FakePredictor(self.branch, params)

    def load(self, directory: Path, fold: int) -> FakePredictor:
        path = Path(directory) / f"fake_{self.branch}.json"
        params = json.loads(path.read_text(encoding="utf-8"))
        if params["fold"] != fold:
            raise ValueError(f"{path} belongs to fold {params['fold']}, not {fold}")
        predictor = FakePredictor(self.branch, params)
        predictor.checkpoint_path = str(path)
        return predictor

    def _after_all_folds(self, train_dir: Path) -> None:
        self.hook_calls.append(Path(train_dir))
        (Path(train_dir) / f"fake_{self.branch}_after_all_folds.txt").write_text("ok\n", encoding="utf-8")


def make_fake_cohort(n_patients: int = 40, *, multi_study: bool = False, num_classes: int = 2, seed: int = 0,
                     separation: float = 1.5) -> CohortTable:
    """Synthetic study-level cohort; ``multi_study`` gives patient i ``1 + i % 3`` studies (same label)."""
    rng = np.random.default_rng(seed)
    scheme = "oncotype_binary_18" if num_classes == 2 else "oncotype_3class_18_31"
    label_def = label_definition(scheme)
    rows = []
    patient_labels = rng.permutation(np.arange(n_patients) % num_classes)
    for i in range(n_patients):
        label = int(patient_labels[i])
        score = {0: 10.0, 1: 25.0, 2: 40.0}[label] if num_classes == 3 else (10.0 if label == 0 else 30.0)
        for j in range(1 + (i % 3 if multi_study else 0)):
            rows.append({PATIENT_ID: f"P{i:03d}", STUDY_ID: f"1.2.840.{i}.{j}", LABEL: label_def.encode(score),
                         SERIES_UIDS: tuple(f"1.3.{i}.{j}.{k}" for k in range(3)), ONCOTYPE_SCORE: score,
                         "signal_image": separation * label + rng.normal(),
                         "signal_tabular": separation * label + rng.normal(),
                         "age": float(40 + (i % 30))})
    frame = pd.DataFrame(rows)
    audit = pd.DataFrame({STUDY_ID: frame[STUDY_ID], "status": "included"})
    return CohortTable(frame=frame, continuous_columns=["signal_tabular", "age"], categorical_columns=[],
                       label=label_def, audit=audit, notes=["synthetic"])


def make_cfg(stage: str = "validate", **overrides: Any) -> dict[str, Any]:
    """Resolved config with small, fast defaults (3 folds, 3 inner folds)."""
    base: dict[str, Any] = {"num_folds": 3, "threshold_calibration_folds": 3, "seed": 42}
    if overrides.get("label_scheme") == "oncotype_3class_18_31":
        base["threshold_strategy"] = "fixed_0.5"
    base.update(overrides)
    if stage == "train":
        base = {k: v for k, v in base.items() if schema.FIELDS[k].section != "validate"}
    return schema.resolve(stage, {}, {k: (schema.format_value(schema.FIELDS[k], v) if not isinstance(v, str) else v)
                                      for k, v in base.items()})


def make_trainers(branches=("image", "tabular"), num_classes: int = 2, **kwargs: Any) -> dict[str, FakeTrainer]:
    columns = {"image": "signal_image", "tabular": "signal_tabular"}
    return {b: FakeTrainer(b, columns[b], num_classes=num_classes, **kwargs) for b in branches}


def import_legacy(module: str):
    r"""Import ``mriBreastDuke.<module>`` from the reference repo or ``pytest.skip``.

    The repo is found via ``MRI_BREAST_LEGACY_REPO`` or ``C:\code\mri_brest``; it is never modified.
    """
    import importlib
    import os
    import sys

    import pytest

    candidates = [os.environ.get("MRI_BREAST_LEGACY_REPO"), r"C:\code\mri_brest"]
    for root in candidates:
        if root and (Path(root) / "mriBreastDuke" / f"{module}.py").is_file():
            if root not in sys.path:
                sys.path.insert(0, root)
            try:
                return importlib.import_module(f"mriBreastDuke.{module}")
            except ImportError as exc:  # pragma: no cover - missing legacy deps
                pytest.skip(f"legacy module {module} not importable: {exc}")
    pytest.skip("legacy reference repository not available (set MRI_BREAST_LEGACY_REPO)")


class FakeSliceSegmenter:
    """Box-prompted ``SliceSegmenter`` stand-in: positive logits where the window is bright AND inside the
    prompt box, so a wrongly mapped box gives a wrong mask (docs/segmentation-mri-core.md)."""

    model_id = "fake"
    model_version = "v1"

    def __init__(self, level: float = 0.5):
        self.level = level
        self.calls: list[tuple[tuple[int, int], tuple[float, ...]]] = []

    def describe(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "model_version": self.model_version, "level": self.level}

    def segment(self, windows, boxes):
        import numpy as np

        from mribreast.prepare.segmentation.mri_core import SliceResult

        out = []
        for window, (x0, y0, x1, y1) in zip(windows, boxes):
            assert window.dtype == np.float32 and 0.0 <= window.min() and window.max() <= 1.0
            self.calls.append((window.shape, (x0, y0, x1, y1)))
            inside = np.zeros(window.shape, dtype=bool)
            inside[int(np.floor(y0)):int(np.ceil(y1)), int(np.floor(x0)):int(np.ceil(x1))] = True
            out.append(SliceResult(np.where(inside, (window - self.level) * 20.0, -20.0).astype(np.float32), 0.9))
        return out
