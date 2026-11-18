"""Opt-in parity with the legacy repo on synthetic tables (set MRI_BREAST_LEGACY_REPO=C:\\code\\mri_brest).

Loads legacy modules by file path (no torch/monai import) and compares cohort,
labels, features, preprocessing and LASSO output exactly (tolerance 0).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mribreast.data.duke import DukeCohortSource
from mribreast.synthetic.duke import write_duke_tables
from mribreast.tabular import LassoFeatureSelector, TabularPreprocessor, balanced_sample_weights

LEGACY = os.environ.get("MRI_BREAST_LEGACY_REPO")
pytestmark = [pytest.mark.parity, pytest.mark.skipif(not LEGACY, reason="set MRI_BREAST_LEGACY_REPO"),
              pytest.mark.filterwarnings("ignore")]


def _load(name: str, relative: str):
    if LEGACY not in sys.path:
        sys.path.insert(0, LEGACY)  # legacy modules import mriBreastDuke.constants
    spec = importlib.util.spec_from_file_location(name, Path(LEGACY) / "mriBreastDuke" / "dataLoaders" / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("binary,scheme", [(True, "oncotype_binary_18"), (False, "oncotype_3class_18_31")])
def test_cohort_features_preprocessing_lasso_match_legacy(tmp_path, binary, scheme):
    rx = _load("legacy_read_from_xlmx", "read_from_xlmx.py")
    pif = _load("legacy_precomputed", "precomputed_imaging_features.py")
    cp = _load("legacy_clinical_pre", "clinical_preprocessing.py")
    lasso = _load("legacy_lasso", "lasso_feature_selection.py")
    tables = write_duke_tables(tmp_path, n_patients=60, seed=5)
    groups = ["kinetic", "morphology", "heterogeneity"]
    old = rx.get_oncotype_clinical_predictors_as_study_df(isBinary=binary, features_file=str(tables["clinical_features_file"]),
                                                          metadata_file=str(tables["metadata_file"]))
    old, _ = pif.merge_precomputed_imaging_features(old, str(tables["imaging_features_file"]), groups)
    cohort = DukeCohortSource().load({"clinical_features_file": tables["clinical_features_file"],
                                      "metadata_file": tables["metadata_file"], "label_scheme": scheme,
                                      "imaging_features_file": tables["imaging_features_file"],
                                      "feature_groups": ("clinical", *groups)})
    new = cohort.frame
    assert list(old["studyId"]) == list(new["study_id"]) and list(old["patientId"]) == list(new["patient_id"])
    assert [tuple(s) for s in old["series_ids"]] == list(new["series_uids"])
    assert list(old["label"]) == list(new["label"])
    cont, cat = cohort.continuous_columns, cohort.categorical_columns
    pre_old, pre_new = cp.ClinicalFeaturePreprocessor(cont, cat).fit(old), TabularPreprocessor(cont, cat).fit(new)
    x_old, x_new = pre_old.transform(old), pre_new.transform(new)
    assert np.array_equal(x_old, x_new) and pre_old.get_feature_names_out() == pre_new.get_feature_names_out()
    y, g = new["label"].to_numpy(), new["patient_id"].to_numpy()
    w = balanced_sample_weights(y, 1.5)
    s_old = lasso.LassoFeatureSelector(cv_folds=3, cs=6, n_jobs=1, random_state=42).fit(x_old, y, sample_weight=w, groups=g)
    s_new = LassoFeatureSelector(cv_folds=3, cs=6, n_jobs=1, random_state=42).fit(x_new, y, sample_weight=w, groups=g)
    assert np.array_equal(s_old.coefficients_, s_new.coefficients_)
    assert s_old.selection_report(pre_old.get_feature_names_out()).equals(s_new.selection_report(pre_new.get_feature_names_out()))
