"""Parity of MRI network inputs and class/sample weights against the reference repository.

Same synthetic NIfTI series and labels are fed to the legacy ``NiftiDataset`` (MONAI path, the legacy
default) and to the new ``StudyVolumeLoader``. Tolerances: image inputs atol=1e-4 / rtol=1e-4 (float32
resampling done by different MONAI/torch code paths), weights exact to 1e-12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import legacy_loader
from mribreast.imaging.loader import StudyVolumeLoader
from mribreast.models.classifier import compute_balanced_class_weights
from mribreast.synthetic.imaging import synthetic_studies, write_nifti_series
from mribreast.tabular.branch import balanced_sample_weights

pytestmark = pytest.mark.parity

TARGET = (12, 10, 6)  # asymmetric on purpose


@pytest.fixture(scope="module")
def series(tmp_path_factory, legacy_repo):
    root = tmp_path_factory.mktemp("nifti")
    studies = synthetic_studies(4, phases=5, seed=11)
    write_nifti_series(root, studies, shape=(20, 16, 9), seed=11)
    return root, studies


@pytest.mark.parametrize("mode", ["none", "post_minus_pre", "consecutive"])
def test_image_inputs_match_legacy(series, legacy_repo, tmp_path, mode):
    root, studies = series
    nifti_dataset = legacy_loader.import_legacy(legacy_repo, "mriBreastDuke.dataLoaders.NiftiDataset")
    frame = pd.DataFrame({
        "series_ids": [list(u if isinstance(u, (list, tuple)) else str(u).split("|")) for u in studies["series_uids"]],
        "label": studies["label"].astype(int).to_numpy(),
        "studyId": studies["study_id"].astype(str).to_numpy(),
    })
    legacy = nifti_dataset.NiftiDataset(frame, target_size=TARGET, image_root=str(root), subtraction_mode=mode,
                                        prepared_root=str(tmp_path / "legacy_prep"),
                                        subtraction_root=str(tmp_path / "legacy_sub"), training=False)
    channels = 5 if mode == "none" else 4
    new = StudyVolumeLoader(root, subtraction_mode=mode, channels=channels, target_size=TARGET,
                            prepared_root=tmp_path / "new_prep", subtraction_root=tmp_path / "new_sub")
    for index in range(len(frame)):
        legacy_volume, _ = legacy[index]
        legacy_array = legacy_volume.squeeze(1).numpy() if legacy_volume.ndim == 5 else legacy_volume.numpy()
        new_array = new.load(frame["series_ids"][index]).data
        assert new_array.shape == legacy_array.shape == (channels, *TARGET)
        np.testing.assert_allclose(new_array, legacy_array, atol=1e-4, rtol=1e-4)


def test_class_weights_match_legacy(legacy_repo):
    cv = legacy_loader.import_legacy(legacy_repo, "mriBreastDuke.n_fold_cv_run")
    rng = np.random.default_rng(5)
    for boost in (1.0, 2.0, 0.5):
        labels = rng.integers(0, 2, size=37)
        legacy = cv._compute_balanced_class_weights(labels, positive_boost=boost).numpy()
        new = np.asarray(compute_balanced_class_weights(labels, 2, positive_boost=boost), dtype=float)
        np.testing.assert_allclose(new, legacy, rtol=1e-6)


def test_tabular_sample_weights_match_legacy():
    from sklearn.utils.class_weight import compute_sample_weight

    rng = np.random.default_rng(6)
    for boost in (1.0, 3.0):
        labels = rng.integers(0, 2, size=41)
        legacy = compute_sample_weight(class_weight="balanced", y=labels)  # n_fold_cv_run.py:360-366
        if boost != 1.0:
            legacy = legacy.copy()
            legacy[labels == 1] *= boost
        new = balanced_sample_weights(labels, positive_boost=boost, positive_class=1, num_classes=2)
        np.testing.assert_allclose(new, legacy, rtol=0, atol=1e-12)
