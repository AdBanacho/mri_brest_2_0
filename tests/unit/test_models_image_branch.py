"""ImageBranch fit/load/predict round trips on tiny synthetic 3D and 2D data (CPU, 1 epoch)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mribreast.config import schema
from mribreast.contracts import BranchTrainer, FittedPredictor, FoldData, check_probabilities
from mribreast.imaging.masks import LocalFileMaskProvider, SyntheticMaskProvider
from mribreast.models.image_branch import build_image_branch
from mribreast.synthetic.imaging import (make_annotation_boxes, synthetic_studies, write_2d_dataset, write_masks,
                                         write_nifti_series)

QUIET = '{"enable_progress_bar": false, "enable_model_summary": false, "num_sanity_val_steps": 0}'


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    studies = synthetic_studies(8, phases=3)
    boxes = make_annotation_boxes(studies, (16, 16, 8), seed=0)
    boxes.to_csv(root / "boxes.csv", index=False)
    write_nifti_series(root / "nifti", studies, (16, 16, 8), boxes=boxes, seed=0)
    manifest = write_2d_dataset(root / "img2d", studies, (24, 24), 3, seed=0, vary_count=True)
    masks = write_masks(root / "masks", manifest, seed=0, nifti_studies=studies, image_root=root / "nifti",
                        boxes=boxes)
    return root, studies, manifest, masks


def _cfg(root, **overrides):
    base = {"image_root": str(root / "nifti"), "prepared_root": str(root / "prep"),
            "subtraction_root": str(root / "sub"), "mri_model": "fcn", "target_size": "16 16 8",
            "phase_count": "3", "epoch": "1", "batch_size": "4", "num_workers": "0", "trainer_extra_json": QUIET}
    base.update({k: str(v) for k, v in overrides.items()})
    return schema.resolve("train", {}, base)


def _fold(studies, out):
    return FoldData(fold=1, fit=studies.iloc[:6], selection=studies.iloc[6:], seed=0, output_dir=out)


def test_3d_round_trip_with_annotation_boxes(synthetic, tmp_path):
    root, studies, _, _ = synthetic
    cfg = _cfg(root, use_annotation_boxes="true", annotation_boxes_file=str(root / "boxes.csv"))
    branch = build_image_branch(cfg, num_classes=2, logs_dir=tmp_path / "logs")
    assert isinstance(branch, BranchTrainer) and branch.branch == "image"
    out = tmp_path / "tr-x" / "fold_1" / "image"
    predictor = branch.fit(_fold(studies, out))
    assert isinstance(predictor, FittedPredictor)
    for name in ("best.ckpt", "last.ckpt", "image_branch.json", "history.csv", "annotation_coverage.csv"):
        assert (out / name).exists(), name
    meta = json.loads((out / "image_branch.json").read_text())
    assert meta["network"] == {"name": "fcn", "spatial_dims": 3, "in_channels": 3, "out_channels": 2, "extra": {}}
    assert meta["annotation_boxes"]["applies_to"] == "fit rows only"
    assert meta["annotation_boxes"]["eligible_fit_studies"] == 6
    assert meta["cache"]["cache_schema_version"] == 1 and meta["selection"]["monitor"] == "val_auc_roc"
    history = pd.read_csv(out / "history.csv")
    assert {"val_auc_roc", "val_balanced_accuracy", "val_sensitivity", "train_loss"} <= set(history.columns)
    probs = predictor.predict_proba(studies)
    check_probabilities(probs, 2)
    reloaded = branch.load(out, 1).predict_proba(studies.iloc[::-1])
    np.testing.assert_allclose(reloaded, probs[::-1], atol=1e-6)                 # row-aligned to the frame
    joined = studies.assign(series_uids=studies["series_uids"].map("|".join))   # '|'-joined form accepted
    np.testing.assert_allclose(branch.load(out, 1).predict_proba(joined), probs, atol=1e-6)
    audits = branch.audit(studies)
    assert set(audits["annotation_coverage"]["status"]) == {"eligible"}
    assert (tmp_path / "logs" / "tr-x" / "fold_1" / "metrics.csv").exists()


def test_3d_subtraction_channels_and_load_rejects_other_config(synthetic, tmp_path):
    root, studies, _, _ = synthetic
    branch = build_image_branch(_cfg(root, subtraction_mode="consecutive"), num_classes=2)
    predictor = branch.fit(_fold(studies, tmp_path / "img"))
    assert predictor.channels == 2
    check_probabilities(predictor.predict_proba(studies.iloc[:3]), 2)
    other = build_image_branch(_cfg(root, subtraction_mode="none"), num_classes=2)
    with pytest.raises(ValueError, match="subtraction_mode"):
        other.load(tmp_path / "img", 1)
    with pytest.raises(ValueError, match="classes"):
        build_image_branch(_cfg(root, subtraction_mode="consecutive"), num_classes=3).load(tmp_path / "img", 1)


def test_2d_manifest_with_masks_and_aggregation(synthetic, tmp_path):
    root, studies, manifest, masks = synthetic
    common = dict(profile="general", image_dim="2d", image_source="manifest_2d", image_manifest_file=str(manifest),
                  mri_model="resnet18", target_size_2d="24 24", mask_provider="local_files",
                  mask_manifest_file=str(masks), mask_kind="predicted", mask_mode="mask_channel",
                  mask_missing_policy="full_image", mask_provenance="frozen_external")
    branch = build_image_branch(_cfg(root, **common), num_classes=2, mask_provider=LocalFileMaskProvider(masks))
    out = tmp_path / "mean"
    predictor = branch.fit(_fold(studies, out))
    meta = json.loads((out / "image_branch.json").read_text())
    assert meta["network"]["in_channels"] == 2 and meta["channels"][-1] == "roi_mask"
    assert meta["mask"]["provider"]["provider"] == "local_files" and (out / "mask_audit_fit.csv").exists()
    mean = predictor.predict_proba(studies)
    check_probabilities(mean, 2)
    table, item_probs = predictor.item_probabilities(studies)
    counts = table.items.groupby("study_index").size()
    assert counts.min() == 1 and counts.max() == 3                               # 1..n images per study
    for study, group in table.items.groupby("study_index"):
        np.testing.assert_allclose(mean[study], item_probs[group.index].mean(axis=0), atol=1e-6)
    peak_branch = build_image_branch(_cfg(root, **common, aggregation="max_prob"), num_classes=2,
                                     mask_provider=LocalFileMaskProvider(masks))
    peak = peak_branch.load(out, 1).predict_proba(studies)
    for study, group in table.items.groupby("study_index"):
        assert peak[study, 1] == pytest.approx(item_probs[group.index, 1].max(), abs=1e-6)
    audits = branch.audit(studies)
    assert {"mask_audit", "mask_audit_summary"} <= set(audits)
    assert int(audits["mask_audit_summary"].loc[0, "status_missing"]) == 1


def test_2d_nifti_mask_selection_drop_gives_nan_row(synthetic, tmp_path):
    root, studies, _, masks = synthetic
    cfg = _cfg(root, profile="general", image_dim="2d", image_source="nifti", target_size_2d="16 16",
               slice_selection="mask_max_area", slices_per_study=2, mask_provider="local_files",
               mask_manifest_file=str(masks), mask_kind="predicted", mask_mode="roi_crop",
               mask_missing_policy="drop", mask_empty_policy="full_image")
    branch = build_image_branch(cfg, num_classes=2, mask_provider=LocalFileMaskProvider(masks))
    predictor = branch.fit(_fold(studies, tmp_path / "n2d"))
    probs = predictor.predict_proba(studies)
    dropped = predictor.last_audit.query("action == 'drop'")["study_id"].tolist()
    assert len(dropped) == 1                                                     # one NIfTI mask deliberately missing
    rows = studies["study_id"].isin(dropped).to_numpy()
    assert np.isnan(probs[rows]).all()
    check_probabilities(probs[~rows], 2)


def test_2d_nifti_synthetic_provider_bound_by_branch(synthetic, tmp_path):
    root, studies, _, _ = synthetic
    cfg = _cfg(root, profile="general", image_dim="2d", target_size_2d="16 16", slice_selection="uniform",
               slices_per_study=3, mask_provider="synthetic", mask_mode="foreground_only",
               subtraction_mode="post_minus_pre", aggregation="max_prob")
    branch = build_image_branch(cfg, num_classes=2, mask_provider=SyntheticMaskProvider(seed=0))
    check_probabilities(branch.fit(_fold(studies, tmp_path / "s")).predict_proba(studies), 2)


def test_protected_trainer_keys_and_missing_provider(synthetic, tmp_path):
    root, studies, _, _ = synthetic
    cfg = dict(_cfg(root))
    cfg["trainer_extra_json"] = {"callbacks": []}                                # bypasses schema validation
    with pytest.raises(ValueError, match="cannot override"):
        build_image_branch(cfg, num_classes=2).fit(_fold(studies, tmp_path / "p"))
    masked = dict(_cfg(root), mask_mode="mask_channel")
    with pytest.raises(ValueError, match="mask provider"):
        build_image_branch(masked, num_classes=2)


class _ConfidentNet:
    """Logit gap ~20 (+ per-study variation): float32 softmax saturates to exactly 1.0, float64 does not."""

    def __call__(self, x):
        import torch

        m = x.flatten(1).amax(1) / 10
        return torch.stack([torch.zeros_like(m), 20.0 + m], dim=1)

    def eval(self):
        return self


def test_probabilities_are_float64_and_do_not_saturate(synthetic):
    import torch

    from mribreast.imaging.settings import ImagingSettings
    from mribreast.models.image_branch import ImagePredictor

    root, studies, _, _ = synthetic
    settings = ImagingSettings(image_root=str(root / "nifti"), phase_count=3, target_size=(8, 8, 4))
    predictor = ImagePredictor(_ConfidentNet(), settings, num_classes=2, channels=3, provider=None, fold=1,
                               meta={}, directory=root)
    probs = predictor.predict_proba(studies)
    assert probs.dtype == np.float64
    assert len(np.unique(probs[:, 1])) > 1 and not np.all(probs[:, 1] == 1.0)
    float32 = torch.softmax(torch.tensor([[0.0, 20.0], [0.0, 20.3]]), dim=1)[:, 1]
    assert float(float32[0]) == float(float32[1]) == 1.0          # the tie float64 avoids
