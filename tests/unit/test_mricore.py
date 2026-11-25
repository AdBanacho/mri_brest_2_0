"""Feature group ``mricore``: embedding CSV reader/merge, policy, LASSO family, extractor helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mribreast.contracts import PATIENT_ID, STUDY_ID
from mribreast.data.duke import CohortError, DukeCohortSource
from mribreast.data.mricore import MriCoreFeaturesError, merge_mricore, read_mricore_features
from mribreast.data.predictor_policy import load_policy
from mribreast.prepare import mricore_embed as me
from mribreast.synthetic.duke import write_duke_tables
from mribreast.tabular.lasso import feature_group


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return write_duke_tables(tmp_path_factory.mktemp("duke"), n_patients=40, seed=3)


def _cfg(tables, **extra):
    cfg = {"clinical_features_file": str(tables["clinical_features_file"]), "metadata_file": str(tables["metadata_file"]),
           "profile": "duke_legacy_3d", "phase_order": "metadata", "phase_count": 5, "label_scheme": "oncotype_binary_18",
           "positive_class": 1, "branches": ("tabular",), "feature_groups": ("clinical",),
           "include_sensitive": False, "prediction_time": "mri", "allowed_predictors_file": ""}
    cfg.update(extra)
    return cfg


def _write_embeddings(path, studies: pd.DataFrame, dim: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = studies[[PATIENT_ID, STUDY_ID]].copy()
    frame["series_uid"] = "1.2.3"
    frame["n_slices"] = 5
    for k in range(dim):
        frame[f"mricore_roi_{k:03d}"] = rng.normal(size=len(frame))
    frame.to_csv(path, index=False)
    return frame


def test_merge_reports_coverage_and_ignores_provenance(tmp_path):
    studies = pd.DataFrame({PATIENT_ID: ["p1", "p2", "p3"], STUDY_ID: ["s1", "s2", "s3"], "x": [1, 2, 3]})
    _write_embeddings(tmp_path / "e.csv", studies.iloc[:2])
    merged, columns, report = merge_mricore(studies, tmp_path / "e.csv")
    assert columns == [f"mricore_roi_{k:03d}" for k in range(4)]
    assert "series_uid" not in merged and "n_slices" not in merged
    assert report["studies_with_mricore"] == 2 and report["coverage"] == pytest.approx(2 / 3)
    assert merged.loc[2, columns].isna().all() and merged["x"].tolist() == [1, 2, 3]


def test_reader_rejects_duplicates_missing_and_nonfinite(tmp_path):
    base = pd.DataFrame({PATIENT_ID: ["p1", "p1"], STUDY_ID: ["s1", "s1"], "mricore_roi_000": [0.1, 0.2]})
    base.to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(MriCoreFeaturesError, match="duplicate"):
        read_mricore_features(tmp_path / "dup.csv")
    base.assign(**{STUDY_ID: ["s1", "s2"], "mricore_roi_000": [0.1, np.inf]}).to_csv(tmp_path / "inf.csv", index=False)
    with pytest.raises(MriCoreFeaturesError, match="non-finite"):
        read_mricore_features(tmp_path / "inf.csv")
    base.drop(columns="mricore_roi_000").to_csv(tmp_path / "none.csv", index=False)
    with pytest.raises(MriCoreFeaturesError, match="no 'mricore_"):
        read_mricore_features(tmp_path / "none.csv")


def test_cohort_group_opt_in(tables, tmp_path):
    with pytest.raises(CohortError, match="mricore_features_file"):
        DukeCohortSource().load(_cfg(tables, feature_groups=("clinical", "mricore")))
    plain = DukeCohortSource().load(_cfg(tables))
    assert not any(c.startswith("mricore_") for c in plain.frame.columns)
    _write_embeddings(tmp_path / "e.csv", plain.frame)
    cohort = DukeCohortSource().load(_cfg(tables, feature_groups=("clinical", "mricore"),
                                          mricore_features_file=str(tmp_path / "e.csv")))
    assert [c for c in cohort.continuous_columns if c.startswith("mricore_")] == [f"mricore_roi_{k:03d}" for k in range(4)]
    assert cohort.frame["mricore_roi_000"].notna().all()
    assert any("mricore coverage" in n for n in cohort.notes)


def test_policy_and_lasso_family():
    policy = load_policy()
    assert policy.version == "1.1"
    assert policy.check(["mricore_roi_000", "mricore_global_255"]) == {"mricore_roi_000": "mricore",
                                                                       "mricore_global_255": "mricore"}
    assert feature_group("num__mricore_roi_007") == "mricore"


# --------------------------------------------------------------------------- extractor helpers
def test_slice_box_and_grid_helpers():
    assert me.select_slices(10, 14, 16) == [10, 11, 12, 13]
    picked = me.select_slices(0, 100, 5)
    assert picked[0] == 0 and picked[-1] == 99 and len(picked) == 5
    assert me.select_slices(5, 5, 4) == []
    assert me.widen_box(((10, 20, 3), (20, 40, 6)), (64, 64, 8), 0.1) == ((9, 18, 3), (21, 42, 6))
    assert me.widen_box(((0, 0, 0), (64, 64, 8)), (64, 64, 8), 0.5) == ((0, 0, 0), (64, 64, 8))
    assert me.feature_window(0, 512, 512, 64) == (0, 64)
    assert me.feature_window(100, 101, 512, 64) == (12, 13)        # always >= 1 cell
    assert me.feature_window(511, 512, 512, 64) == (63, 64)
    assert me.feature_columns("both", 2) == ["mricore_roi_000", "mricore_roi_001", "mricore_global_000", "mricore_global_001"]


def test_normalise_volume():
    out = me.normalise_volume(np.arange(1000, dtype=float).reshape(10, 10, 10))
    assert out.min() == 0.0 and out.max() == 1.0 and out.dtype == np.float32
    assert not me.normalise_volume(np.full((4, 4, 4), 7.0)).any()


def test_study_embedding_pools_box_region():
    torch = pytest.importorskip("torch")

    class Identityish(torch.nn.Module):
        """Feature map = 16x16-average of channel 0 -> grid cells reflect image intensity."""
        def forward(self, x):
            return torch.nn.functional.avg_pool2d(x[:, :1], 16)

    vol = np.zeros((64, 64, 6), dtype=np.float32)       # (column, row, slice); 64 = image_size -> no resize blur
    vol[32:64, 0:32, 2:4] = 1.0                         # bright lesion: columns 32.., rows ..32, slices 2-3
    box = ((32, 0, 2), (64, 32, 4))
    kwargs = dict(encoder=Identityish(), layer="neck", image_size=64, max_slices=8, roi_margin=0.0,
                  batch_size=2, device="cpu")
    roi, used = me.study_embedding(vol, box, pool="roi", **kwargs)
    glob, _ = me.study_embedding(vol, box, pool="global", **kwargs)
    both, _ = me.study_embedding(vol, box, pool="both", **kwargs)
    std = me.IMAGENET_STD[0]
    lesion, background = (1 - me.IMAGENET_MEAN[0]) / std, -me.IMAGENET_MEAN[0] / std
    assert used == 2 and roi.shape == (1,)
    assert roi[0] == pytest.approx(lesion, abs=1e-4)                      # box = lesion only
    assert glob[0] == pytest.approx((lesion + 3 * background) / 4, abs=1e-4)
    assert both.tolist() == pytest.approx([roi[0], glob[0]])


def test_backbone_mapping_keeps_pos_embed_and_splits_final_norm():
    torch = pytest.importorskip("torch")
    state = {"teacher": {
        "backbone.cls_token": torch.zeros(1, 1, 8), "backbone.mask_token": torch.zeros(1, 8),
        "backbone.pos_embed": torch.arange(17 * 8, dtype=torch.float32).reshape(1, 17, 8),   # cls + 4x4 tokens
        "backbone.patch_embed.proj.weight": torch.ones(8, 3, 16, 16),
        "backbone.blocks.1.4.mlp.fc1.weight": torch.ones(32, 8),
        "backbone.blocks.1.4.attn.qkv.bias": torch.ones(24),
        "backbone.norm.weight": torch.ones(8), "backbone.norm.bias": torch.zeros(8),
        "dino_head.mlp.0.weight": torch.ones(2, 2)}}
    mapped, norm = me.map_backbone_state(state, grid=4)
    assert set(mapped) == {"pos_embed", "patch_embed.proj.weight", "blocks.4.mlp.lin1.weight", "blocks.4.attn.qkv.bias"}
    assert set(norm) == {"weight", "bias"}
    assert mapped["pos_embed"].shape == (1, 4, 4, 8)
    assert torch.equal(mapped["pos_embed"].reshape(16, 8), state["teacher"]["backbone.pos_embed"][0, 1:])  # cls dropped
    assert me.map_backbone_state(state, grid=8)[0]["pos_embed"].shape == (1, 8, 8, 8)               # resized
