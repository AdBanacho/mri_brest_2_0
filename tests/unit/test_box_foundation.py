"""box_mode foundation: config rules, box decisions/policies/audit, ItemBuilder items, tuple-input plumbing,
meta + load checks. The three mode builders are stubbed here (monkeypatched); each mode has its own test file.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from mribreast.config import schema
from mribreast.config.schema import ConfigError
from mribreast.contracts import FoldData, check_probabilities
from mribreast.imaging.annotations import load_annotation_boxes
from mribreast.imaging.boxes import (BOX_AUDIT_COLUMNS, BoxInput, BoxPolicyError, box_audit_frame, decide_boxes,
                                     expand_box)
from mribreast.imaging.datasets import ImageDataset, ItemBuilder, expected_channels
from mribreast.imaging.settings import settings_from_cfg
from mribreast.models.classifier import ImageClassifier, map_inputs
from mribreast.models.image_branch import ImagePredictor, build_image_branch
from mribreast.synthetic.imaging import make_annotation_boxes, synthetic_studies, write_nifti_series

QUIET = '{"enable_progress_bar": false, "enable_model_summary": false, "num_sanity_val_steps": 0}'
SHAPE = (16, 16, 8)


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("boxes")
    studies = synthetic_studies(8, phases=3)
    boxes = make_annotation_boxes(studies, SHAPE, seed=0)
    write_nifti_series(root / "nifti", studies, SHAPE, boxes=boxes, seed=0)
    boxes.to_csv(root / "boxes.csv", index=False)
    # the last patient has no box -> exercises box_missing_policy
    boxes.iloc[:-1].to_csv(root / "boxes_missing_last.csv", index=False)
    return root, studies


def _cfg(root, boxes="boxes.csv", **overrides):
    base = {"image_root": str(root / "nifti"), "prepared_root": str(root / "prep"),
            "subtraction_root": str(root / "sub"), "mri_model": "fcn", "target_size": "16 16 8",
            "phase_count": "3", "epoch": "1", "batch_size": "4", "num_workers": "0", "trainer_extra_json": QUIET,
            "annotation_boxes_file": str(root / boxes)}
    base.update({k: str(v) for k, v in overrides.items()})
    return schema.resolve("train", {}, base)


# ----------------------------------------------------------------------------- config
def test_box_defaults_and_settings():
    cfg = schema.resolve("validate")
    assert cfg["box_mode"] == "none" and cfg["box_margin"] == 0.1 and cfg["box_local_size"] == (64, 64, 32)
    assert cfg["dual_stream_embed_dim"] == 128 and cfg["box_missing_policy"] == "error"
    s = settings_from_cfg(schema.resolve("train", {"box_mode": "dual_stream", "annotation_boxes_file": "b.xlsx",
                                                   "box_local_size": ["8", "8", "4"]}))
    assert s.uses_boxes and s.box_local_size == (8, 8, 4) and s.to_json()["box_local_size"] == [8, 8, 4]
    assert not settings_from_cfg(schema.resolve("train")).uses_boxes


@pytest.mark.parametrize("overrides, message", [
    ({"box_mode": "roi_crop"}, "requires annotation_boxes_file"),
    ({"box_mode": "roi_crop", "annotation_boxes_file": "b", "profile": "general", "image_dim": "2d"},
     "requires image_dim=3d"),
    ({"box_mode": "box_channel", "annotation_boxes_file": "b", "profile": "general", "mask_provider": "synthetic",
      "mask_mode": "mask_channel"}, "mutually exclusive"),
    ({"box_mode": "box_channel", "annotation_boxes_file": "b", "profile": "general", "mask_provider": "synthetic",
      "slice_selection": "mask_max_area"}, "mutually exclusive"),
    ({"box_mode": "roi_crop", "annotation_boxes_file": "b", "use_annotation_boxes": "true"}, "box_mode=roi_crop"),
    ({"box_mode": "dual_stream", "annotation_boxes_file": "b", "branches": "tabular"}, "need the image branch"),
    ({"box_margin": "2.5"}, "box_margin"),
    ({"box_margin": "-0.1"}, "box_margin"),
    ({"box_local_size": "8 8"}, "box_local_size"),
    ({"box_local_size": "8 0 8"}, "positive integers"),
    ({"dual_stream_embed_dim": "0"}, "dual_stream_embed_dim"),
    ({"box_missing_policy": "skip"}, "not in"),
])
def test_box_rules_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        schema.resolve("validate", overrides)


@pytest.mark.parametrize("mode", ["roi_crop", "box_channel", "dual_stream"])
def test_box_modes_accepted(mode):
    cfg = schema.resolve("validate", {"box_mode": mode, "annotation_boxes_file": "b.xlsx"})
    assert cfg["box_mode"] == mode
    if mode != "roi_crop":   # erasing on the global volume is allowed
        schema.resolve("validate", {"box_mode": mode, "annotation_boxes_file": "b", "use_annotation_boxes": "true"})


def test_box_fields_change_train_identity():
    base = schema.train_id(schema.resolve("train"))
    assert schema.train_id(schema.resolve("train", {"box_mode": "roi_crop", "annotation_boxes_file": "b"})) != base
    assert schema.train_id(schema.resolve("train", {"box_margin": "0.2"})) != base


# ----------------------------------------------------------------------------- geometry helper
def test_expand_box_margin_and_clipping():
    box = ((4, 4, 2), (8, 6, 4))                     # extents 4, 2, 2
    assert expand_box(box, (16, 16, 8), 0.0) == ([4, 4, 2], [8, 6, 4])
    assert expand_box(box, (16, 16, 8), 0.5) == ([2, 3, 1], [10, 7, 5])   # ceil(0.5*extent) per side
    assert expand_box(box, (16, 16, 8), 2.0) == ([0, 0, 0], [16, 10, 8])  # clipped to the volume
    assert expand_box(((0, 0, 0), (1, 1, 1)), (4, 4, 4), 0.1) == ([0, 0, 0], [2, 2, 2])
    with pytest.raises(ValueError):
        expand_box(((0, 0, 0), (17, 1, 1)), (16, 16, 8), 0.1)
    with pytest.raises(ValueError):
        expand_box(((2, 0, 0), (2, 1, 1)), (16, 16, 8), 0.1)


# ----------------------------------------------------------------------------- decisions / audit
def test_decide_boxes_policies(synthetic):
    root, studies = synthetic
    boxes = load_annotation_boxes(root / "boxes_missing_last.csv")
    last = str(studies.iloc[-1]["study_id"])
    decisions, report = decide_boxes(studies, boxes, root / "nifti", policy="full_image", box_mode="roi_crop")
    assert list(decisions) == studies["study_id"].astype(str).tolist()
    assert decisions[last].status == "missing_box" and decisions[last].action == "full_image"
    assert decisions[last].box is None
    first = decisions[str(studies.iloc[0]["study_id"])]
    assert first.action == "use" and first.source_shape == SHAPE and first.box == boxes[first.patient_id]
    assert set(report["status"]) == {"eligible", "missing_box"}
    assert decide_boxes(studies, boxes, root / "nifti", policy="drop")[0][last].action == "drop"
    with pytest.raises(BoxPolicyError, match="missing_box"):
        decide_boxes(studies, boxes, root / "nifti", policy="error")
    audit_only, _ = decide_boxes(studies, boxes, root / "nifti", policy="error", raise_errors=False)
    assert audit_only[last].action == "error"
    audit = box_audit_frame(list(decisions.values()))
    assert tuple(audit.columns) == BOX_AUDIT_COLUMNS
    row = audit[audit["study_id"] == first.study_id].iloc[0]
    assert row["box_start"] == " ".join(map(str, first.box[0])) and row["source_shape"] == "16 16 8"
    # subset frames only see their own studies (unmatched patients are not decisions)
    sub, _ = decide_boxes(studies.iloc[:2], boxes, root / "nifti", policy="error")
    assert len(sub) == 2


def test_item_builder_box_items_and_policies(synthetic):
    root, studies = synthetic
    boxes = load_annotation_boxes(root / "boxes_missing_last.csv")
    with pytest.raises(ValueError, match="annotation boxes"):
        ItemBuilder(settings_from_cfg(_cfg(root, box_mode="box_channel")), None, None)
    s = settings_from_cfg(_cfg(root, box_mode="box_channel", box_missing_policy="drop"))
    table = ItemBuilder(s, None, None, annotation_boxes=boxes).build(studies, None)
    assert len(table.items) == len(studies) - 1 and len(table.box_decisions) == len(studies)
    assert table.notes[-1]["note"] == "dropped_by_box_policy"
    assert set(table.items["box_action"]) == {"use"}
    assert isinstance(table.items.iloc[0]["box"], tuple) and table.items.iloc[0]["box_source_shape"] == SHAPE
    s = settings_from_cfg(_cfg(root, box_mode="box_channel", box_missing_policy="full_image"))
    table = ItemBuilder(s, None, None, annotation_boxes=boxes).build(studies, None)
    assert len(table.items) == len(studies) and table.items.iloc[-1]["box_action"] == "full_image"
    assert table.items.iloc[-1]["box"] is None
    s = settings_from_cfg(_cfg(root, box_mode="box_channel"))
    with pytest.raises(BoxPolicyError):
        ItemBuilder(s, None, None, annotation_boxes=boxes).build(studies, None)
    # image-only items are unchanged (no box columns)
    plain = ItemBuilder(settings_from_cfg(_cfg(root)), None, None).build(studies, None)
    assert "box" not in plain.items.columns


def test_expected_channels_box_channel(synthetic):
    root, _ = synthetic
    assert expected_channels(settings_from_cfg(_cfg(root, box_mode="box_channel"))) == 4
    for mode in ("none", "roi_crop", "dual_stream"):
        assert expected_channels(settings_from_cfg(_cfg(root, box_mode=mode))) == 3


# ----------------------------------------------------------------------------- dataset dispatch + tuple plumbing
def _fake_builders(monkeypatch, record=None):
    """Simple stand-ins for the three mode builders (the real ones live in imaging/boxes/<mode>.py)."""
    import mribreast.imaging.boxes.channel as channel
    import mribreast.imaging.boxes.dual as dual
    import mribreast.imaging.boxes.roi_crop as roi_crop

    def fake_roi(ctx: BoxInput, s, log=None):
        if record is not None:
            record.append((ctx.study_id, ctx.box, ctx.source_shape, ctx.native().shape))
        return ctx.target

    def fake_channel(ctx: BoxInput, s, log=None):
        extra = np.full((1,) + ctx.target.shape[1:], 0.0 if ctx.box is None else 1.0, dtype=np.float32)
        return np.concatenate([ctx.target, extra], axis=0)

    def fake_dual(ctx: BoxInput, s, log=None):
        local = ctx.native()[(slice(None),) + tuple(slice(0, n) for n in s.box_local_size)]
        return ctx.target, local

    monkeypatch.setattr(roi_crop, "build_roi_crop", fake_roi)
    monkeypatch.setattr(channel, "build_box_channel", fake_channel)
    monkeypatch.setattr(dual, "build_dual_stream", fake_dual)


def test_dataset_dispatch_and_tuple_collate(synthetic, monkeypatch):
    root, studies = synthetic
    boxes = load_annotation_boxes(root / "boxes.csv")
    record: list = []
    _fake_builders(monkeypatch, record)
    s = settings_from_cfg(_cfg(root, box_mode="roi_crop"))
    table = ItemBuilder(s, None, None, annotation_boxes=boxes).build(studies, None)
    ds = ImageDataset(table.items, s, channels=3)
    log: list = []
    x = ds.load_item(0, log)
    assert x.shape == (3, 16, 16, 8) and log[-1]["op"] == "box_mode" and log[-1]["fallback"] is False
    study, box, shape, native_shape = record[0]
    assert box == boxes[str(studies.iloc[0]["patient_id"])] and shape == SHAPE and native_shape == (3,) + SHAPE

    s = settings_from_cfg(_cfg(root, box_mode="dual_stream", box_local_size="8 8 4"))
    ds = ImageDataset(ItemBuilder(s, None, None, annotation_boxes=boxes).build(studies, None).items, s, channels=3)
    x, y, idx = ds[0]
    assert isinstance(x, tuple) and x[0].shape == (3, 16, 16, 8) and x[1].shape == (3, 8, 8, 4)
    assert x[0].dtype == torch.float32 and x[1].is_contiguous()
    batch = next(iter(DataLoader(ds, batch_size=3)))
    assert isinstance(batch[0], list) and batch[0][0].shape == (3, 3, 16, 16, 8) and batch[0][1].shape[0] == 3
    with pytest.raises(ValueError, match="global stream has 3 channels, expected 4"):
        ImageDataset(ds.items, s, channels=4).load_item(0)

    s = settings_from_cfg(_cfg(root, box_mode="box_channel"))
    ds = ImageDataset(ItemBuilder(s, None, None, annotation_boxes=boxes).build(studies, None).items, s, channels=4)
    assert ds.load_item(0).shape == (4, 16, 16, 8)


def test_image_only_item_is_unchanged_by_box_plumbing(synthetic):
    root, studies = synthetic
    s = settings_from_cfg(_cfg(root))
    ds = ImageDataset(ItemBuilder(s, None, None).build(studies, None).items, s, channels=3)
    x, _, _ = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 16, 16, 8)
    np.testing.assert_array_equal(x.numpy(), ds._study_volume(ds.items.iloc[0]["series_uids"].split("|")).data)


class _TwoInput(nn.Module):
    def __init__(self, channels: int = 3, classes: int = 2):
        super().__init__()
        self.a = nn.Linear(channels, classes)
        self.b = nn.Linear(channels, classes)

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == 2
        return self.a(x[0].mean(dim=(2, 3, 4))) + self.b(x[1].mean(dim=(2, 3, 4)))


def test_map_inputs_and_classifier_step_with_tuple():
    t = torch.ones(2, 3)
    assert map_inputs(t, lambda v: v * 2) is not t and torch.equal(map_inputs(t, lambda v: v * 2), t * 2)
    pair = map_inputs([t, t.double()], lambda v: v.float())
    assert isinstance(pair, tuple) and all(v.dtype == torch.float32 for v in pair)
    module = ImageClassifier(_TwoInput(), 2)
    logits = module((torch.randn(4, 3, 4, 4, 2), torch.randn(4, 3, 2, 2, 2)))
    assert logits.shape == (4, 2)


def test_classifier_training_step_accepts_collated_list(monkeypatch):
    module = ImageClassifier(_TwoInput(), 2)
    monkeypatch.setattr(ImageClassifier, "log", lambda self, *a, **k: None)
    monkeypatch.setattr(ImageClassifier, "trainer", property(lambda self: type("T", (), {"sanity_checking": True})()))
    x = [torch.randn(4, 3, 4, 4, 2), torch.randn(4, 3, 2, 2, 2)]
    loss = module.training_step((x, torch.tensor([0, 1, 0, 1]), torch.arange(4)), 0)
    loss.backward()
    assert module.model.a.weight.grad is not None and module.model.b.weight.grad is not None


def test_predictor_with_tuple_inputs_and_box_inference(synthetic, monkeypatch, tmp_path):
    root, studies = synthetic
    _fake_builders(monkeypatch)
    cfg = _cfg(root, "boxes_missing_last.csv", box_mode="dual_stream", box_local_size="8 8 4",
               box_missing_policy="drop")
    s = settings_from_cfg(cfg)
    boxes = load_annotation_boxes(root / "boxes_missing_last.csv")
    with pytest.raises(ValueError, match="annotation boxes at inference"):
        ImagePredictor(_TwoInput(), s, num_classes=2, channels=3, provider=None, fold=0, meta={},
                       directory=tmp_path)
    predictor = ImagePredictor(_TwoInput(), s, num_classes=2, channels=3, provider=None, fold=0, meta={},
                               directory=tmp_path, annotation_boxes=boxes)
    probs = predictor.predict_proba(studies)
    assert probs.shape == (len(studies), 2)
    assert np.isnan(probs[-1]).all() and np.isfinite(probs[:-1]).all()      # dropped -> NaN row
    assert tuple(predictor.last_audit.columns) == BOX_AUDIT_COLUMNS
    assert predictor.last_audit["action"].tolist()[-1] == "drop"
    assert set(predictor.last_box_coverage["status"]) == {"eligible", "missing_box"}


# ----------------------------------------------------------------------------- branch: fit / meta / load
def test_branch_fit_meta_audit_and_load_checks(synthetic, monkeypatch, tmp_path):
    root, studies = synthetic
    _fake_builders(monkeypatch)
    cfg = _cfg(root, "boxes_missing_last.csv", box_mode="box_channel", box_missing_policy="full_image",
               use_annotation_boxes="true")
    branch = build_image_branch(cfg, num_classes=2)
    out = tmp_path / "img"
    predictor = branch.fit(FoldData(fold=1, fit=studies.iloc[:5], selection=studies.iloc[5:], seed=0,
                                    output_dir=out))
    audit = pd.read_csv(out / "box_audit.csv")
    assert list(audit["split"]) == ["fit"] * 5 + ["selection"] * 3
    assert audit["action"].tolist()[-1] == "full_image"          # selection study without a box
    meta = json.loads((out / "image_branch.json").read_text())
    assert meta["box"]["mode"] == "box_channel" and meta["box"]["requires_box_at_inference"] is True
    assert meta["box"]["coverage"]["selection"]["action_full_image"] == 1
    assert meta["channels"][-1] == "annotation_box" and meta["network"]["in_channels"] == 4
    assert "wrapper" not in meta["network"]
    assert meta["annotation_boxes"]["applies_to"] == "fit rows only"
    check_probabilities(predictor.predict_proba(studies), 2)
    assert predictor.annotation_boxes is not None and len(predictor.last_audit) == len(studies)
    audits = branch.audit(studies)
    assert set(audits["box_audit"]["action"]) == {"use", "full_image"}
    for key, value in (("box_margin", "0.3"), ("box_mode", "roi_crop")):
        other = build_image_branch(_cfg(root, "boxes_missing_last.csv", **{"box_mode": "box_channel",
                                                                            "box_missing_policy": "full_image",
                                                                            key: value}), num_classes=2)
        with pytest.raises(ValueError, match=key):
            other.load(out, 1)


def test_load_accepts_branch_saved_before_box_keys(synthetic, tmp_path):
    root, studies = synthetic
    branch = build_image_branch(_cfg(root), num_classes=2)
    out = tmp_path / "img"
    branch.fit(FoldData(fold=1, fit=studies.iloc[:5], selection=studies.iloc[5:], seed=0, output_dir=out))
    meta = json.loads((out / "image_branch.json").read_text())
    assert meta["box"]["mode"] == "none" and meta["box"]["requires_box_at_inference"] is False
    assert meta["geometry"]["input_shape"] == [3, 16, 16, 8]
    for key in ("box_mode", "box_margin", "box_local_size", "dual_stream_embed_dim", "box_missing_policy"):
        meta["imaging"].pop(key)
    (out / "image_branch.json").write_text(json.dumps(meta))
    check_probabilities(branch.load(out, 1).predict_proba(studies.iloc[:2]), 2)
    dual = build_image_branch(_cfg(root, box_mode="dual_stream"), num_classes=2)
    with pytest.raises(ValueError, match="box_mode"):
        dual.load(out, 1)


def test_dual_stream_meta_records_wrapper(synthetic):
    from mribreast.models.image_branch import _input_shape, _network_meta

    root, _ = synthetic
    s = settings_from_cfg(_cfg(root, box_mode="dual_stream", box_local_size="8 8 4", dual_stream_embed_dim="16"))
    meta = _network_meta({"mri_model": "resnet10"}, s, 3, 3, 2)
    assert meta["wrapper"] == "dual_stream" and meta["dual_stream"]["embed_dim"] == 16
    assert meta["dual_stream"]["local_input"] == [3, 8, 8, 4]
    assert _input_shape(s, 3) == {"global": [3, 16, 16, 8], "local": [3, 8, 8, 4]}
