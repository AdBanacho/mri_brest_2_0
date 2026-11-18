"""mribreast.legacy.checkpoints: legacy filename eras, ranking key and fold inventory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mribreast.legacy import checkpoints as ck

AUC = "best-epoch=07-val_auc_roc=0.7500-val_balanced_accuracy=0.6058-val_sensitivity=0.2500.ckpt"
AUC_V1 = "best-epoch=09-val_auc_roc=0.7500-val_balanced_accuracy=0.6100-val_sensitivity=0.2500-v1.ckpt"
BAL = "best-epoch=03-val_balanced_accuracy=0.9000-val_sensitivity=0.9000-val_auc_roc=0.9500.ckpt"
SENS = "best-epoch=02-val_sensitivity=1.0000-val_auc_roc=0.6000.ckpt"
SENS_NO_AUC = "best-epoch=05-val_sensitivity=0.8000.ckpt"
UNKNOWN = "best-something.ckpt"


def _touch(directory: Path, name: str, mtime: float) -> Path:
    path = directory / name
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))
    return path


def test_parse_eras():
    assert ck.parse_checkpoint_name(AUC).format == ck.FORMAT_AUC
    parsed = ck.parse_checkpoint_name(AUC_V1)
    assert (parsed.format, parsed.epoch, parsed.balanced_accuracy, parsed.version) == (ck.FORMAT_AUC, 9, 0.61, 1)
    assert ck.parse_checkpoint_name(BAL).format == ck.FORMAT_BALANCED
    assert ck.parse_checkpoint_name(SENS).format == ck.FORMAT_SENSITIVITY
    assert ck.parse_checkpoint_name(SENS_NO_AUC).auc_roc is None
    assert ck.parse_checkpoint_name("/a/b/" + UNKNOWN).format == ck.FORMAT_UNKNOWN
    assert ck.parse_checkpoint_name("last.ckpt").format == ck.FORMAT_UNKNOWN


def test_score_tuples_match_legacy_definition():
    assert ck.score_from_name(AUC, 5.0) == (1, 2, 0.75, 0.6058, 0.25, 7, 5.0)
    assert ck.score_from_name(BAL, 5.0) == (1, 1, 0.9, 0.9, 0.95, 3, 5.0)
    assert ck.score_from_name(SENS, 5.0) == (1, 0, 1.0, 0.6, float("-inf"), 2, 5.0)
    assert ck.score_from_name(SENS_NO_AUC, 5.0) == (1, 0, 0.8, float("-inf"), float("-inf"), 5, 5.0)
    assert ck.score_from_name(UNKNOWN, 5.0) == (0, 0, float("-inf"), float("-inf"), float("-inf"), -1, 5.0)


def test_ranking_prefers_current_era_then_metrics_then_epoch_then_mtime(tmp_path):
    paths = [_touch(tmp_path, n, 1000 + i) for i, n in enumerate([UNKNOWN, SENS, BAL, AUC, AUC_V1, SENS_NO_AUC])]
    ranked = [p.name for p in ck.rank_checkpoints(paths)]
    # AUC_V1 ties AUC on auc, wins on balanced accuracy; BAL (older era) never beats an AUC file.
    assert ranked == [AUC_V1, AUC, BAL, SENS, SENS_NO_AUC, UNKNOWN]
    same = [_touch(tmp_path, "best-epoch=01-val_auc_roc=0.5000-val_balanced_accuracy=0.5000-val_sensitivity=0.5000.ckpt", 10),
            _touch(tmp_path, "best-epoch=01-val_auc_roc=0.5000-val_balanced_accuracy=0.5000-val_sensitivity=0.5000-v2.ckpt", 20)]
    assert ck.rank_checkpoints(same)[0].name.endswith("-v2.ckpt")  # newer mtime breaks the full tie
    assert ck.find_best_checkpoint(tmp_path).name == AUC_V1


def test_find_best_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ck.find_best_checkpoint(tmp_path)


def test_fold_inventory_flags(tmp_path):
    fold = ck.legacy_fold_checkpoint_dir(tmp_path, "Exp", 1)
    fold.mkdir(parents=True)
    _touch(fold, AUC, 1)
    (fold / "inner_calibration_split.csv").write_text(
        "studyId,patientId,label,split,requested_inner_folds,effective_inner_folds\n1.2,P1,0,fit,5,5\n", encoding="utf-8")
    (fold / "tabular_preprocessor.joblib").write_bytes(b"")
    (fold / "xgboost_model.joblib").write_bytes(b"")
    inv = ck.inventory_fold(fold, 1, expect_tabular="xgboost")
    assert inv.has_split_metadata and not inv.requires_retraining and inv.tabular_models.keys() == {"xgboost"}
    old = ck.legacy_fold_checkpoint_dir(tmp_path, "Old", 2)
    old.mkdir(parents=True)
    _touch(old, BAL, 1)
    inv_old = ck.inventory_fold(old, 2, expect_tabular="mlp", expect_selector=True)
    assert set(inv_old.reasons) >= {"checkpoint_format_balanced_accuracy", "split_metadata_missing",
                                    "tabular_model_missing:mlp", "tabular_selector_missing"}
    assert ck.inventory_fold(tmp_path / "nope", 3).reasons == ["fold_directory_missing"]
    rows = [i.as_row() for i in ck.inventory_experiment(tmp_path, "Exp", 2)]
    assert rows[0]["requires_retraining"] is False and rows[1]["reasons"] == "fold_directory_missing"


def test_historical_dir_resolution(tmp_path):
    (tmp_path / "fold_2" / "checkpoints").mkdir(parents=True)
    assert ck.resolve_historical_fold_dir(tmp_path, "Model", 2) == tmp_path / "fold_2" / "checkpoints"
    assert ck.resolve_historical_fold_dir(tmp_path, "Model", 1) == tmp_path / "Model" / "fold_1" / "checkpoints"


def test_network_state_strip_prefix():
    state = {"model.features.conv0.weight": 1, "model.class_layers.out.bias": 2, "class_weights": 3}
    assert ck.legacy_network_state(state) == {"features.conv0.weight": 1, "class_layers.out.bias": 2}
    with pytest.raises(ValueError):
        ck.legacy_network_state({"class_weights": 1})


def test_load_legacy_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    net = torch.nn.Linear(3, 2)
    payload = {"epoch": 4, "global_step": 40, "state_dict": {**{f"model.{k}": v for k, v in net.state_dict().items()},
                                                              "class_weights": torch.tensor([1.0, 2.0])}}
    path = tmp_path / AUC
    torch.save(payload, path)
    state, extras = ck.load_legacy_network_state(path)
    fresh = torch.nn.Linear(3, 2)
    fresh.load_state_dict(state, strict=True)
    assert extras["epoch"] == 4 and extras["name"].format == ck.FORMAT_AUC
    assert torch.equal(fresh.weight, net.weight)
