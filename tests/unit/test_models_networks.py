"""Networks, legacy loss/class weights and protected trainer options."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mribreast.models.classifier import (PROTECTED_TRAINER_KEYS, compute_balanced_class_weights,
                                         soft_sensitivity_loss, trainer_options)
from mribreast.models.networks import MRI_MODELS, Simple3DFCN, make_network


@pytest.mark.parametrize("name", MRI_MODELS)
@pytest.mark.parametrize("dims", [2, 3])
def test_networks_shapes(name, dims):
    size = ((32, 32, 32) if name == "densenet121" else (32, 32, 16)) if dims == 3 else (32, 32)
    network = make_network(name, spatial_dims=dims, in_channels=4, num_classes=3)
    out = network(torch.zeros((2, 4) + size))
    assert tuple(out.shape) == (2, 3)


def test_extra_kwargs_and_fcn_port():
    net = make_network("densenet121", spatial_dims=2, in_channels=1, num_classes=2, extra={"dropout_prob": 0.3})
    assert any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p == 0.3 for m in net.modules())
    fcn = Simple3DFCN(num_classes=2, in_channels=5)
    assert fcn.conv1.in_channels == 5 and not any(isinstance(m, torch.nn.Linear) for m in fcn.modules())
    with pytest.raises(ValueError):
        make_network("vit", spatial_dims=3, in_channels=1, num_classes=2)


def test_balanced_class_weights_match_legacy():
    labels = [0, 0, 0, 1]
    weights = compute_balanced_class_weights(labels, 2, positive_boost=2.0)
    torch.testing.assert_close(weights, torch.tensor([4 / (2 * 3), 4 / (2 * 1) * 2.0]))
    # absent highest class keeps the vector length (legacy bincount would shrink it)
    assert compute_balanced_class_weights([0, 0], 2).shape == (2,)
    assert compute_balanced_class_weights([0, 1, 2, 2], 3, positive_boost=5.0).tolist() == pytest.approx(
        [4 / 3, 4 / 3, 4 / 6])


def test_soft_sensitivity_loss_matches_legacy_formula():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1])
    weights = torch.tensor([1.5, 0.75])
    total, ce, sens_loss, soft = soft_sensitivity_loss(logits, y, num_classes=2, class_weights=weights,
                                                       sensitivity_lambda=0.2)
    p1 = torch.softmax(logits, 1)[:, 1]
    expected_soft = (p1[1] + p1[2]) / (2 + 1e-6)
    torch.testing.assert_close(soft, expected_soft)
    torch.testing.assert_close(ce, F.cross_entropy(logits, y, weight=weights))
    torch.testing.assert_close(total, ce + 0.2 * (1 - expected_soft))
    total_ce, ce2, _, _ = soft_sensitivity_loss(logits, y, num_classes=2, class_weights=weights,
                                                sensitivity_lambda=0.2, use_sensitivity=False)
    torch.testing.assert_close(total_ce, ce2)
    _, _, _, no_pos = soft_sensitivity_loss(logits, torch.tensor([0, 0, 0]), num_classes=2, class_weights=None,
                                            sensitivity_lambda=0.2)
    assert float(no_pos) == 1.0
    multi = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _, _, _, macro = soft_sensitivity_loss(multi, torch.tensor([0, 1]), num_classes=3, class_weights=None,
                                           sensitivity_lambda=0.1)
    p = torch.softmax(multi, 1)
    torch.testing.assert_close(macro, (p[0, 0] / (1 + 1e-6) + p[1, 1] / (1 + 1e-6)) / 2)


def test_protected_trainer_keys_cannot_be_overridden():
    base = {"epoch": 3, "trainer_extra_json": {"limit_train_batches": 2}}
    options = trainer_options(base, logger=False, callbacks=["cb"], default_root_dir="root", accelerator="cpu")
    assert options["callbacks"] == ["cb"] and options["max_epochs"] == 3 and options["limit_train_batches"] == 2
    assert options["enable_checkpointing"] is True and options["default_root_dir"] == "root"
    for key in sorted(PROTECTED_TRAINER_KEYS):
        with pytest.raises(ValueError, match="cannot override"):
            trainer_options({"trainer_extra_json": {key: None}}, logger=False, callbacks=[], default_root_dir="r")
