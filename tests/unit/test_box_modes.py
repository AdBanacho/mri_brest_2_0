"""Real (non-stub) behaviour of the three ``box_mode`` builders and the dual-stream network."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mribreast.imaging.boxes import BoxInput
from mribreast.imaging.boxes.channel import build_box_channel
from mribreast.imaging.boxes.dual import build_dual_stream
from mribreast.imaging.boxes.roi_crop import build_roi_crop
from mribreast.models.dual_stream import DualStreamNetwork
from mribreast.models.networks import MRI_MODELS

NATIVE_SHAPE = (20, 12, 10)
BOX = ((4, 2, 3), (10, 6, 7))        # extent (6, 4, 4)


class S:
    """Minimal stand-in for the ImagingSettings fields the builders read."""

    def __init__(self, target_size=(8, 8, 4), box_margin=0.0, box_local_size=(6, 4, 4)):
        self.target_size, self.box_margin, self.box_local_size = target_size, box_margin, box_local_size


def _native(channels=2):
    return np.arange(channels * np.prod(NATIVE_SHAPE), dtype=np.float32).reshape((channels,) + NATIVE_SHAPE)


def _ctx(target_size=(8, 8, 4), box=BOX, source_shape=NATIVE_SHAPE, calls=None):
    native = _native()

    def load():
        if calls is not None:
            calls.append(1)
        return native

    target = np.full((2,) + tuple(target_size), 7.0, dtype=np.float32)
    return BoxInput(target=target, native=load, box=box, source_shape=None if box is None else source_shape,
                    study_id="s1")


# ----------------------------------------------------------------------------- roi_crop
def test_roi_crop_takes_native_box_region():
    # target size == box extent -> resize is the identity, so the crop content is checked exactly
    out = build_roi_crop(_ctx(), S(target_size=(6, 4, 4), box_margin=0.0))
    assert out.shape == (2, 6, 4, 4) and out.dtype == np.float32
    np.testing.assert_allclose(out, _native()[:, 4:10, 2:6, 3:7], rtol=1e-6)


def test_roi_crop_margin_is_clipped_and_resized():
    log: list = []
    out = build_roi_crop(_ctx(), S(target_size=(8, 8, 4), box_margin=1.0), log)
    entry = next(e for e in log if e["op"] == "box_roi_crop")
    # margin 1.0 = one extent per side: (4-6, 10+6) -> (0, 16); (2-4, 6+4) -> (0, 10); (3-4, 7+4) -> (0, 10)
    assert entry["crop"] == [[0, 0, 0], [16, 10, 10]]
    assert out.shape == (2, 8, 8, 4)


def test_roi_crop_fallback_returns_target_without_loading_native():
    calls: list = []
    ctx = _ctx(box=None, calls=calls)
    out = build_roi_crop(ctx, S())
    np.testing.assert_array_equal(out, ctx.target)
    assert calls == []


def test_roi_crop_rejects_grid_mismatch():
    with pytest.raises(ValueError, match="source_shape"):
        build_roi_crop(_ctx(source_shape=(21, 12, 10)), S())


# ----------------------------------------------------------------------------- box_channel
def test_box_channel_appends_outward_rounded_binary_channel():
    calls: list = []
    out = build_box_channel(_ctx(target_size=(8, 8, 4), calls=calls), S(target_size=(8, 8, 4)))
    assert out.shape == (3, 8, 8, 4) and calls == []
    np.testing.assert_array_equal(out[:2], 7.0)
    expected = np.zeros((8, 8, 4), dtype=np.float32)
    # x: 4*8/20=1.6 -> 1, 10*8/20=4 -> 4 ; y: 2*8/12 -> 1, 6*8/12=4 -> 4 ; z: 3*4/10 -> 1, 7*4/10=2.8 -> 3
    expected[1:4, 1:4, 1:3] = 1.0
    np.testing.assert_array_equal(out[2], expected)


def test_box_channel_never_loses_a_lesion_voxel():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lo = [int(rng.integers(0, n - 1)) for n in NATIVE_SHAPE]
        hi = [int(rng.integers(a + 1, n + 1)) for a, n in zip(lo, NATIVE_SHAPE)]
        box = (tuple(lo), tuple(hi))
        out = build_box_channel(_ctx(box=box, target_size=(7, 5, 3)), S(target_size=(7, 5, 3)))
        # every native lesion voxel centre must land inside the marked target region
        idx = np.argwhere(out[2] > 0)
        t_lo, t_hi = idx.min(0), idx.max(0) + 1
        for a, (n, t) in enumerate(zip(NATIVE_SHAPE, (7, 5, 3))):
            first, last = (lo[a] + 0.5) * t / n, (hi[a] - 0.5) * t / n
            assert t_lo[a] <= np.floor(first) and np.floor(last) < t_hi[a]


def test_training_erase_never_overlaps_box_channel():
    from mribreast.imaging.annotations import erase_background_box, resize_box

    target_size = (8, 8, 4)
    lesion = resize_box(BOX, NATIVE_SHAPE, target_size)    # what ImageDataset passes to the eraser
    for seed in range(30):
        volume = np.ones((2,) + target_size, dtype=np.float32)
        erased, origin = erase_background_box(volume, lesion, np.random.default_rng(seed))
        out = build_box_channel(BoxInput(target=erased, native=lambda: None, box=BOX,
                                         source_shape=NATIVE_SHAPE), S(target_size=target_size))
        assert (out[0][out[2] > 0] == 1.0).all()            # no erased voxel inside the marked box


def test_box_channel_fallback_is_zeros():
    out = build_box_channel(_ctx(box=None), S())
    assert out.shape == (3, 8, 8, 4) and not out[2].any()


# ----------------------------------------------------------------------------- dual_stream inputs
def test_dual_stream_global_is_target_and_local_is_native_crop():
    ctx = _ctx()
    g, loc = build_dual_stream(ctx, S(box_local_size=(6, 4, 4), box_margin=0.0))
    np.testing.assert_array_equal(g, ctx.target)
    np.testing.assert_allclose(loc, _native()[:, 4:10, 2:6, 3:7], rtol=1e-6)


def test_dual_stream_fallback_local_is_full_native_resized():
    g, loc = build_dual_stream(_ctx(box=None), S(box_local_size=(5, 3, 2)))
    assert loc.shape == (2, 5, 3, 2) and g.shape == (2, 8, 8, 4)


def test_dual_stream_is_deterministic():
    a = build_dual_stream(_ctx(), S(box_margin=0.3))
    b = build_dual_stream(_ctx(), S(box_margin=0.3))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


# ----------------------------------------------------------------------------- dual_stream network
@pytest.mark.parametrize("model", MRI_MODELS)
def test_dual_stream_network_forward_backward(model):
    torch.manual_seed(0)
    # densenet121 downsamples 32x; give it enough voxels in both streams
    g_size, l_size = ((32, 32, 32), (32, 32, 32)) if model == "densenet121" else ((16, 16, 8), (8, 8, 8))
    net = DualStreamNetwork(model, spatial_dims=3, in_channels=2, num_classes=3, embed_dim=8)
    x = (torch.randn(2, 2, *g_size), torch.randn(2, 2, *l_size))
    logits = net(x)
    assert logits.shape == (2, 3)
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.global_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.local_encoder.parameters())
    assert net(list(x)).shape == (2, 3)


def test_dual_stream_network_has_independent_encoders_and_validates_input():
    net = DualStreamNetwork("fcn", spatial_dims=3, in_channels=1, num_classes=2, embed_dim=4)
    g_params = {id(p) for p in net.global_encoder.parameters()}
    assert g_params.isdisjoint(id(p) for p in net.local_encoder.parameters())
    with pytest.raises(TypeError):
        net(torch.randn(1, 1, 8, 8, 8))
    with pytest.raises(ValueError, match="batch sizes"):
        net((torch.randn(2, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8)))
    with pytest.raises(ValueError):
        DualStreamNetwork("fcn", spatial_dims=3, in_channels=1, num_classes=2, embed_dim=0)
