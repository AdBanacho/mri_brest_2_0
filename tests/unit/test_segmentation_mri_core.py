"""Real MRI-CORE / SAM weights (docs/segmentation-mri-core.md §2.2-2.4).

Skipped unless the read-only ``mri_foundation`` checkout (``$MRI_FOUNDATION_REPO``, else
``C:\\code\\mri_foundation``) with both weight files and the ``.[segment]`` extras (einops, torchvision) are
available. CPU is enough: a few ViT-B forward passes at 1024².
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(os.environ.get("MRI_FOUNDATION_REPO", r"C:\code\mri_foundation"))
SAM = REPO / "pretrained_weights" / "sam_vit_b_01ec64.pth"
CORE = REPO / "pretrained_weights" / "mri_foundation.pth"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (SAM.is_file() and CORE.is_file()), reason=f"MRI-CORE weights not found under {REPO}"),
    pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in ("torch", "einops", "torchvision")),
                       reason="needs the .[segment] extras (einops, torchvision)"),
]


def _repo_state() -> tuple[str, list[str]]:
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
                            capture_output=True, text=True).stdout
    return status, sorted(str(p) for p in REPO.rglob("__pycache__"))


@pytest.fixture(scope="module", autouse=True)
def repo_untouched():
    before = _repo_state()
    yield
    assert _repo_state() == before, "the mri_foundation checkout was modified"


def test_loader_matches_the_repos_own_builder(monkeypatch):
    """Our loader == ``sam_model_registry['vit_b'](..., num_classes=3, pretrained_sam=True)`` weight for weight."""
    import torch

    from mribreast.prepare.segmentation.mri_core import build_model

    ours, info = build_model(REPO, "mri_core", sam_checkpoint=SAM, mri_core_checkpoint=CORE)
    assert info["n_mri_core_tensors"] == 146 and info["pos_embed_source"] == "sam"

    original = torch.load
    monkeypatch.setattr(torch, "load", lambda f, map_location=None, **kw: original(f, map_location="cpu",
                                                                                     weights_only=True))
    monkeypatch.chdir(REPO)                          # the repo reads pretrained_weights/ relative to the cwd
    monkeypatch.syspath_prepend(str(REPO))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    try:
        from models.sam import sam_model_registry

        args = argparse.Namespace(if_encoder_adapter=False, encoder_adapter_depths=[], if_mask_decoder_adapter=False,
                                  decoder_adapt_depth=2, arch="vit_b")
        theirs = sam_model_registry["vit_b"](args, checkpoint=str(CORE), num_classes=3, image_size=1024,
                                             pretrained_sam=True)
    finally:
        for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
            del sys.modules[name]
    a, b = ours.state_dict(), theirs.state_dict()
    assert a.keys() == b.keys()
    assert [k for k in a if not torch.equal(a[k], b[k])] == []


@pytest.mark.parametrize("weights", ["mri_core", "sam"])
def test_zero_shot_box_prompt_on_a_toy_lesion(weights):
    """Mechanics check only (toy image, not MRI): a box prompt must yield the lesion, not the background."""
    from scipy.ndimage import gaussian_filter

    from mribreast.prepare.segmentation.mri_core import MriCoreSegmenter
    from mribreast.prepare.segmentation.pipeline import SegmentSettings, segment_planes

    rng = np.random.default_rng(0)
    y, x = np.mgrid[:256, :256]
    lesion = ((y - 140) / 22.0) ** 2 + ((x - 120) / 30.0) ** 2 <= 1
    plane = (gaussian_filter(rng.normal(0, 1, (256, 256)), 12) * 0.25 + 0.35
             + 0.6 * gaussian_filter(lesion.astype(float), 1.5) + rng.normal(0, 0.04, (256, 256))).astype(np.float32)
    ys, xs = np.where(lesion)
    box2d = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    segmenter = MriCoreSegmenter(REPO, weights, sam_checkpoint=SAM, mri_core_checkpoint=CORE, device="cpu")
    (outcome,) = segment_planes([plane], box2d, SegmentSettings(), segmenter, float(plane.min()))
    dice = 2 * (outcome.mask & lesion).sum() / (outcome.mask.sum() + lesion.sum())
    assert dice > 0.85, dice
    assert segmenter.describe()["repo_revision"]["sha"] != "unknown"
