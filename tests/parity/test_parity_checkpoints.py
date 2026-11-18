"""Parity: legacy checkpoint_selection.py vs mribreast.legacy.checkpoints (exact, no tolerance)."""

from __future__ import annotations

import os
import random

import pytest

from legacy_loader import load_file_module
from mribreast.legacy import checkpoints as ck

pytestmark = pytest.mark.parity


def _names(rng: random.Random, n: int) -> list[str]:
    names = set()
    while len(names) < n:
        era = rng.choice(["auc", "auc", "bal", "bal_noauc", "sens", "sens_noauc", "junk"])
        e = rng.randint(0, 40)
        auc, bal, sens = (rng.choice([0.5, 0.6, round(rng.random(), 4)]) for _ in range(3))
        version = rng.choice(["", "", "-v1", "-v2"])
        name = {
            "auc": f"best-epoch={e:02d}-val_auc_roc={auc:.4f}-val_balanced_accuracy={bal:.4f}-val_sensitivity={sens:.4f}",
            "bal": f"best-epoch={e:02d}-val_balanced_accuracy={bal:.2f}-val_sensitivity={sens:.2f}-val_auc_roc={auc:.2f}",
            "bal_noauc": f"best-epoch={e:02d}-val_balanced_accuracy={bal:.2f}-val_sensitivity={sens:.2f}",
            "sens": f"best-epoch={e:02d}-val_sensitivity={sens:.2f}-val_auc_roc={auc:.2f}",
            "sens_noauc": f"best-epoch={e}-val_sensitivity={sens:.3f}",
            "junk": f"best-{rng.randint(0, 999)}",
        }[era]
        names.add(name + version + ".ckpt")
    return sorted(names)


@pytest.mark.parametrize("seed", range(5))
def test_rank_and_score_identical(legacy_repo, tmp_path, seed):
    legacy = load_file_module(legacy_repo, "mriBreastDuke/checkpoint_selection.py")
    rng = random.Random(seed)
    paths = []
    for name in _names(rng, 40):
        path = tmp_path / name
        path.write_bytes(b"")
        mtime = 1_700_000_000 + rng.choice([0, 1, 2])          # coarse mtimes -> exercise the mtime tie-break
        os.utime(path, (mtime, mtime))
        paths.append(path)
    rng.shuffle(paths)
    for path in paths:
        assert ck.checkpoint_score(path) == legacy.checkpoint_score(path), path.name
    assert ck.rank_checkpoints(paths) == legacy.rank_checkpoints(paths)


def test_patterns_are_identical(legacy_repo):
    legacy = load_file_module(legacy_repo, "mriBreastDuke/checkpoint_selection.py")
    assert ck.AUC_CHECKPOINT_PATTERN.pattern == legacy.AUC_CHECKPOINT_PATTERN.pattern
    assert ck.BALANCED_CHECKPOINT_PATTERN.pattern == legacy.BALANCED_CHECKPOINT_PATTERN.pattern
