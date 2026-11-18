"""Parity fixtures: skip without the reference repo; guarantee it is never written to."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import legacy_loader  # noqa: E402


@pytest.fixture(scope="session")
def legacy_repo():
    repo = legacy_loader.legacy_repo()
    if repo is None:
        pytest.skip(f"reference repository not found (set {legacy_loader.ENV_VAR})")
    before = legacy_loader.tree_state(repo)
    yield repo
    after = legacy_loader.tree_state(repo)
    assert after == before, f"parity tests modified the reference repo: {sorted(after ^ before)[:10]}"


@pytest.fixture(scope="session")
def legacy_head(legacy_repo):
    return legacy_loader.legacy_sha(legacy_repo)
