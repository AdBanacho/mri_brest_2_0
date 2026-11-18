"""Deterministic cache: keys, reuse, corrupt-entry rebuild, shared/exclusive locking across processes."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mribreast.imaging.cache import VolumeCache, cache_key, cache_key_payload, load_entry
from mribreast.imaging.loader import StudyVolumeLoader
from mribreast.imaging.locking import ReadWriteFileLock
from tests.unit.test_imaging_cache_workers import build_counter as _build_counter
from tests.unit.test_imaging_cache_workers import cache_worker, hold_exclusive, try_shared


def _source(tmp_path: Path, name: str = "s.nii.gz", seed: int = 0) -> Path:
    data = np.random.default_rng(seed).normal(100, 5, (8, 6, 4)).astype(np.float32)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.diag([0.7, 0.8, 2.0, 1.0])), str(path))
    return path


def test_keys_separate_options_sources_and_versions(tmp_path):
    a = _source(tmp_path, "a.nii.gz", 0)
    b = _source(tmp_path, "b.nii.gz", 1)
    base = cache_key(cache_key_payload("phase", [("a", a)], {"target_size": [4, 4, 2], "normalisation": "z"}))
    assert base == cache_key(cache_key_payload("phase", [("a", a)], {"target_size": [4, 4, 2], "normalisation": "z"}))
    variants = [
        cache_key_payload("phase", [("a", a)], {"target_size": [4, 4, 4], "normalisation": "z"}),
        cache_key_payload("phase", [("a", a)], {"target_size": [4, 4, 2], "normalisation": "eps"}),
        cache_key_payload("phase", [("a", a)], {"target_size": [4, 4, 2], "normalisation": "z", "resize_backend": "x"}),
        cache_key_payload("difference", [("a", a)], {"target_size": [4, 4, 2], "normalisation": "z"}),
        cache_key_payload("phase", [("a", b)], {"target_size": [4, 4, 2], "normalisation": "z"}),
    ]
    keys = {cache_key(v) for v in variants}
    assert base not in keys and len(keys) == len(variants)
    payload = cache_key_payload("phase", [("a", a)], {})
    payload["cache_schema_version"] = 999
    assert cache_key(payload) != cache_key(cache_key_payload("phase", [("a", a)], {}))


def test_loader_reuses_cache_and_subtraction_uses_its_own_root(tmp_path):
    _source(tmp_path, "u0.nii.gz", 0)
    _source(tmp_path, "u1.nii.gz", 1)
    prepared, subtraction = tmp_path / "prep", tmp_path / "sub"
    loader = StudyVolumeLoader(tmp_path, subtraction_mode="none", channels=2, target_size=(4, 3, 2),
                               prepared_root=prepared, subtraction_root=subtraction)
    first = loader.load(["u0", "u1"])
    entries = sorted(prepared.rglob("*.nii.gz"))
    assert len(entries) == 2 and not subtraction.exists()
    mtimes = [e.stat().st_mtime_ns for e in entries]
    second = loader.load(["u0", "u1"])
    np.testing.assert_array_equal(first.data, second.data)
    assert [e.stat().st_mtime_ns for e in entries] == mtimes
    sub = StudyVolumeLoader(tmp_path, subtraction_mode="consecutive", channels=1, target_size=(4, 3, 2),
                            prepared_root=prepared, subtraction_root=subtraction)
    sub.load(["u0", "u1"])
    assert len(list(subtraction.rglob("*.nii.gz"))) == 1 and len(list(prepared.rglob("*.nii.gz"))) == 2
    # a different target size never reuses the entry
    StudyVolumeLoader(tmp_path, subtraction_mode="none", channels=2, target_size=(2, 3, 2),
                      prepared_root=prepared).load(["u0", "u1"])
    assert len(list(prepared.rglob("*.nii.gz"))) == 4
    data, affine = load_entry(entries[0])
    assert data.shape == (4, 3, 2) and affine.shape == (4, 4)


@pytest.mark.parametrize("damage", ["truncate", "flip_payload"])
def test_corrupt_entry_is_rebuilt_from_source(tmp_path, damage):
    source = _source(tmp_path)
    cache = VolumeCache(tmp_path / "cache")
    payload = cache_key_payload("phase", [("uid", source)], {"t": 1})
    counter = tmp_path / "count.txt"
    original = cache.get_or_build("phase", payload, _build_counter(counter), source=str(source))
    path = original.path
    raw = path.read_bytes()
    if damage == "truncate":
        path.write_bytes(raw[: len(raw) // 2])
    else:  # valid gzip/NIfTI with different voxels -> checksum mismatch
        img = nib.load(str(path))
        bad = nib.Nifti1Image(np.zeros(img.shape, np.float32), img.affine, img.header)
        nib.save(bad, str(path))
    source_bytes = source.read_bytes()
    with pytest.warns(RuntimeWarning, match="Rebuilding corrupt MRI cache"):
        rebuilt = cache.get_or_build("phase", payload, _build_counter(counter), source=str(source))
    np.testing.assert_array_equal(rebuilt.data, original.data)
    assert rebuilt.built and counter.read_text() == "xx"
    assert source.read_bytes() == source_bytes                     # source stays read-only
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp") or p.name.startswith(".")]


def test_permission_error_is_not_treated_as_corruption(tmp_path, monkeypatch):
    source = _source(tmp_path)
    cache = VolumeCache(tmp_path / "cache")
    payload = cache_key_payload("phase", [("uid", source)], {})
    counter = tmp_path / "c.txt"
    cache.get_or_build("phase", payload, _build_counter(counter))

    def denied(path):
        raise PermissionError("denied")

    monkeypatch.setattr("mribreast.imaging.cache.load_entry", denied)
    with pytest.raises(RuntimeError, match="cannot read MRI cache"):
        cache.get_or_build("phase", payload, _build_counter(counter))
    assert counter.read_text() == "x"


def test_failed_build_publishes_nothing(tmp_path):
    source = _source(tmp_path)
    cache = VolumeCache(tmp_path / "cache")
    payload = cache_key_payload("phase", [("uid", source)], {})

    def boom():
        raise OSError("disk full")

    with pytest.raises(RuntimeError, match="cannot create MRI cache"):
        cache.get_or_build("phase", payload, boom, source=str(source))
    assert not list((tmp_path / "cache").rglob("*.nii.gz"))


def test_concurrent_processes_build_once_and_all_read_complete_entry(tmp_path):
    source = _source(tmp_path)
    counter = tmp_path / "count.txt"
    counter.write_text("", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    results, start = context.Queue(), context.Event()
    workers = [context.Process(target=cache_worker,
                               args=(str(tmp_path / "cache"), str(counter), str(source), start, results))
               for _ in range(6)]
    for process in workers:
        process.start()
    start.set()
    try:
        values = [results.get(timeout=120) for _ in workers]
    finally:
        for process in workers:
            process.join(timeout=60)
    assert values == [3.0 * 32] * len(workers), values
    assert counter.read_text() == "x"                               # exactly one writer built the entry
    assert all(p.exitcode == 0 for p in workers)


def test_shared_lock_waits_for_writer_and_readers_share(tmp_path):
    lock_path = str(tmp_path / "entry.lock")
    context = multiprocessing.get_context("spawn")
    ready, release, acquired = context.Event(), context.Event(), context.Event()
    writer = context.Process(target=hold_exclusive, args=(lock_path, ready, release))
    writer.start()
    assert ready.wait(60)
    reader = context.Process(target=try_shared, args=(lock_path, acquired))
    reader.start()
    assert not acquired.wait(1.0)                                    # blocked by the writer
    release.set()
    assert acquired.wait(60)
    writer.join(30)
    reader.join(30)
    # two readers at once (same process, separate handles)
    lock = ReadWriteFileLock(lock_path)
    with lock.shared():
        second = context.Event()
        other = context.Process(target=try_shared, args=(lock_path, second))
        other.start()
        assert second.wait(60)                                       # shared does not block shared
        other.join(30)
    assert os.path.exists(lock_path)
