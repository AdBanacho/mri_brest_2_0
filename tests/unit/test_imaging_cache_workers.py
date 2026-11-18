"""Child-process helpers for test_imaging_cache (kept free of torch imports so spawn stays fast).

Contains no tests; pytest collects nothing here.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mribreast.imaging.cache import VolumeCache, cache_key_payload
from mribreast.imaging.locking import ReadWriteFileLock


def build_counter(counter_file: Path, delay: float = 0.0):
    def build():
        with open(counter_file, "a", encoding="utf-8") as handle:
            handle.write("x")
        time.sleep(delay)
        return np.full((4, 4, 2), 3.0, np.float32), np.eye(4)
    return build


def cache_worker(root: str, counter_file: str, source: str, start, results) -> None:
    try:
        start.wait(60)
        cache = VolumeCache(root)
        payload = cache_key_payload("phase", [("uid", source)], {"target_size": [4, 4, 2]})
        result = cache.get_or_build("phase", payload, build_counter(Path(counter_file), 0.3), source=source)
        results.put(float(result.data.sum()))
    except Exception as error:  # pragma: no cover - reported to the parent
        results.put(repr(error))


def hold_exclusive(lock_path: str, ready, release) -> None:
    with ReadWriteFileLock(lock_path).exclusive():
        ready.set()
        release.wait(30)


def try_shared(lock_path: str, acquired) -> None:
    with ReadWriteFileLock(lock_path).shared():
        acquired.set()
