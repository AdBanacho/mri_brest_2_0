"""Atomic publication, completion flags, fingerprints."""

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from mribreast.artifacts import (atomic_dir, file_fingerprint, is_complete, mark_complete, paths_fingerprint,
                                 read_json, write_json_atomic)


def test_atomic_dir_publishes_only_on_success(tmp_path):
    final = tmp_path / "fold_1"
    with pytest.raises(RuntimeError):
        with atomic_dir(final) as staging:
            (staging / "a.txt").write_text("x")
            raise RuntimeError("crash")
    assert not final.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".fold_1.staging")]
    with atomic_dir(final) as staging:
        (staging / "a.txt").write_text("y")
        mark_complete(staging, {"fold": 1})
    assert is_complete(final) and (final / "a.txt").read_text() == "y"
    with pytest.raises(FileExistsError):
        with atomic_dir(final):
            pass


def test_atomic_dir_replaces_incomplete(tmp_path):
    final = tmp_path / "fold_2"
    final.mkdir()
    (final / "partial.bin").write_text("old")
    with atomic_dir(final) as staging:
        (staging / "new.bin").write_text("new")
        mark_complete(staging)
    assert not (final / "partial.bin").exists() and is_complete(final)


def _writer(path, value):
    for _ in range(50):
        write_json_atomic(path, {"value": value, "pad": "x" * 10000})


def test_concurrent_json_writers_never_leave_torn_files(tmp_path):
    target = tmp_path / "m.json"
    procs = [mp.Process(target=_writer, args=(target, i)) for i in range(4)]
    for p in procs:
        p.start()
    for _ in range(200):
        if target.exists():
            try:
                data = read_json(target)
            except PermissionError:  # Windows: os.replace racing a reader
                continue
            assert data["pad"] == "x" * 10000
    for p in procs:
        p.join()
    assert json.loads(target.read_text())["value"] in range(4)


def test_fingerprints(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("1,2\n")
    d = tmp_path / "d"
    d.mkdir()
    (d / "x.nii.gz").write_bytes(b"abc")
    fp = paths_fingerprint({"file": f, "dir": d, "none": None, "missing": tmp_path / "zz"})
    assert fp["file"] == file_fingerprint(f) and fp["file"].startswith("sha256:")
    assert fp["dir"].startswith("dir-listing:") and fp["none"] == "absent" and fp["missing"] == "absent"
    (d / "y.nii.gz").write_bytes(b"abcd")
    assert paths_fingerprint({"dir": d})["dir"] != fp["dir"]
