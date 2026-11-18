"""Portable shared (reader) / exclusive (writer) inter-process file locks.

* POSIX: ``fcntl.flock`` with ``LOCK_SH`` / ``LOCK_EX``.
* Windows: ``LockFileEx`` (shared unless ``LOCKFILE_EXCLUSIVE_LOCK``) via ctypes.
* Fallback (neither available): ``filelock.FileLock`` -- exclusive for readers
  too, i.e. correct but without reader parallelism (the legacy behaviour).

Locks are advisory and block until acquired. They protect a cache entry while a
reader has the published file open (Windows cannot ``os.replace`` an open file)
and serialise writers so an entry is built once.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

try:  # POSIX
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

_WINDOWS = os.name == "nt"


def _windows_lock(fd: int, exclusive: bool) -> None:  # pragma: no cover - platform specific
    import ctypes
    import ctypes.wintypes as wt
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wt.DWORD), ("OffsetHigh", wt.DWORD), ("hEvent", wt.HANDLE)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = wt.HANDLE(msvcrt.get_osfhandle(fd))
    flags = 0x2 if exclusive else 0x0   # LOCKFILE_EXCLUSIVE_LOCK
    overlapped = _Overlapped()
    if not kernel32.LockFileEx(handle, wt.DWORD(flags), wt.DWORD(0), wt.DWORD(1), wt.DWORD(0),
                               ctypes.byref(overlapped)):
        raise OSError(ctypes.get_last_error(), "LockFileEx failed")


def _windows_unlock(fd: int) -> None:  # pragma: no cover - platform specific
    import ctypes
    import ctypes.wintypes as wt
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wt.DWORD), ("OffsetHigh", wt.DWORD), ("hEvent", wt.HANDLE)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = wt.HANDLE(msvcrt.get_osfhandle(fd))
    overlapped = _Overlapped()
    kernel32.UnlockFileEx(handle, wt.DWORD(0), wt.DWORD(1), wt.DWORD(0), ctypes.byref(overlapped))


class ReadWriteFileLock:
    """``with lock.shared(): ...`` for readers, ``with lock.exclusive(): ...`` for writers."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    @property
    def backend(self) -> str:
        if fcntl is not None:
            return "fcntl.flock"
        if _WINDOWS:
            return "LockFileEx"
        return "filelock.FileLock(exclusive)"

    @contextlib.contextmanager
    def _acquire(self, exclusive: bool) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None and not _WINDOWS:  # pragma: no cover - exotic platforms
            from filelock import FileLock

            with FileLock(str(self.path)):
                yield
            return
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o666)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            else:
                _windows_lock(fd, exclusive)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    _windows_unlock(fd)
        finally:
            os.close(fd)

    def shared(self) -> contextlib.AbstractContextManager[None]:
        return self._acquire(False)

    def exclusive(self) -> contextlib.AbstractContextManager[None]:
        return self._acquire(True)
