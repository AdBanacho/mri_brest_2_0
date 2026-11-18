"""Filesystem helper: legacy run names are ~150 characters, which overflows Windows MAX_PATH (260).

On Windows every path handed to ``open``/``mkdir``/``rglob`` is converted to the
extended-length form ``\\\\?\\C:\\...``; elsewhere paths are returned unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

_PREFIX = "\\\\?\\"


def fs(path: str | os.PathLike) -> Path:
    if os.name != "nt":
        return Path(path)
    text = str(path)
    if text.startswith(_PREFIX):
        return Path(text)
    absolute = os.path.abspath(text)
    if absolute.startswith("\\\\"):  # UNC share
        return Path(_PREFIX + "UNC\\" + absolute[2:])
    return Path(_PREFIX + absolute)


def display(path: str | os.PathLike) -> str:
    """Strip the extended-length prefix for messages/CSV values."""
    text = str(path)
    if text.startswith(_PREFIX + "UNC\\"):
        return "\\\\" + text[len(_PREFIX) + 4:]
    return text[len(_PREFIX):] if text.startswith(_PREFIX) else text
