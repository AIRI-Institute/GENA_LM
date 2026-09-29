"""Small helpers for importing trusted user and model code."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temporary_sys_path(path: str | Path):
    """Temporarily prepend one resolved directory to ``sys.path``."""

    path_str = str(Path(path).expanduser().resolve())
    added = path_str not in sys.path
    if added:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path_str)
