"""Compatibility package for the ReTA source tree.

The repository keeps its implementation folders (``data``, ``model``,
``knowledge``, ``policy``, ``train``, ``inference``) at the project root so
they remain easy to inspect. This package extends its module search path to
that root, which makes documented commands such as ``python -m reta.train.warmup``
resolve those folders as ``reta.*`` subpackages.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __path__:
    __path__.append(str(_ROOT))

__all__ = []
