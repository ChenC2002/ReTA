"""
Logging utilities.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


def get_run_dir(root: str = "runs", name: Optional[str] = None) -> str:
    """Create and return a run directory path."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}" if name is None else f"{ts}-{name}"
    path = os.path.join(root, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def setup_logger(log_dir: Optional[str] = None, name: str = "reta", level: str = "INFO"):
    """Setup and return a logger."""
    try:
        from loguru import logger
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=level)
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            logger.add(
                os.path.join(log_dir, f"{name}.log"),
                level=level,
                enqueue=True,
                backtrace=False,
                diagnose=False,
            )
        return logger
    except Exception:
        import logging
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers = []
        logger.propagate = False

        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
            fh.setLevel(getattr(logging, level.upper(), logging.INFO))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger
