from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


def setup_logger(name: str = "yolo_playground", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def timer() -> Iterator[list[float]]:
    """Context manager that stores elapsed ms in returned list."""
    elapsed: list[float] = []
    t0 = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.append((time.perf_counter() - t0) * 1000.0)
