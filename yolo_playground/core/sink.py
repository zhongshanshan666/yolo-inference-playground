from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from yolo_playground.core.types import FrameMeta, InferResult


class BaseResultSink(ABC):
    """Output handler abstraction."""

    def open(self, save_dir: Path | None = None) -> None:
        """Called once before inference loop."""
        pass

    @abstractmethod
    def write(self, frame: np.ndarray, result: InferResult) -> None:
        ...

    def close(self) -> None:
        """Called once after inference loop."""
        pass

    def __enter__(self) -> BaseResultSink:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
