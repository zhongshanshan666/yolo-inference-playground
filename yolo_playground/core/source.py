from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from yolo_playground.core.types import FrameMeta


class BaseFrameSource(ABC):
    """Input source abstraction — yields (BGR image, metadata) pairs."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameMeta]]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self) -> BaseFrameSource:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
