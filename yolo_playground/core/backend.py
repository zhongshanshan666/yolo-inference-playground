from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseBackend(ABC):
    """Inference runtime abstraction (ONNX Runtime / TensorRT / RKNN)."""

    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ...

    @abstractmethod
    def get_io_info(self) -> dict[str, Any]:
        """Return input/output tensor names and shapes."""
        ...

    def warmup(self, input_shape: tuple[int, ...]) -> None:
        """Optional warmup run."""
        pass

    def release(self) -> None:
        """Release resources."""
        pass
