from __future__ import annotations

from typing import Any

from yolo_playground.core.backend import BaseBackend


class TensorRTBackend(BaseBackend):
    """Placeholder for future TensorRT integration."""

    def load(self, model_path: str, **kwargs: Any) -> None:
        raise NotImplementedError("TensorRT backend is not implemented yet.")

    def run(self, inputs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_io_info(self) -> dict[str, Any]:
        raise NotImplementedError


class RknnBackend(BaseBackend):
    """Placeholder for future RKNN integration."""

    def load(self, model_path: str, **kwargs: Any) -> None:
        raise NotImplementedError("RKNN backend is not implemented yet.")

    def run(self, inputs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_io_info(self) -> dict[str, Any]:
        raise NotImplementedError
