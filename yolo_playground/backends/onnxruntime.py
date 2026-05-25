from __future__ import annotations

from typing import Any

import numpy as np

from yolo_playground.core.backend import BaseBackend
from yolo_playground.core.registry import register_backend


@register_backend("onnxruntime")
class OnnxRuntimeBackend(BaseBackend):
    """ONNX Runtime backend for Ultralytics-exported models."""

    def __init__(self) -> None:
        self._session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: str, **kwargs: Any) -> None:
        import onnxruntime as ort

        providers = kwargs.get("providers") or ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        num_threads = kwargs.get("num_threads")
        if num_threads:
            sess_options.intra_op_num_threads = num_threads

        available = ort.get_available_providers()
        selected = [p for p in providers if p in available]
        if not selected:
            selected = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, sess_options=sess_options, providers=selected)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._session is None:
            raise RuntimeError("Backend not loaded. Call load() first.")
        outputs = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, outputs))

    def get_io_info(self) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("Backend not loaded.")
        return {
            "input_names": self._input_names,
            "output_names": self._output_names,
            "inputs": [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in self._session.get_inputs()
            ],
            "outputs": [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in self._session.get_outputs()
            ],
        }

    def warmup(self, input_shape: tuple[int, ...]) -> None:
        if self._session is None or not self._input_names:
            return
        dummy = np.zeros(input_shape, dtype=np.float32)
        self.run({self._input_names[0]: dummy})

    def release(self) -> None:
        self._session = None
