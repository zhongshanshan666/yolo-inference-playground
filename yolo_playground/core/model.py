from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from yolo_playground.core.backend import BaseBackend
from yolo_playground.core.types import Detection, PreprocessMeta


class BaseYoloModel(ABC):
    """YOLO model abstraction — owns pre/post-processing logic."""

    def __init__(self, backend: BaseBackend, cfg: dict[str, Any]) -> None:
        self.backend = backend
        self.cfg = cfg
        self.input_size: tuple[int, int] = tuple(cfg.get("input_size", [640, 640]))  # (H, W)
        self.conf_threshold: float = cfg.get("conf_threshold", 0.25)
        self.iou_threshold: float = cfg.get("iou_threshold", 0.45)
        self.class_names: list[str] = self._resolve_class_names(cfg)

    @staticmethod
    def _resolve_class_names(cfg: dict[str, Any]) -> list[str]:
        names = cfg.get("class_names")
        if names is None:
            return _COCO80_NAMES
        if isinstance(names, list):
            return names
        raise ValueError(f"Unsupported class_names format: {type(names)}")

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, PreprocessMeta]:
        ...

    @abstractmethod
    def postprocess(
        self, outputs: dict[str, np.ndarray], meta: PreprocessMeta
    ) -> list[Detection]:
        ...

    def infer(self, image: np.ndarray) -> tuple[list[Detection], PreprocessMeta]:
        tensor, meta = self.preprocess(image)
        input_name = self.backend.get_io_info()["input_names"][0]
        outputs = self.backend.run({input_name: tensor})
        detections = self.postprocess(outputs, meta)
        return detections, meta

    @classmethod
    def from_config(cls, cfg: dict[str, Any], backend: BaseBackend) -> BaseYoloModel:
        from yolo_playground.core.registry import MODELS

        model_type = cfg["type"]
        if model_type not in MODELS:
            raise KeyError(f"Unknown model type '{model_type}'. Available: {list(MODELS)}")
        return MODELS[model_type](backend, cfg)


# COCO 80 class names (Ultralytics default)
_COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
