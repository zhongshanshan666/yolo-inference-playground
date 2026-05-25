from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # xyxy in original image pixels


@dataclass
class PreprocessMeta:
    orig_shape: tuple[int, int]  # (H, W)
    ratio: tuple[float, float]
    pad: tuple[float, float]  # (pad_w, pad_h)
    input_size: tuple[int, int]  # (H, W)


@dataclass
class FrameMeta:
    frame_id: int
    source_path: str | None = None
    timestamp_ms: float | None = None
    orig_shape: tuple[int, int] = (0, 0)  # (H, W)


@dataclass
class InferResult:
    frame_meta: FrameMeta
    detections: list[Detection]
    latency_ms: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)
