from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from yolo_playground.core.source import BaseFrameSource
from yolo_playground.core.types import FrameMeta


class VideoSource(BaseFrameSource):
    """Video file input."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {self.path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameMeta]]:
        frame_id = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            timestamp_ms = (frame_id / self._fps) * 1000.0
            meta = FrameMeta(
                frame_id=frame_id,
                source_path=str(self.path),
                timestamp_ms=timestamp_ms,
                orig_shape=(h, w),
            )
            yield frame, meta
            frame_id += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

    @property
    def fps(self) -> float:
        return self._fps
