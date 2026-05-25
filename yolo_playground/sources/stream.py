from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from yolo_playground.core.source import BaseFrameSource
from yolo_playground.core.types import FrameMeta


class StreamSource(BaseFrameSource):
    """
    Camera / RTSP stream input (reserved for future use).

    Examples:
        StreamSource(0)                  # webcam index 0
        StreamSource("rtsp://...")       # RTSP URL
    """

    def __init__(self, source: int | str, fps: float | None = None) -> None:
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open stream: {source}")
        self._fps = fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0

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
                source_path=str(self.source),
                timestamp_ms=timestamp_ms,
                orig_shape=(h, w),
            )
            yield frame, meta
            frame_id += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
