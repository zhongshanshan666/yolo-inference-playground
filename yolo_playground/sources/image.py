from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from yolo_playground.core.source import BaseFrameSource
from yolo_playground.core.types import FrameMeta


class ImageSource(BaseFrameSource):
    """Single image input."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Image not found: {self.path}")

    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameMeta]]:
        image = cv2.imread(str(self.path))
        if image is None:
            raise ValueError(f"Failed to read image: {self.path}")
        h, w = image.shape[:2]
        meta = FrameMeta(frame_id=0, source_path=str(self.path), orig_shape=(h, w))
        yield image, meta

    def close(self) -> None:
        pass
