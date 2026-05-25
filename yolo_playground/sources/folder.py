from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from yolo_playground.core.source import BaseFrameSource
from yolo_playground.core.types import FrameMeta

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class FolderSource(BaseFrameSource):
    """Iterate over all images in a directory."""

    def __init__(self, path: str | Path, recursive: bool = False) -> None:
        self.root = Path(path)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.root}")
        self.recursive = recursive
        self._paths = self._collect_paths()

    def _collect_paths(self) -> list[Path]:
        if self.recursive:
            files = [p for p in self.root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS]
        else:
            files = [p for p in self.root.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        return sorted(files)

    def __iter__(self) -> Iterator[tuple[np.ndarray, FrameMeta]]:
        for idx, path in enumerate(self._paths):
            image = cv2.imread(str(path))
            if image is None:
                continue
            h, w = image.shape[:2]
            meta = FrameMeta(frame_id=idx, source_path=str(path), orig_shape=(h, w))
            yield image, meta

    def close(self) -> None:
        pass
