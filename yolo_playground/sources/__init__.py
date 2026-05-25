from __future__ import annotations

from pathlib import Path

from yolo_playground.sources.folder import FolderSource
from yolo_playground.sources.image import ImageSource
from yolo_playground.sources.stream import StreamSource
from yolo_playground.sources.video import VideoSource


def create_source(source: str | int) -> "BaseFrameSource":
    """
    Factory: auto-detect source type from path or special prefix.

    - Existing file + image ext  -> ImageSource
    - Existing file + video ext  -> VideoSource
    - Existing directory         -> FolderSource
    - "webcam:N"                 -> StreamSource(N)
    - int                        -> StreamSource(int)
    - "rtsp://..." / "http://..." -> StreamSource(url)
    """
    from yolo_playground.core.source import BaseFrameSource

    if isinstance(source, int):
        return StreamSource(source)

    text = str(source)
    if text.startswith("webcam:"):
        return StreamSource(int(text.split(":", 1)[1]))

    path = Path(text)
    if path.is_dir():
        return FolderSource(path)

    if path.is_file():
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        if path.suffix.lower() in video_exts:
            return VideoSource(path)
        return ImageSource(path)

    if text.startswith(("rtsp://", "http://", "https://")):
        return StreamSource(text)

    raise FileNotFoundError(f"Cannot resolve source: {source}")


__all__ = ["ImageSource", "FolderSource", "VideoSource", "StreamSource", "create_source"]
