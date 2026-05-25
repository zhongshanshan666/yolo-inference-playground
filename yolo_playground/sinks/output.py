from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from yolo_playground.core.registry import register_sink
from yolo_playground.core.sink import BaseResultSink
from yolo_playground.core.types import Detection, InferResult


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    line_thickness: int = 2,
) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = _class_color(det.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _class_color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 42)
    return tuple(int(c) for c in rng.integers(0, 255, size=3))


@register_sink("visualize")
class VisualizeSink(BaseResultSink):
    """Draw boxes and save annotated images."""

    def __init__(self, show: bool = False) -> None:
        self.show = show
        self.save_dir: Path | None = None

    def open(self, save_dir: Path | None = None) -> None:
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray, result: InferResult) -> None:
        vis = draw_detections(frame, result.detections)
        if self.save_dir and result.frame_meta.source_path:
            stem = Path(result.frame_meta.source_path).stem
            suffix = f"_{result.frame_meta.frame_id}" if result.frame_meta.frame_id else ""
            out_path = self.save_dir / f"{stem}{suffix}_det.jpg"
            cv2.imwrite(str(out_path), vis)
        if self.show:
            cv2.imshow("yolo-playground", vis)
            cv2.waitKey(1)

    def close(self) -> None:
        if self.show:
            cv2.destroyAllWindows()


@register_sink("json")
class JsonExportSink(BaseResultSink):
    """Export detection results as JSON."""

    def __init__(self) -> None:
        self.save_dir: Path | None = None
        self._records: list[dict] = []

    def open(self, save_dir: Path | None = None) -> None:
        self.save_dir = save_dir
        self._records = []
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray, result: InferResult) -> None:
        self._records.append(
            {
                "frame_id": result.frame_meta.frame_id,
                "source": result.frame_meta.source_path,
                "latency_ms": result.latency_ms,
                "detections": [
                    {
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox_xyxy": list(d.bbox),
                    }
                    for d in result.detections
                ],
            }
        )

    def close(self) -> None:
        if self.save_dir and self._records:
            out_path = self.save_dir / "results.json"
            out_path.write_text(json.dumps(self._records, indent=2, ensure_ascii=False), encoding="utf-8")


@register_sink("video_writer")
class VideoWriterSink(BaseResultSink):
    """Write annotated video (for video/stream sources)."""

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self.save_dir: Path | None = None
        self._writer: cv2.VideoWriter | None = None

    def open(self, save_dir: Path | None = None) -> None:
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray, result: InferResult) -> None:
        vis = draw_detections(frame, result.detections)
        if self._writer is None and self.save_dir:
            h, w = vis.shape[:2]
            out_path = self.save_dir / "output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
        if self._writer is not None:
            self._writer.write(vis)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
