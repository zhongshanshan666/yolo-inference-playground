from __future__ import annotations

import numpy as np

from yolo_playground.core.model import BaseYoloModel
from yolo_playground.core.registry import register_model
from yolo_playground.core.types import Detection, PreprocessMeta
from yolo_playground.models.decode import decode_detections, normalize_output_rows
from yolo_playground.preprocess.letterbox import bgr_to_tensor, letterbox


class UltralyticsYoloModel(BaseYoloModel):
    """
    Base class for Ultralytics-exported ONNX models (v8 / v11 / v26).

    Output tensor shape: (1, 4 + num_classes, num_anchors)
    First 4 columns per row: cx, cy, w, h (letterbox pixel space)
    """

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, PreprocessMeta]:
        img, meta = letterbox(image, self.input_size)
        tensor = bgr_to_tensor(img)
        return tensor, meta

    def postprocess(
        self, outputs: dict[str, np.ndarray], meta: PreprocessMeta
    ) -> list[Detection]:
        pred = next(iter(outputs.values()))
        rows = normalize_output_rows(pred, num_classes=len(self.class_names))

        boxes_xywh = rows[:, :4]
        class_scores = rows[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        return decode_detections(
            boxes_xywh=boxes_xywh,
            confidences=confidences,
            class_ids=class_ids,
            meta=meta,
            class_names=self.class_names,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )


@register_model("yolov8")
class YoloV8Model(UltralyticsYoloModel):
    """YOLOv8 — Ultralytics official ONNX export."""


@register_model("yolo11")
class Yolo11Model(UltralyticsYoloModel):
    """YOLO11 — same output layout as v8 in Ultralytics export."""


@register_model("yolo26")
class Yolo26Model(UltralyticsYoloModel):
    """
    YOLO26 — inherits v8 decode for now.

    Override postprocess here if v26 export format diverges.
    """
