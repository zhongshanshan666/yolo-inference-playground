from __future__ import annotations

import numpy as np

from yolo_playground.core.model import BaseYoloModel
from yolo_playground.core.registry import register_model
from yolo_playground.core.types import Detection, PreprocessMeta
from yolo_playground.models.decode import decode_detections, normalize_output_rows
from yolo_playground.preprocess.letterbox import bgr_to_tensor, letterbox


@register_model("yolov5")
class YoloV5Model(BaseYoloModel):
    """
    YOLOv5 — Ultralytics official ONNX export.

    Supports two common Ultralytics export layouts:

    1. Classic YOLOv5 (yolov5n/s/m/l/x.pt):
       output0 shape (1, N, 4+1+nc), e.g. (1, 25200, 85)
       per row: cx, cy, w, h, objectness, class_scores...

    2. YOLOv5u unified head (yolov5nu.pt etc.):
       output0 shape (1, 4+nc, N) or (1, N, 4+nc), e.g. (1, 84, 8400)
       per row: cx, cy, w, h, class_scores...  (no separate objectness)
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

        num_classes = len(self.class_names)
        num_features = rows.shape[1]
        expected_with_obj = 4 + 1 + num_classes
        expected_anchor_free = 4 + num_classes

        if num_features == expected_with_obj:
            return self._decode_with_objectness(rows, meta)
        if num_features == expected_anchor_free:
            return self._decode_anchor_free(rows, meta)

        # Custom nc: infer layout from channel count
        if num_features > 5 and (num_features - 5) == num_classes:
            return self._decode_with_objectness(rows, meta)
        if num_features > 4 and (num_features - 4) == num_classes:
            return self._decode_anchor_free(rows, meta)

        raise ValueError(
            f"Unsupported YOLOv5 output shape {pred.shape}. "
            f"Expected {expected_with_obj} (with objectness) or "
            f"{expected_anchor_free} (anchor-free) features per anchor."
        )

    def _decode_with_objectness(
        self, rows: np.ndarray, meta: PreprocessMeta
    ) -> list[Detection]:
        boxes_xywh = rows[:, :4]
        objectness = rows[:, 4]
        class_scores = rows[:, 5:]

        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        confidences = objectness * class_conf

        return decode_detections(
            boxes_xywh=boxes_xywh,
            confidences=confidences,
            class_ids=class_ids,
            meta=meta,
            class_names=self.class_names,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

    def _decode_anchor_free(self, rows: np.ndarray, meta: PreprocessMeta) -> list[Detection]:
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
