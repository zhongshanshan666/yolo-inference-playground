from __future__ import annotations

import numpy as np

from yolo_playground.core.types import Detection, PreprocessMeta
from yolo_playground.postprocess.coord_transform import scale_boxes_to_orig, xywh_to_xyxy
from yolo_playground.postprocess.nms import nms_numpy


def normalize_output_rows(pred: np.ndarray, num_classes: int = 80) -> np.ndarray:
    """
    Normalize ONNX output to (N, C) row layout.

    Handles:
        (1, N, C)  classic YOLOv5, e.g. (1, 25200, 85)
        (1, C, N)  YOLOv8-style / YOLOv5u, e.g. (1, 84, 8400)
    """
    if pred.ndim == 3:
        pred = pred[0]

    if pred.ndim != 2:
        raise ValueError(f"Expected 2D or 3D prediction tensor, got shape {pred.shape}")

    rows, cols = pred.shape
    feature_sizes = {4 + num_classes, 5 + num_classes, rows, cols}
    feature_sizes = {size for size in feature_sizes if 5 <= size <= 512}

    if rows in feature_sizes and cols not in feature_sizes:
        pred = pred.T
    elif cols in feature_sizes and rows not in feature_sizes:
        pass
    elif rows < cols:
        pred = pred.T
    return pred


def decode_detections(
    boxes_xywh: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    meta: PreprocessMeta,
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    mask = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms_numpy(boxes_xyxy, confidences, iou_threshold)
    boxes_xyxy = boxes_xyxy[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    boxes_xyxy = scale_boxes_to_orig(boxes_xyxy, meta)

    detections: list[Detection] = []
    for i in range(len(boxes_xyxy)):
        cid = int(class_ids[i])
        name = class_names[cid] if cid < len(class_names) else str(cid)
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        detections.append(
            Detection(
                class_id=cid,
                class_name=name,
                confidence=float(confidences[i]),
                bbox=(x1, y1, x2, y2),
            )
        )
    return detections
