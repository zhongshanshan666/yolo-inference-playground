from __future__ import annotations

import numpy as np

from yolo_playground.core.types import PreprocessMeta


def scale_boxes_to_orig(
    boxes_xyxy: np.ndarray,
    meta: PreprocessMeta,
) -> np.ndarray:
    """
    Map xyxy boxes from letterboxed input space back to original image pixels.

    Ultralytics inverse: subtract pad, divide by ratio.
    """
    boxes = boxes_xyxy.copy().astype(np.float32)
    pad_w, pad_h = meta.pad
    r = meta.ratio[0]
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= r

    h, w = meta.orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """cxcywh -> xyxy."""
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy
