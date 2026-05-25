from __future__ import annotations

import cv2
import numpy as np

from yolo_playground.core.types import PreprocessMeta


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, PreprocessMeta]:
    """
    Ultralytics-style letterbox resize (numpy + cv2).

    Args:
        image: BGR uint8, shape (H, W, 3)
        new_shape: (target_h, target_w)

    Returns:
        resized image and metadata for coordinate inverse transform
    """
    shape = image.shape[:2]  # (H, W)
    target_h, target_w = new_shape

    r = min(target_h / shape[0], target_w / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    pad_w = (target_w - new_unpad[0]) / 2
    pad_h = (target_h - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    meta = PreprocessMeta(
        orig_shape=shape,
        ratio=(r, r),
        pad=(pad_w, pad_h),
        input_size=(target_h, target_w),
    )
    return image, meta


def bgr_to_tensor(image: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> NCHW float32 [0, 1] (Ultralytics ONNX input)."""
    img = image[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis, ...])
