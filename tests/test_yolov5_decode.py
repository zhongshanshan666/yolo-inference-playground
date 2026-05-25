import numpy as np
import pytest

from yolo_playground.core.types import PreprocessMeta
from yolo_playground.models.decode import decode_detections, normalize_output_rows
from yolo_playground.models.yolov5 import YoloV5Model


class _DummyBackend:
    pass


def _meta() -> PreprocessMeta:
    return PreprocessMeta(orig_shape=(480, 640), ratio=(1.0, 1.0), pad=(0.0, 0.0), input_size=(640, 640))


def test_normalize_classic_v5_layout():
    pred = np.zeros((1, 3, 85), dtype=np.float32)
    rows = normalize_output_rows(pred)
    assert rows.shape == (3, 85)


def test_normalize_transposed_layout():
    pred = np.zeros((1, 85, 3), dtype=np.float32)
    rows = normalize_output_rows(pred)
    assert rows.shape == (3, 85)


def test_yolov5_classic_decode_with_objectness():
    model = YoloV5Model(_DummyBackend(), {"input_size": [640, 640], "conf_threshold": 0.25, "iou_threshold": 0.45})

    # One anchor: person at center, high objectness + class0 score
    row = np.zeros(85, dtype=np.float32)
    row[0:4] = [320, 240, 100, 80]
    row[4] = 0.9
    row[5] = 0.85
    pred = row.reshape(1, 1, 85)

    dets = model.postprocess({"output0": pred}, _meta())
    assert len(dets) == 1
    assert dets[0].class_id == 0
    assert dets[0].class_name == "person"
    assert dets[0].confidence == pytest.approx(0.9 * 0.85, rel=1e-4)


def test_yolov5_anchor_free_decode():
    model = YoloV5Model(_DummyBackend(), {"input_size": [640, 640], "conf_threshold": 0.25, "iou_threshold": 0.45})

    row = np.zeros(84, dtype=np.float32)
    row[0:4] = [320, 240, 100, 80]
    row[4] = 0.92
    pred = row.reshape(1, 1, 84)

    dets = model.postprocess({"output0": pred}, _meta())
    assert len(dets) == 1
    assert dets[0].class_id == 0
    assert dets[0].confidence == pytest.approx(0.92, rel=1e-4)


def test_decode_filters_low_confidence():
    boxes = np.array([[320, 240, 100, 80]], dtype=np.float32)
    confidences = np.array([0.1])
    class_ids = np.array([0])

    dets = decode_detections(
        boxes_xywh=boxes,
        confidences=confidences,
        class_ids=class_ids,
        meta=_meta(),
        class_names=["person"],
        conf_threshold=0.25,
        iou_threshold=0.45,
    )
    assert dets == []
