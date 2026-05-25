import numpy as np

from yolo_playground.postprocess.nms import nms_numpy


def test_nms_empty():
    assert nms_numpy(np.zeros((0, 4)), np.array([]), 0.5) == []


def test_nms_single_box():
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    scores = np.array([0.9])
    keep = nms_numpy(boxes, scores, 0.5)
    assert keep == [0]


def test_nms_overlapping():
    boxes = np.array(
        [[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 250, 250]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.85])
    keep = nms_numpy(boxes, scores, 0.5)
    assert 0 in keep
    assert 2 in keep
    assert len(keep) == 2
