import numpy as np
from tracker import WhiteTargetTracker


def test_detects_bright_white_blob():
    frame = np.zeros((160, 200, 3), dtype=np.uint8)
    frame[55:95, 80:120] = 255
    tracker = WhiteTargetTracker()
    point = tracker.update(frame, threshold=220, min_area=50)
    assert point is not None
    assert 90 <= point[0] <= 110
    assert 65 <= point[1] <= 85


def test_ignores_small_blob():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:13, 10:13] = 255
    tracker = WhiteTargetTracker()
    assert tracker.update(frame, threshold=220, min_area=80) is None
