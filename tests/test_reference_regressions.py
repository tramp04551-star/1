import cv2
import numpy as np
from inminigame_assistant import ShapeTracker, EndMarker


def test_shape_tracker_ignores_upper_countdown_and_locks_center_piece():
    frame = np.zeros((503, 740, 3), np.uint8)
    cv2.rectangle(frame, (330, 55), (390, 145), (245, 245, 245), -1)
    polygon = np.array([[340, 190], [420, 220], [390, 330], [300, 290]], np.int32)
    cv2.fillConvexPoly(frame, polygon, (255, 255, 255))
    tracker = ShapeTracker(use_gpu=False)
    point, mode, confidence, detection = tracker.update(frame, 1.0)
    assert point is not None and detection is not None
    assert detection.area >= 6000
    assert 135 <= point[1] <= 420


def test_end_marker_detects_bright_success_banner():
    frame = np.zeros((503, 740, 3), np.uint8)
    cv2.putText(frame, "SUCCESS", (170, 290), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 255), 7, cv2.LINE_AA)
    found, kind, pixels, box = EndMarker.detect(frame)
    assert found and "成功" in kind and pixels > 450 and box is not None
