"""Reference replay regressions for the active tracker implementation."""
import cv2
import numpy as np

from tracker import WhiteTargetTracker


def test_initial_lock_does_not_switch_to_connected_countdown_overlay():
    frame = np.zeros((563, 759, 3), np.uint8)
    # Countdown overlay touches the top of the target, as in the reference
    # sequence. The acquisition gate must still choose the playfield object.
    cv2.rectangle(frame, (345, 60), (405, 205), (245, 245, 245), -1)
    polygon = np.array(
        [[315, 196], [400, 208], [410, 285], [365, 330], [305, 280]],
        np.int32,
    )
    cv2.fillPoly(frame, [polygon], (255, 255, 255))

    tracker = WhiteTargetTracker(use_gpu=False)
    point = tracker.update(frame, threshold=210, min_area=500)

    assert point is not None
    assert point[1] >= 200
    assert tracker.last_detected
    assert cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0
