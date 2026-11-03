import cv2
import numpy as np

from tracker import WhiteTargetTracker


def test_forward_backward_optical_flow_estimates_target_translation():
    height, width = 240, 320
    previous = np.full((height, width), 100, dtype=np.uint8)
    cv2.rectangle(previous, (110, 80), (210, 160), 220, -1)
    for y in range(88, 155, 12):
        for x in range(118, 205, 12):
            cv2.circle(previous, (x, y), 3, 245, -1)

    target_mask = np.zeros((height, width), dtype=np.uint8)
    target_mask[80:161, 110:211] = 255
    transform = np.float32([[1, 0, 5], [0, 1, 3]])
    current = cv2.warpAffine(previous, transform, (width, height))

    tracker = WhiteTargetTracker(use_gpu=False)
    tracker.prev_gray = previous
    tracker.last_mask = target_mask
    flow = tracker._estimate_flow(current)

    assert flow is not None
    dx, dy, confidence = flow
    assert abs(dx - 5.0) < 0.75
    assert abs(dy - 3.0) < 0.75
    assert 0.0 < confidence <= 1.0


def test_optical_flow_rejects_shape_mismatch():
    tracker = WhiteTargetTracker(use_gpu=False)
    tracker.prev_gray = np.zeros((100, 100), dtype=np.uint8)
    tracker.last_mask = np.ones((100, 100), dtype=np.uint8)
    assert tracker._estimate_flow(np.zeros((120, 100), dtype=np.uint8)) is None


def test_prediction_confidence_decays_when_visual_evidence_disappears():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:160, 110:210] = 255
    tracker = WhiteTargetTracker(use_gpu=False, max_missing=60)
    # Disable the background-subtractor motion fallback for this deterministic
    # no-observation regression.
    tracker.warmup = -1000
    assert tracker.update(frame, threshold=210, min_area=500) is not None
    initial_confidence = tracker.last_confidence

    blank = np.zeros_like(frame)
    for _ in range(20):
        tracker.update(blank, threshold=210, min_area=500)

    assert tracker.last_confidence < initial_confidence
    assert tracker.missing > 0
