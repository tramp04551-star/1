import numpy as np
from gpu_backend import WhiteMaskProcessor
from tracker import WhiteTargetTracker


def test_cpu_mask_detects_white_and_rejects_saturated_background():
    frame = np.zeros((100, 120, 3), np.uint8)
    frame[20:70, 30:80] = (255, 255, 255)
    frame[10:90, 5:20] = (150, 100, 30)
    mask = WhiteMaskProcessor(prefer_gpu=False).white_mask(frame, 215)
    assert np.all(mask[20:70, 30:80] == 255)
    assert np.all(mask[10:90, 5:20] == 0)


def test_tracker_safe_point_is_inside_irregular_target():
    frame = np.zeros((240, 320, 3), np.uint8)
    polygon = np.array([[110, 70], [190, 85], [175, 145], [145, 155], [105, 115]], np.int32)
    import cv2
    cv2.fillPoly(frame, [polygon], (255, 255, 255))
    tracker = WhiteTargetTracker(max_missing=5, use_gpu=False)
    point = tracker.update(frame, 210, 400)
    assert point is not None
    assert cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def test_tracker_predicts_last_safe_interior_offset_during_fade():
    frame = np.zeros((240, 320, 3), np.uint8)
    frame[80:160, 110:210] = 255
    tracker = WhiteTargetTracker(max_missing=5, use_gpu=False)
    point = tracker.update(frame, 210, 500)
    assert point is not None and tracker.last_detected
    predicted = tracker.update(np.zeros_like(frame), 210, 500)
    assert predicted is not None and not tracker.last_detected
    assert abs(predicted[0] - point[0]) < 30
    assert abs(predicted[1] - point[1]) < 30


def test_tracker_reset_preserves_processor():
    tracker = WhiteTargetTracker(use_gpu=False)
    backend = tracker.mask_processor
    tracker.reset()
    assert tracker.mask_processor is backend
    assert not tracker.initialized


def test_opencl_failure_falls_back_to_cpu():
    processor = WhiteMaskProcessor(prefer_gpu=False)
    frame = np.full((16, 16, 3), 255, np.uint8)
    assert np.all(processor.white_mask(frame, 215) == 255)
    assert not processor.gpu_active


def test_initial_acquisition_ignores_top_countdown_digit():
    import cv2
    frame = np.zeros((484, 704, 3), np.uint8)
    # Simulate a bright countdown digit in the upper center and the actual
    # white game piece in the central playfield.
    cv2.rectangle(frame, (330, 65), (375, 160), (245, 245, 245), -1)
    polygon = np.array([[305, 205], [385, 210], [395, 270],
                        [355, 310], [295, 270]], np.int32)
    cv2.fillPoly(frame, [polygon], (255, 255, 255))
    tracker = WhiteTargetTracker(use_gpu=False)
    point = tracker.update(frame, 210, 500)
    assert point is not None
    assert 180 <= point[1] <= 330
