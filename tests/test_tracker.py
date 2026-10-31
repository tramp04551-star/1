import cv2
import numpy as np
from cuda_backend import WhiteMaskProcessor
from tracker import WhiteTargetTracker


def reference_mask(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    result = cv2.inRange(gray, threshold, 255)
    result &= cv2.inRange(hsv[:, :, 1], 0, 95)
    return result


def test_cpu_white_mask_matches_opencv_reference():
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    frame[10:35, 20:55] = (255, 255, 255)
    frame[40:65, 20:55] = (230, 230, 230)
    frame[70:95, 20:55] = (255, 255, 0)
    processor = WhiteMaskProcessor(prefer_cuda=False)
    np.testing.assert_array_equal(
        processor.white_mask(frame, 215), reference_mask(frame, 215)
    )


def test_cuda_mask_matches_opencv_when_gpu_is_available():
    processor = WhiteMaskProcessor(prefer_cuda=True)
    if not processor.cuda_active:
        return
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[5:30, 5:30] = (255, 255, 255)
    frame[35:60, 5:30] = (230, 230, 230)
    frame[65:90, 5:30] = (255, 255, 0)
    np.testing.assert_array_equal(
        processor.white_mask(frame, 215), reference_mask(frame, 215)
    )


def test_cuda_runtime_failure_falls_back_to_cpu():
    processor = WhiteMaskProcessor(prefer_cuda=False)
    processor.cuda_active = True
    processor.torch = object()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated CUDA failure")

    processor._cuda_white_mask = fail
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = processor.white_mask(frame, 215)
    assert processor.cuda_active is False
    assert "simulated CUDA failure" in processor.failure_reason
    assert np.all(mask == 255)


def test_tracker_acquires_white_target_and_predicts_occlusion():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:160, 110:210] = 255
    tracker = WhiteTargetTracker(max_missing=5, use_cuda=False)
    point = tracker.update(frame, 210, 500)
    assert point is not None
    assert 145 <= point[0] <= 175 and 105 <= point[1] <= 135
    assert tracker.last_detected
    assert tracker.update(np.zeros_like(frame), 210, 500) is not None
    assert not tracker.last_detected


def test_tracker_reset_preserves_backend_instance():
    tracker = WhiteTargetTracker(use_cuda=False)
    backend = tracker.mask_processor
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[80:160, 110:210] = 255
    tracker.update(frame, 210, 500)
    tracker.reset()
    assert tracker.mask_processor is backend
    assert not tracker.initialized
    assert tracker.last_box is None
