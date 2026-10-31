"""Moving bright/translucent target tracker with optional CUDA mask extraction."""
import cv2
import numpy as np
from cuda_backend import WhiteMaskProcessor


class WhiteTargetTracker:
    def __init__(self, max_missing=180, use_cuda=True, mask_processor=None):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=450, varThreshold=10, detectShadows=False
        )
        self.initialized = False
        self.missing = 0
        self.max_missing = max_missing
        self.use_cuda = use_cuda
        self.mask_processor = mask_processor or WhiteMaskProcessor(prefer_cuda=use_cuda)
        self.last_detected = False
        self.last_box = None
        self.warmup = 0

    @property
    def backend_name(self):
        return self.mask_processor.backend_name

    @property
    def cuda_active(self):
        return self.mask_processor.cuda_active

    def reset(self):
        processor = self.mask_processor
        self.__init__(
            max_missing=self.max_missing,
            use_cuda=self.use_cuda,
            mask_processor=processor,
        )

    @staticmethod
    def _components(mask, lo, hi):
        n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            ratio = w / max(h, 1)
            if lo <= area <= hi and 0.45 <= ratio <= 2.2:
                out.append(
                    (
                        int(area),
                        (float(centers[i][0]), float(centers[i][1])),
                        (int(x), int(y), int(w), int(h)),
                    )
                )
        return out

    def update(self, frame, threshold=215, min_area=700):
        fg = self.bg.apply(frame, learningRate=0.004)
        self.warmup += 1
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white = self.mask_processor.white_mask(frame, threshold)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        candidates = self._components(white, min_area, 50000)
        if not candidates and self.initialized and self.warmup > 25:
            candidates = self._components(fg, max(250, min_area // 2), 50000)

        px = py = None
        if self.initialized:
            prediction = self.kf.predict()
            px, py = float(prediction[0, 0]), float(prediction[1, 0])

        if candidates:
            if self.initialized:
                candidates = [
                    c for c in candidates
                    if (c[1][0] - px) ** 2 + (c[1][1] - py) ** 2 < 260 ** 2
                ]
            if candidates:
                chosen = (
                    min(candidates, key=lambda c: (c[1][0] - px) ** 2 + (c[1][1] - py) ** 2)
                    if self.initialized
                    else max(candidates, key=lambda c: c[0])
                )
                _, (x, y), box = chosen
                if not self.initialized:
                    self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
                    self.initialized = True
                else:
                    self.kf.correct(np.array([[x], [y]], np.float32))
                self.missing = 0
                self.last_detected = True
                self.last_box = box
                return int(x), int(y)

        self.last_detected = False
        if self.initialized:
            self.missing += 1
            if self.missing <= self.max_missing:
                return int(px), int(py)
            self.initialized = False
            self.last_box = None
        return None
