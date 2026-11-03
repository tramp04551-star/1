"""Shape-aware white target tracker.

Detects bright low-saturation game pieces, selects the connected component that
matches the previous target, and chooses a point safely inside the silhouette.
During fade/occlusion, a Kalman prediction carries the last safe interior offset.
"""
import cv2
import numpy as np
from gpu_backend import WhiteMaskProcessor


class WhiteTargetTracker:
    def __init__(self, max_missing=180, use_gpu=True, mask_processor=None):
        self.max_missing = int(max_missing)
        self.use_gpu = bool(use_gpu)
        self.mask_processor = mask_processor or WhiteMaskProcessor(prefer_gpu=use_gpu)
        self._init_filter()
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=8, detectShadows=False
        )
        self.initialized = False
        self.missing = 0
        self.warmup = 0
        self.last_detected = False
        self.last_box = None
        self.last_safe_offset = (0.0, 0.0)
        self.last_safe_point = None
        self.last_confidence = 0.0

    def _init_filter(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.diag([0.08, 0.08, 0.8, 0.8]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10

    @property
    def backend_name(self):
        return self.mask_processor.backend_name

    @property
    def gpu_active(self):
        return self.mask_processor.gpu_active

    def reset(self):
        processor = self.mask_processor
        self.__init__(
            max_missing=self.max_missing,
            use_gpu=self.use_gpu,
            mask_processor=processor,
        )

    @staticmethod
    def _interior_point(binary):
        """Return a point deep inside a component, not its possibly external centroid."""
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, radius, _, location = cv2.minMaxLoc(dist)
        if radius <= 0:
            ys, xs = np.where(binary > 0)
            if not len(xs):
                return None
            return int(np.median(xs)), int(np.median(ys))
        return int(location[0]), int(location[1])

    @staticmethod
    def _components(mask, min_area, max_area):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < min_area or area > max_area:
                continue
            # Ignore thin UI lines and long text strokes but allow rotated pieces.
            ratio = w / max(h, 1)
            if ratio < 0.32 or ratio > 3.1:
                continue
            component = np.zeros((h, w), np.uint8)
            component[labels[y:y+h, x:x+w] == label] = 255
            safe = WhiteTargetTracker._interior_point(component)
            if safe is None:
                continue
            sx, sy = safe
            out.append({
                "area": int(area),
                "center": (float(centers[label][0]), float(centers[label][1])),
                "box": (int(x), int(y), int(w), int(h)),
                "safe": (int(x + sx), int(y + sy)),
                "radius": float(cv2.distanceTransform(component, cv2.DIST_L2, 5)[sy, sx]),
                "mask": component,
            })
        return out

    def _build_masks(self, frame, threshold):
        # Hysteresis levels allow a fading white piece to remain detectable
        # without accepting the saturated brown rock background.
        levels = [
            (int(threshold), 95),
            (max(150, int(threshold) - 28), 82),
            (max(125, int(threshold) - 55), 68),
        ]
        combined = np.zeros(frame.shape[:2], np.uint8)
        for lum, sat in levels:
            mask = self.mask_processor.white_mask(frame, lum)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            mask &= cv2.inRange(hsv[:, :, 1], 0, sat)
            combined = cv2.bitwise_or(combined, mask)
        combined = cv2.morphologyEx(
            combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        return combined

    def update(self, frame, threshold=215, min_area=500, dt=1.0):
        # dt is expressed in nominal 30-fps frames. The capture loop can vary
        # with CPU/GPU load, so a fixed one-frame Kalman step mispredicts motion.
        dt = float(np.clip(dt, 0.20, 4.0))
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt],
             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.diag(
            [0.08 * dt, 0.08 * dt, 0.8 * dt, 0.8 * dt]
        ).astype(np.float32)
        fg = self.bg.apply(frame, learningRate=0.003)
        self.warmup += 1
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(
            fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        white = self._build_masks(frame, threshold)
        h, w = white.shape[:2]
        if not self.initialized:
            # The reference recording shows a countdown at the top-center before
            # the actual white game piece appears near the middle of the playfield.
            # Restrict first acquisition so a bright countdown digit cannot steal
            # the target lock. Once acquired, tracking uses the full game window.
            x0, x1 = int(w * 0.10), int(w * 0.90)
            y0, y1 = int(h * 0.30), int(h * 0.88)
            acquire_mask = white[y0:y1, x0:x1]
            candidates = self._components(
                acquire_mask, max(1200, int(min_area)), max(50000, h*w//12)
            )
            for candidate in candidates:
                cx, cy = candidate["center"]
                sx, sy = candidate["safe"]
                bx, by, bw, bh = candidate["box"]
                candidate["center"] = (cx + x0, cy + y0)
                candidate["safe"] = (sx + x0, sy + y0)
                candidate["box"] = (bx + x0, by + y0, bw, bh)
        else:
            candidates = self._components(
                white, max(120, int(min_area)), max(50000, h*w//12)
            )

        predicted = None
        if self.initialized:
            state = self.kf.predict()
            predicted = (float(state[0, 0]), float(state[1, 0]))

        # If the silhouette has faded, motion foreground can preserve a weak cue.
        if not candidates and self.initialized and self.warmup > 12:
            moving = self._components(fg, max(180, int(min_area)//2), max(50000, h*w//12))
            if moving:
                candidates = moving

        chosen = None
        if candidates:
            if predicted is None:
                # The piece is usually a compact isolated white object; prefer
                # strong, filled shapes over tiny highlights.
                chosen = max(candidates, key=lambda c: c["area"] * min(c["radius"], 20))
            else:
                gate = max(150.0, min(340.0, 90.0 + self.missing * 2.0))
                near = [
                    c for c in candidates
                    if (c["center"][0]-predicted[0])**2 +
                       (c["center"][1]-predicted[1])**2 < gate**2
                ]
                if near:
                    chosen = min(
                        near,
                        key=lambda c: (
                            (c["center"][0]-predicted[0])**2 +
                            (c["center"][1]-predicted[1])**2
                        ) / max(c["area"], 1)**0.15,
                    )

        if chosen is not None:
            x, y = chosen["center"]
            if not self.initialized:
                self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
                self.initialized = True
            else:
                self.kf.correct(np.array([[x], [y]], np.float32))
            safe_x, safe_y = chosen["safe"]
            self.last_safe_offset = (safe_x - x, safe_y - y)
            self.last_safe_point = (safe_x, safe_y)
            self.last_box = chosen["box"]
            self.last_confidence = min(1.0, chosen["radius"] / 8.0)
            self.missing = 0
            self.last_detected = True
            return safe_x, safe_y

        self.last_detected = False
        if self.initialized and predicted is not None:
            self.missing += 1
            if self.missing <= self.max_missing:
                dx, dy = self.last_safe_offset
                px = int(np.clip(predicted[0] + dx, 0, w - 1))
                py = int(np.clip(predicted[1] + dy, 0, h - 1))
                return px, py
            self.initialized = False
            self.last_box = None
            self.last_safe_point = None
            self.last_confidence = 0.0
        return None
