"""Color-based white target detection and short-term Kalman prediction."""
import cv2
import numpy as np


class WhiteTargetTracker:
    def __init__(self, max_missing=12):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.8
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.missing = 0
        self.max_missing = max_missing
        self.last_detected = False

    def detect(self, frame, threshold=220, min_area=80):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(gray, int(threshold), 255)
        mask &= cv2.inRange(hsv[:, :, 1], 0, 75)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < int(min_area):
                continue
            m = cv2.moments(contour)
            if m["m00"] > 0:
                found.append((area, (m["m10"]/m["m00"], m["m01"]/m["m00"])))
        if not found:
            return None
        if self.initialized and self.missing <= self.max_missing:
            px, py = float(self.kalman.statePost[0,0]), float(self.kalman.statePost[1,0])
            return min(found, key=lambda item: (item[1][0]-px)**2 + (item[1][1]-py)**2)[1]
        return max(found, key=lambda item: item[0])[1]

    def update(self, frame, threshold=220, min_area=80):
        point = self.detect(frame, threshold, min_area)
        self.last_detected = point is not None
        if point is None:
            if self.initialized:
                pred = self.kalman.predict()
                self.missing += 1
                if self.missing <= self.max_missing:
                    return int(pred[0,0]), int(pred[1,0])
                self.initialized = False
            return None
        x, y = point
        if not self.initialized:
            self.kalman.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized = True
        else:
            self.kalman.predict()
            self.kalman.correct(np.array([[x],[y]], np.float32))
        self.missing = 0
        return int(x), int(y)
