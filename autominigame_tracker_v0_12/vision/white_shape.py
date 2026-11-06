"""Detect the briefly visible white figure in the minigame playfield.

The detector deliberately has two thresholds:
- strict: isolates the bright white core seen in the reference recording;
- tolerant: catches the larger/off-white silhouette in compressed captures.

It only provides a measured box while pixels are visible. Faded-phase tracking is
handled by CSRT in tracking/csrt_tracker.py; no constant-velocity extrapolation is used.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class ShapeDetection:
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    safe_point: tuple[int, int]
    area: float
    score: float
    threshold: str = "tolerant"


class WhiteShapeDetector:
    def __init__(
        self,
        min_area_ratio: float = 0.0009,
        max_area_ratio: float = 0.075,
        min_aspect: float = 0.55,
        max_aspect: float = 1.55,
        min_value: int = 225,
        max_saturation: int = 65,
        strict_value: int = 245,
        strict_saturation: int = 18,
        center_x_range: tuple[float, float] = (0.25, 0.75),
        center_y_range: tuple[float, float] = (0.30, 0.70),
    ):
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self.min_value = int(min_value)
        self.max_saturation = int(max_saturation)
        self.strict_value = int(strict_value)
        self.strict_saturation = int(strict_saturation)
        self.center_x_range = center_x_range
        self.center_y_range = center_y_range
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _mask(self, frame: np.ndarray, strict: bool) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        if strict:
            mask = ((v >= self.strict_value) & (s <= self.strict_saturation)).astype(np.uint8) * 255
            # Keep strict white components separate; only close tiny codec gaps.
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = ((v >= self.min_value) & (s <= self.max_saturation)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    @staticmethod
    def _safe_point(component: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        x, y, w, h = bbox
        roi = component[y:y+h, x:x+w]
        dist = cv2.distanceTransform((roi > 0).astype(np.uint8), cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        return x + int(max_loc[0]), y + int(max_loc[1])

    def _from_mask(self, frame: np.ndarray, mask: np.ndarray, threshold: str,
                   tracking: bool = False) -> list[ShapeDetection]:
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        cx0, cy0 = w * 0.5, h * 0.5
        diag = max(1.0, float(np.hypot(w, h)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found: list[ShapeDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            ratio_area = area / frame_area
            if not self.min_area_ratio <= ratio_area <= self.max_area_ratio:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = bw / max(1.0, float(bh))
            if not self.min_aspect <= aspect <= self.max_aspect:
                continue
            center_x, center_y = x + bw / 2.0, y + bh / 2.0
            nx, ny = center_x / w, center_y / h
            # Initial acquisition is restricted to the central spawn zone so
            # countdown text and unrelated white UI elements are rejected.
            # Once a track exists, reacquisition must be allowed across the
            # whole playfield because the figure can travel far from spawn.
            x_range = (0.04, 0.96) if tracking else self.center_x_range
            y_range = (0.05, 0.95) if tracking else self.center_y_range
            if not (x_range[0] <= nx <= x_range[1]):
                continue
            if not (y_range[0] <= ny <= y_range[1]):
                continue
            component = np.zeros((h, w), np.uint8)
            cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
            fill_ratio = area / max(1.0, float(bw * bh))
            if fill_ratio < 0.15:
                continue
            hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
            solidity = min(1.0, area / hull_area)
            center_penalty = float(np.hypot(center_x-cx0, center_y-cy0)) / diag
            aspect_penalty = abs(np.log(max(1e-4, aspect)))
            # Compact, near-square, central shapes outrank countdown glyphs and UI text.
            score = (1.8 * fill_ratio + 0.35 * solidity
                     - 0.55 * aspect_penalty - 0.75 * center_penalty)
            px, py = self._safe_point(component, (x, y, bw, bh))
            found.append(ShapeDetection(
                bbox=(int(x), int(y), int(bw), int(bh)),
                contour=contour.copy(), mask=component,
                safe_point=(int(px), int(py)), area=area,
                score=float(score), threshold=threshold,
            ))
        found.sort(key=lambda d: d.score, reverse=True)
        return found

    def detect_all(self, frame: np.ndarray, allow_tolerant: bool = False,
                   tracking: bool = False) -> list[ShapeDetection]:
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return []
        strict = self._from_mask(frame, self._mask(frame, True), "strict", tracking)
        if strict or not allow_tolerant:
            return strict
        # Tolerant threshold is only for correction while a tracker is already
        # active. It is never allowed to start a fresh track from a background
        # highlight in an otherwise empty playfield.
        return self._from_mask(frame, self._mask(frame, False), "tolerant", tracking)

    def detect(self, frame: np.ndarray) -> Optional[ShapeDetection]:
        candidates = self.detect_all(frame, allow_tolerant=False)
        if not candidates:
            return None
        if len(candidates) > 1 and candidates[0].score - candidates[1].score < 0.055:
            return None
        return candidates[0]