"""CSRT appearance tracker for the fading figure.

CSRT updates its appearance/correlation model on each new frame. This is not a
long-history trajectory predictor. If an update fails, the tracker returns no
usable point for that frame; it never moves the cursor to a stale/predicted point.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import cv2


@dataclass
class TrackResult:
    valid: bool
    bbox: Optional[tuple[int, int, int, int]]
    point: Optional[tuple[int, int]]
    status: str
    lost_frames: int
    score: Optional[float] = None


def create_csrt():
    params = cv2.TrackerCSRT_Params() if hasattr(cv2, "TrackerCSRT_Params") else None
    if params is not None:
        params.use_hog = True
        params.use_color_names = True
        params.padding = 2.0
        params.template_size = 200
    factories = []
    if hasattr(cv2, "TrackerCSRT_create"):
        factories.append(cv2.TrackerCSRT_create)
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        factories.append(legacy.TrackerCSRT_create)
    for factory in factories:
        try:
            return factory(params) if params is not None else factory()
        except TypeError:
            try:
                return factory()
            except Exception:
                continue
    raise RuntimeError(
        "当前 OpenCV 不含 CSRT。请卸载 opencv-python，安装 opencv-contrib-python。"
    )


def create_mil():
    factory = getattr(cv2, "TrackerMIL_create", None)
    if factory is None:
        legacy = getattr(cv2, "legacy", None)
        factory = getattr(legacy, "TrackerMIL_create", None) if legacy else None
    if factory is None:
        raise RuntimeError("当前 OpenCV 不提供 MIL 跟踪器")
    return factory()


def create_named_tracker(name: str):
    name = str(name).lower()
    if name == "csrt":
        return create_csrt()
    if name == "mil":
        return create_mil()
    raise ValueError(f"不支持的跟踪器：{name}")


class CSRTWhiteShapeTracker:
    def __init__(self, max_lost_frames: int = 5, max_jump_px: float = 100.0,
                 tracker_factory=None, tracker_name: str = "csrt"):
        self.max_lost_frames = int(max_lost_frames)
        self.max_jump_px = float(max_jump_px)
        self.tracker_factory = tracker_factory
        self.tracker_name = str(tracker_name).lower()
        self.tracker = None
        self.bbox = None
        self.anchor_rel = (0.5, 0.5)
        self.initial_bbox = None
        self.lost_frames = 0
        self.initialized = False
        self.last_score = None

    def reset(self, keep_last_box: bool = False):
        last = self.bbox if keep_last_box else None
        lost = self.lost_frames if keep_last_box else 0
        self.tracker = None
        self.bbox = last
        self.anchor_rel = (0.5, 0.5)
        self.initial_bbox = None
        self.lost_frames = lost
        self.initialized = False
        self.last_score = None

    def _set_anchor(self, detection):
        x, y, w, h = self.bbox
        sx, sy = detection.safe_point
        self.anchor_rel = (
            min(1.0, max(0.0, (sx - x) / max(1, w))),
            min(1.0, max(0.0, (sy - y) / max(1, h))),
        )

    def initialize(self, frame, detection):
        factory = self.tracker_factory or create_csrt
        self.tracker = factory()
        self.bbox = tuple(map(int, detection.bbox))
        self.initial_bbox = self.bbox
        self._set_anchor(detection)
        ok = self.tracker.init(frame, self.bbox)
        self.initialized = ok is None or bool(ok)
        self.lost_frames = 0
        self.last_score = self._score()
        return self.result(self.initialized, "initialized")

    def _score(self):
        if self.tracker is None:
            return None
        getter = getattr(self.tracker, "getTrackingScore", None)
        if not callable(getter):
            return None
        try:
            value = float(getter())
            return value if math.isfinite(value) else None
        except Exception:
            return None

    def result(self, valid: bool, status: str, score=None) -> TrackResult:
        if self.bbox is None:
            return TrackResult(False, None, None, status, self.lost_frames, score)
        if not valid:
            return TrackResult(False, self.bbox, None, status, self.lost_frames, score)
        x, y, w, h = self.bbox
        ax, ay = self.anchor_rel
        point = (int(round(x + ax * w)), int(round(y + ay * h)))
        return TrackResult(True, self.bbox, point, status, self.lost_frames, score)

    def update(self, frame) -> TrackResult:
        if not self.initialized or self.tracker is None:
            return self.result(False, "not_initialized")
        try:
            ok, raw = self.tracker.update(frame)
        except Exception:
            ok, raw = False, None
        score = self._score()
        self.last_score = score
        if ok and raw is not None:
            x, y, w, h = [int(round(v)) for v in raw]
            fh, fw = frame.shape[:2]
            if w <= 4 or h <= 4 or x < 0 or y < 0 or x + w > fw or y + h > fh:
                ok = False
            else:
                old = self.bbox
                old_c = (old[0] + old[2] / 2.0, old[1] + old[3] / 2.0)
                new_c = (x + w / 2.0, y + h / 2.0)
                jump = math.hypot(new_c[0] - old_c[0], new_c[1] - old_c[1])
                init_area = max(1.0, float(self.initial_bbox[2] * self.initial_bbox[3]))
                area_ratio = (w * h) / init_area
                if jump > self.max_jump_px or not 0.30 <= area_ratio <= 3.2:
                    ok = False
                else:
                    self.bbox = (x, y, w, h)
                    self.lost_frames = 0
                    return self.result(True, self.tracker_name, score)
        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.reset(keep_last_box=True)
            return self.result(False, "lost", score)
        # No stale coordinate is returned as a valid target point.
        return self.result(False, "miss", score)