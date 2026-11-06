"""Visible detection + CSRT fading-phase tracking + visible reacquisition."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import time

from vision.white_shape import WhiteShapeDetector, ShapeDetection
from vision.rotation_geometry import RotationGeometryTracker
from tracking.csrt_tracker import CSRTWhiteShapeTracker, TrackResult


@dataclass
class FollowerResult:
    state: str
    valid: bool
    point: Optional[tuple[int, int]]
    bbox: Optional[tuple[int, int, int, int]]
    source: str
    confidence: str
    lost_frames: int
    score: Optional[float] = None
    cursor_point: Optional[tuple[int, int]] = None
    cursor_trail: tuple[tuple[int, int], ...] = ()
    cursor_error: Optional[float] = None


class WhiteFigureFollower:
    def __init__(self, max_lost_frames: int = 5, acquire_frames: int = 4,
                 tracker_factory=None, tracker_name: str = "csrt"):
        self.detector = WhiteShapeDetector()
        self.max_lost_frames = int(max_lost_frames)
        self.tracker_factory = tracker_factory
        self.tracker_name = str(tracker_name).lower()
        self.tracker = CSRTWhiteShapeTracker(
            max_lost_frames=max_lost_frames,
            tracker_factory=tracker_factory,
            tracker_name=tracker_name,
        )
        self.acquire_frames = max(1, int(acquire_frames))
        self.pending_detection: Optional[ShapeDetection] = None
        self.pending_count = 0
        self.state = "search"
        self.last_detection: Optional[ShapeDetection] = None
        self.last_point = None
        # Normalized bbox area learned from the first valid target. It survives
        # round resets and rejects small countdown/start glyphs after LOST.
        self.expected_bbox_area_ratio: Optional[float] = None
        self.rotation_geometry = RotationGeometryTracker()

    def reset(self):
        self.detector = WhiteShapeDetector()
        self.tracker.reset()
        self.pending_detection = None
        self.pending_count = 0
        self.state = "search"
        self.last_detection = None
        self.last_point = None
        self.rotation_geometry.reset()

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    def _choose_detection(self, detections, frame_shape=None):
        if not detections:
            return None
        # A stale box is only a useful association hint during the configured
        # short reacquisition window. After the tracker declares LOST, allow a
        # fresh central spawn to initialize a new track instead of anchoring to
        # an obsolete position from the previous round.
        reference = self.tracker.bbox
        if (not self.tracker.initialized and
                self.tracker.lost_frames > self.tracker.max_lost_frames):
            reference = None
        if reference is None:
            # After a complete LOST state there is no trustworthy position to
            # associate against. Keep searching the playfield, but retain the
            # normalized target size learned on the first round. This prevents
            # countdown digits and the large START glyph from becoming a new
            # target merely because they persist for several frames.
            if self.expected_bbox_area_ratio is None or frame_shape is None:
                return detections[0]
            frame_area = max(1.0, float(frame_shape[0] * frame_shape[1]))
            ranked = []
            for det in detections:
                candidate_ratio = (det.bbox[2] * det.bbox[3]) / frame_area
                relative_size = candidate_ratio / max(1e-9, self.expected_bbox_area_ratio)
                if not 0.65 <= relative_size <= 1.55:
                    continue
                size_penalty = abs(math.log(max(1e-9, relative_size)))
                ranked.append((det.score - 0.30 * size_penalty, det))
            if not ranked:
                return None
            ranked.sort(key=lambda item: item[0], reverse=True)
            if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.06:
                return None
            return ranked[0][1]
        ox, oy = self._center(reference)
        old_area = max(1.0, float(reference[2] * reference[3]))
        ranked = []
        for det in detections:
            nx, ny = self._center(det.bbox)
            distance = math.hypot(nx-ox, ny-oy)
            area_ratio = (det.bbox[2] * det.bbox[3]) / old_area
            # While CSRT is still healthy, only let similarly-sized bright
            # shapes correct it. This rejects countdown/start glyphs that can
            # appear near the fading figure. After LOST, loosen the size gate
            # so a newly spawned target can be acquired again.
            min_area_ratio = 0.65 if self.expected_bbox_area_ratio is not None else 0.20
            if not min_area_ratio <= area_ratio <= 4.0:
                continue
            reacquire_slack = min(90.0, self.tracker.lost_frames * 12.0)
            if distance > max(65.0, 1.25 * math.hypot(reference[2], reference[3]) + reacquire_slack):
                continue
            score = (1.8 * self._iou(reference, det.bbox)
                     - distance / max(35.0, math.hypot(reference[2], reference[3]))
                     + 0.18 * det.score)
            ranked.append((score, det))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.06:
            return None
        return ranked[0][1]

    def _same_candidate(self, a: ShapeDetection, b: ShapeDetection) -> bool:
        if self._iou(a.bbox, b.bbox) >= 0.25:
            return True
        ax, ay = self._center(a.bbox)
        bx, by = self._center(b.bbox)
        return math.hypot(ax-bx, ay-by) <= max(18.0, 0.35 * math.hypot(a.bbox[2], a.bbox[3]))

    def _initialize(self, frame, detection, source, time_s=None):
        # A fully expired appearance track may belong to a new spawn. Start a
        # fresh geometric orbit rather than carrying the previous pivot over.
        if (source == "visible_reacquire" and
                self.tracker.lost_frames > self.tracker.max_lost_frames):
            self.rotation_geometry.reset()
        track = self.tracker.initialize(frame, detection)
        if not track.valid:
            self.state = "search"
            return FollowerResult(self.state, False, None, track.bbox, "init_failed", "unavailable", 0)
        self.last_detection = detection
        self.last_point = detection.safe_point
        if self.expected_bbox_area_ratio is None:
            frame_area = max(1.0, float(frame.shape[0] * frame.shape[1]))
            self.expected_bbox_area_ratio = (detection.bbox[2] * detection.bbox[3]) / frame_area
        self.pending_detection = None
        self.pending_count = 0
        self.rotation_geometry.update(frame,detection,time_s)
        self.state = "track_visible"
        return FollowerResult(self.state, True, detection.safe_point, detection.bbox,
                              source, "measured", 0, track.score)

    def update(self, frame, time_s=None) -> FollowerResult:
        if time_s is None:
            time_s=time.monotonic()
        else:
            time_s=float(time_s)
        # Do not start or correct a track from low-confidence background highlights.
        # Strict near-white candidates are used; CSRT handles the faded interval.
        tracking = self.tracker.initialized or (
            self.tracker.bbox is not None
            and self.tracker.lost_frames <= self.tracker.max_lost_frames
        )
        detections = self.detector.detect_all(
            frame, allow_tolerant=tracking, tracking=tracking
        )
        detection = self._choose_detection(detections, frame.shape)
        geometry_result=None
        if self.tracker.initialized and detection is not None:
            geometry_result=self.rotation_geometry.update(frame,detection,time_s)

        if detection is not None:
            if not self.tracker.initialized:
                if self.pending_detection is not None and self._same_candidate(self.pending_detection, detection):
                    self.pending_count += 1
                else:
                    self.pending_detection = detection
                    self.pending_count = 1
                if self.pending_count < self.acquire_frames:
                    self.state = "acquire_candidate"
                    return FollowerResult(self.state, False, None, detection.bbox,
                                          "candidate", "pending", 0)
                source = "visible_reacquire" if self.last_detection is not None else "visible_init"
                return self._initialize(frame, detection, source, time_s)

            # Keep CSRT's internal model updated while the figure remains visible.
            track = self.tracker.update(frame)
            must_reinit = (
                not track.valid
                or track.bbox is None
                or self._iou(track.bbox, detection.bbox) < 0.40
            )
            if must_reinit:
                # A single bright patch can be a background highlight or success
                # lettering. Confirm a displaced candidate over several frames
                # before allowing it to replace a working target track.
                if self.pending_detection is not None and self._same_candidate(self.pending_detection, detection):
                    self.pending_count += 1
                else:
                    self.pending_detection = detection
                    self.pending_count = 1
                if self.pending_count < self.acquire_frames:
                    self.state = "reacquire_candidate"
                    return FollowerResult(self.state, False, None, detection.bbox,
                                          "candidate", "pending", track.lost_frames,
                                          track.score)
                return self._initialize(frame, detection, "visible_reacquire", time_s)

            self.pending_detection = None
            self.pending_count = 0
            # Segmentation is the authoritative measurement while the figure is visible.
            self.tracker.bbox = detection.bbox
            x, y, w, h = detection.bbox
            sx, sy = detection.safe_point
            self.tracker.anchor_rel = (
                min(1.0, max(0.0, (sx-x) / max(1, w))),
                min(1.0, max(0.0, (sy-y) / max(1, h))),
            )
            self.tracker.lost_frames = 0
            self.last_detection = detection
            self.last_point = detection.safe_point
            self.state = "track_visible"
            return FollowerResult(self.state, True, detection.safe_point, detection.bbox,
                                  "visible_detection", "measured", 0, track.score)

        self.pending_detection = None
        self.pending_count = 0
        if self.tracker.initialized:
            track: TrackResult = self.tracker.update(frame)
            if track.valid:
                # CSRT already supplies a fresh point; compare it with a very
                # recent geometric result if one exists, then synchronize the
                # angle and advance the flow reference without dense flow.
                recent=self.rotation_geometry.recent_result(time_s,max_age_s=.10)
                if (recent is not None and recent.valid and recent.point is not None
                        and recent.confidence>=0.65):
                    distance=math.hypot(float(track.point[0])-recent.point[0],
                                        float(track.point[1])-recent.point[1])
                    gate=max(18.0,0.22*math.hypot(track.bbox[2],track.bbox[3]))
                    if distance>gate:
                        self.state="tracking_conflict"
                        return FollowerResult(self.state,False,None,track.bbox,
                                              "csrt_geometry_disagreement",
                                              "unavailable",track.lost_frames,
                                              recent.confidence)
                self.rotation_geometry.sync_from_point(track.point,time_s)
                self.rotation_geometry.advance_frame(frame)
                self.last_point = track.point
                self.state = "track_faded"
                return FollowerResult(self.state, True, track.point, track.bbox,
                                      self.tracker.tracker_name, "appearance_tracker", track.lost_frames,
                                      track.score)
            # Only pay for dense residual flow when the appearance tracker has
            # actually failed. It can provide a safe point only with fresh,
            # coherent geometry evidence; otherwise the cursor must stop.
            geometry_result=self.rotation_geometry.update(frame,None,time_s)
            geo_valid=(geometry_result is not None and geometry_result.valid
                       and geometry_result.point is not None
                       and geometry_result.confidence>=self.rotation_geometry.min_fade_confidence)
            if geo_valid:
                point=(int(round(geometry_result.point[0])),
                       int(round(geometry_result.point[1])))
                h,w=frame.shape[:2]
                if 0<=point[0]<w and 0<=point[1]<h:
                    self.last_point=point
                    self.state="track_faded_geometry"
                    return FollowerResult(self.state,True,point,None,
                        "rotation_geometry","geometric_residual_flow",track.lost_frames,
                        geometry_result.confidence)
            self.state = "lost" if track.status == "lost" else "tracking_miss"
            return FollowerResult(self.state, False, None, track.bbox,
                                  track.status, "unavailable", track.lost_frames,
                                  track.score)

        # CSRT has its own short loss budget. Keep the independent geometry
        # tracker alive after CSRT resets, while it still has fresh visual flow.
        fade_tracker=getattr(self.rotation_geometry,"fade",None)
        if fade_tracker is not None and not fade_tracker.lost:
            geometry_result=self.rotation_geometry.update(frame,None,time_s)
            geo_valid=(geometry_result is not None and geometry_result.valid
                       and geometry_result.point is not None
                       and geometry_result.confidence>=self.rotation_geometry.min_fade_confidence)
            if geo_valid:
                point=(int(round(geometry_result.point[0])),int(round(geometry_result.point[1])))
                h,w=frame.shape[:2]
                if 0<=point[0]<w and 0<=point[1]<h:
                    self.last_point=point; self.state="track_faded_geometry"
                    return FollowerResult(self.state,True,point,None,
                        "rotation_geometry","geometric_residual_flow",
                        self.tracker.lost_frames,geometry_result.confidence)
            self.state="lost" if fade_tracker.lost else "tracking_miss"
            return FollowerResult(self.state,False,None,None,
                "rotation_geometry_unavailable","unavailable",self.tracker.lost_frames,
                geometry_result.confidence if geometry_result is not None else None)
        self.state = "search"
        return FollowerResult(self.state, False, None, self.tracker.bbox,
                              "none", "unavailable", self.tracker.lost_frames)