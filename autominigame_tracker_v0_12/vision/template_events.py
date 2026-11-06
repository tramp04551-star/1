"""Multi-scale visual recognition of round result banners.

The supplied success/failure banner crops are matched against the live playfield.
The matcher is intentionally independent from Tesseract so result detection can
work on a clean Windows install without an OCR runtime.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


@dataclass
class TemplateMatch:
    event: str
    score: float
    location: tuple[int, int]
    size: tuple[int, int]
    scale: float


class TemplateEventDetector:
    def __init__(self, template_dir: Optional[str | Path] = None,
                 success_threshold: float = 0.70,
                 failure_threshold: float = 0.72):
        if template_dir is None:
            template_dir = Path(__file__).resolve().parents[1] / "assets" / "round_templates"
        self.template_dir = Path(template_dir)
        self.thresholds = {"success": float(success_threshold), "failure": float(failure_threshold)}
        self.templates: dict[str, np.ndarray] = {}
        for event in ("success", "failure"):
            path = self.template_dir / f"{event}.jpg"
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None and image.size:
                self.templates[event] = cv2.GaussianBlur(image, (3, 3), 0)

    @property
    def available(self) -> bool:
        return bool(self.templates)

    @staticmethod
    def _playfield_roi(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = gray.shape[:2]
        x0, x1 = int(w * 0.10), int(w * 0.90)
        y0, y1 = int(h * 0.10), int(h * 0.90)
        return gray[y0:y1, x0:x1], (x0, y0)

    def _match_one(self, frame: np.ndarray, event: str, template: np.ndarray) -> Optional[TemplateMatch]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        roi, (ox, oy) = self._playfield_roi(gray)
        rh, rw = roi.shape[:2]
        th, tw = template.shape[:2]
        if rh < th or rw < tw:
            return None

        # Coarse matching at reduced resolution is fast enough to run alongside
        # a 30-60 FPS tracker. Event recognition needs banner-level, not subpixel,
        # precision. Coordinates are mapped back to the original frame.
        proc_scale = min(0.4, 720.0 / max(1.0, float(rw)))
        proc_w = max(1, int(round(rw * proc_scale)))
        proc_h = max(1, int(round(rh * proc_scale)))
        roi_small = cv2.resize(roi, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        template_small = cv2.resize(
            template, None, fx=proc_scale, fy=proc_scale, interpolation=cv2.INTER_AREA
        )
        small_h, small_w = template_small.shape[:2]
        min_scale = max(0.35, (rw * 0.24) / max(1, tw))
        max_scale = min(3.50, (rw * 0.72) / max(1, tw))
        if max_scale < min_scale:
            min_scale = max_scale
        scales = sorted(set(np.linspace(min_scale, max_scale, 7).tolist() + [1.0]))
        best_score, best_loc, best_size, best_scale = -1.0, (0, 0), (tw, th), 1.0
        for scale in scales:
            nw, nh = int(round(small_w * scale)), int(round(small_h * scale))
            if nw < 12 or nh < 8 or nw >= proc_w or nh >= proc_h:
                continue
            resized = cv2.resize(
                template_small, (nw, nh),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            response = cv2.matchTemplate(roi_small, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(response)
            if score > best_score:
                best_score = float(score)
                best_loc = (
                    int(round(loc[0] / proc_scale + ox)),
                    int(round(loc[1] / proc_scale + oy)),
                )
                best_size = (
                    int(round(nw / proc_scale)), int(round(nh / proc_scale))
                )
                best_scale = float(scale)
        if best_score < 0:
            return None
        return TemplateMatch(event, best_score, best_loc, best_size, best_scale)

    def detect(self, frame: np.ndarray) -> tuple[Optional[TemplateMatch], dict[str, float]]:
        if frame is None or not self.templates:
            return None, {}
        matches: dict[str, TemplateMatch] = {}
        for event, template in self.templates.items():
            match = self._match_one(frame, event, template)
            if match is not None:
                matches[event] = match
        scores = {key: value.score for key, value in matches.items()}
        accepted = [m for key, m in matches.items() if m.score >= self.thresholds[key]]
        if not accepted:
            return None, scores
        accepted.sort(key=lambda m: m.score, reverse=True)
        return accepted[0], scores