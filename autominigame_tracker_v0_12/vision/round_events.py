"""Template-first round result detection with optional OCR fallback."""
from __future__ import annotations
from dataclasses import dataclass
import time
import re
import cv2

from vision.template_events import TemplateEventDetector


@dataclass
class RoundEvent:
    event: str  # none | success | failure
    text: str = ""
    available: bool = False
    score: float | None = None
    source: str = "none"


class RoundEventDetector:
    def __init__(self, interval_seconds: float = 0.10, confirm_count: int = 2,
                 enable_ocr: bool = False, template_dir=None):
        self.interval_seconds = max(0.12, float(interval_seconds))
        self.confirm_count = max(1, int(confirm_count))
        self.last_run = 0.0
        self.pending_event = "none"
        self.pending_count = 0
        self.last_emitted = "none"
        self.clear_count = 0
        self.clear_confirm_count = 3
        self.templates = TemplateEventDetector(template_dir=template_dir)
        self._ocr = None
        self.ocr_available = False
        if enable_ocr:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._ocr = pytesseract
                self.ocr_available = True
            except Exception:
                self._ocr = None
                self.ocr_available = False

    @property
    def available(self):
        return self.templates.available or self.ocr_available

    def reset(self):
        self.last_run = 0.0
        self.pending_event = "none"
        self.pending_count = 0
        self.last_emitted = "none"
        self.clear_count = 0

    @staticmethod
    def _prepare(roi, scale=3):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _read_text(self, frame):
        if self._ocr is None:
            return ""
        h, w = frame.shape[:2]
        success_roi = frame[int(h*0.30):int(h*0.68), int(w*0.18):int(w*0.82)]
        failure_roi = frame[int(h*0.30):int(h*0.72), int(w*0.18):int(w*0.82)]
        success_text = self._ocr.image_to_string(
            self._prepare(success_roi), lang="eng",
            config="--psm 8 -c tessedit_char_whitelist=SUCCESS!"
        )
        try:
            failure_text = self._ocr.image_to_string(
                self._prepare(failure_roi), lang="chi_sim+eng",
                config="--psm 7 -c tessedit_char_whitelist=成功失败失敗FAILURE"
            )
        except Exception:
            failure_text = ""
        return (success_text or "") + "\n" + (failure_text or "")

    @staticmethod
    def _event_from_text(text: str) -> str:
        compact = re.sub(r"\s+", "", text).upper()
        if any(token in compact for token in ("SUCCESS", "UCCESS", "SUCC", "成功")):
            return "success"
        if any(token in compact for token in ("失败", "失敗", "FAIL")):
            return "failure"
        return "none"

    def detect(self, frame) -> RoundEvent:
        now = time.monotonic()
        if now - self.last_run < self.interval_seconds:
            return RoundEvent("none", "", self.available)
        self.last_run = now
        match, scores = self.templates.detect(frame)
        event = match.event if match is not None else "none"
        source = "template" if match is not None else "none"
        score = match.score if match is not None else None
        text = ""

        if event == "none" and self._ocr is not None:
            try:
                text = self._read_text(frame)
                event = self._event_from_text(text)
                if event != "none":
                    source = "ocr"
            except Exception:
                event = "none"

        if event == "none":
            # Do not clear the emitted-event latch on a single weak frame. A
            # glowing/animated banner can fluctuate around the match threshold;
            # require several clear samples before arming the same event again.
            self.clear_count += 1
            if self.clear_count >= self.clear_confirm_count:
                self.pending_event = "none"
                self.pending_count = 0
                self.last_emitted = "none"
            return RoundEvent("none", text.strip(), self.available,
                              max(scores.values()) if scores else None, source)

        self.clear_count = 0
        if event == self.pending_event:
            self.pending_count += 1
        else:
            self.pending_event = event
            self.pending_count = 1
        # Strong template hits are distinctive enough to accept immediately;
        # weaker matches and OCR still require consecutive confirmation frames.
        required_count = self.confirm_count
        if source == "template" and score is not None:
            threshold = self.templates.thresholds.get(event, 1.0)
            if score >= threshold + 0.02:
                required_count = 1
        if self.pending_count >= required_count and event != self.last_emitted:
            self.last_emitted = event
            return RoundEvent(event, text.strip(), self.available, score, source)
        return RoundEvent("none", text.strip(), self.available, score, source)