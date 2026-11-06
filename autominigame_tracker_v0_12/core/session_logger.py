"""CSV session telemetry and event-frame snapshots."""
from __future__ import annotations
import csv
import os
from pathlib import Path
from datetime import datetime


def default_log_dir() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", Path.home()))
    else:
        base = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "AutoMinigameTracker" / "logs"


class SessionLogger:
    FIELDS = [
        "time_iso", "frame", "round", "state", "valid", "source", "confidence",
        "target_x", "target_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "lost_frames", "csrt_score", "cursor_x", "cursor_y", "cursor_error_px",
        "event", "event_source", "event_score", "fps",
        "capture_ms", "vision_ms", "control_ms", "cursor_step_px",
    ]

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else default_log_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.directory / f"session_{stamp}.csv"
        self._file = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._count = 0

    def write(self, *, frame_index, round_index, result, cursor_point=None,
              cursor_error=None, event=None, fps=None, capture_ms=None,
              vision_ms=None, control_ms=None, cursor_step_px=None):
        bbox = result.bbox or (None, None, None, None)
        point = result.point or (None, None)
        cursor = cursor_point or (None, None)
        event_name = getattr(event, "event", "none") if event is not None else "none"
        event_source = getattr(event, "source", "none") if event is not None else "none"
        event_score = getattr(event, "score", None) if event is not None else None
        self._writer.writerow({
            "time_iso": datetime.now().isoformat(timespec="milliseconds"),
            "frame": frame_index, "round": round_index,
            "state": result.state, "valid": int(bool(result.valid)),
            "source": result.source, "confidence": result.confidence,
            "target_x": point[0], "target_y": point[1],
            "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_w": bbox[2], "bbox_h": bbox[3],
            "lost_frames": result.lost_frames, "csrt_score": result.score,
            "cursor_x": cursor[0], "cursor_y": cursor[1],
            "cursor_error_px": cursor_error,
            "event": event_name, "event_source": event_source, "event_score": event_score,
            "fps": fps,
            "capture_ms": capture_ms, "vision_ms": vision_ms,
            "control_ms": control_ms, "cursor_step_px": cursor_step_px,
        })
        self._count += 1
        if self._count % 10 == 0:
            self._file.flush()

    def save_event_frame(self, frame, event_name: str, frame_index: int):
        try:
            import cv2
            path = self.directory / f"{event_name}_{frame_index:06d}.png"
            cv2.imwrite(str(path), frame)
            return path
        except Exception:
            return None

    def close(self):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()