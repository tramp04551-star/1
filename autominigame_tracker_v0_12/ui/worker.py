from __future__ import annotations
from PySide6.QtCore import QThread, Signal
from core.automation_loop import AutomationLoop


class AutomationWorker(QThread):
    status = Signal(str)
    frame_ready = Signal(object, object, object)
    error = Signal(str)

    def __init__(self, window_title: str, fps: float, follow_mouse: bool = True,
                 enable_ocr: bool = False, max_lost_frames: int = 8,
                 acquire_frames: int = 4, enable_events: bool = True, parent=None,
                 cursor_smoothing: float = 0.0, cursor_deadband_px: float = 1.5,
                 cursor_max_speed_px_s: float = 0.0):
        super().__init__(parent)
        self.window_title = window_title
        self.fps = fps
        self.follow_mouse = follow_mouse
        self.enable_ocr = enable_ocr
        self.max_lost_frames = max_lost_frames
        self.acquire_frames = acquire_frames
        self.enable_events = enable_events
        self.cursor_smoothing = cursor_smoothing
        self.cursor_deadband_px = cursor_deadband_px
        self.cursor_max_speed_px_s = cursor_max_speed_px_s
        self.loop = None
        self._stop_requested = False

    def run(self):
        try:
            self.loop = AutomationLoop(
                self.window_title,
                on_status=self.status.emit,
                on_frame=self.frame_ready.emit,
                fps_limit=self.fps,
                follow_mouse=self.follow_mouse,
                enable_ocr=self.enable_ocr,
                max_lost_frames=self.max_lost_frames,
                acquire_frames=self.acquire_frames,
                enable_events=self.enable_events,
                cursor_smoothing=self.cursor_smoothing,
                cursor_deadband_px=self.cursor_deadband_px,
                cursor_max_speed_px_s=self.cursor_max_speed_px_s,
            )
            if self._stop_requested:
                self.loop.capture.close()
                return
            self.loop.run()
        except Exception as exc:
            self.error.emit(str(exc))

    def stop(self):
        self._stop_requested = True
        if self.loop is not None:
            self.loop.stop()