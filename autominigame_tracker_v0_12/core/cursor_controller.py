"""Frame-rate independent cursor-follow controller.

The controller only proposes absolute cursor positions. It never clicks or
presses buttons. With smoothing=0 and max_speed=0 it preserves direct tracking.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Sequence


@dataclass(frozen=True)
class CursorCommand:
    point: tuple[int, int]
    distance_before: float
    step_distance: float
    reason: str


class CursorFollowerController:
    def __init__(self, smoothing: float = 0.0, deadband_px: float = 1.5,
                 max_speed_px_s: float = 0.0):
        self.configure(smoothing, deadband_px, max_speed_px_s)
        self.last_command: Optional[CursorCommand] = None

    def configure(self, smoothing: float = 0.0, deadband_px: float = 1.5,
                  max_speed_px_s: float = 0.0):
        self.smoothing = min(0.95, max(0.0, float(smoothing)))
        self.deadband_px = max(0.0, float(deadband_px))
        self.max_speed_px_s = max(0.0, float(max_speed_px_s))

    def reset(self):
        self.last_command = None

    @staticmethod
    def _point(value: Sequence[float]) -> tuple[float, float]:
        if value is None or len(value) != 2:
            raise ValueError("坐标必须为 (x, y)")
        x, y = float(value[0]), float(value[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("坐标必须为有限数值")
        return x, y

    def update(self, target: Sequence[float], current: Sequence[float],
               dt_seconds: float) -> Optional[CursorCommand]:
        tx, ty = self._point(target)
        cx, cy = self._point(current)
        dt = max(0.0, min(0.25, float(dt_seconds)))
        dx, dy = tx - cx, ty - cy
        distance = math.hypot(dx, dy)
        if distance <= self.deadband_px:
            self.last_command = None
            return None

        # smoothing is the retained fraction of the current error.
        alpha = 1.0 - self.smoothing
        step = distance * alpha
        if self.max_speed_px_s > 0.0:
            step = min(step, self.max_speed_px_s * dt)
        if step <= 0.25:
            self.last_command = None
            return None

        ratio = min(1.0, step / distance)
        nx = int(round(cx + dx * ratio))
        ny = int(round(cy + dy * ratio))
        if nx == int(round(cx)) and ny == int(round(cy)):
            self.last_command = None
            return None
        actual_step = math.hypot(nx - cx, ny - cy)
        command = CursorCommand((nx, ny), distance, actual_step,
                                "direct" if self.smoothing == 0 and self.max_speed_px_s == 0 else "filtered")
        self.last_command = command
        return command