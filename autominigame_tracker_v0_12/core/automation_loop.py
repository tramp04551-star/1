"""Live loop: Win32 capture -> visible detector/CSRT -> cursor follow -> round events."""
from __future__ import annotations
import time
import math
from collections import deque

from .follower import WhiteFigureFollower, FollowerResult
from .session_logger import SessionLogger
from .cursor_controller import CursorFollowerController
from vision.round_events import RoundEventDetector


class AutomationLoop:
    def __init__(self, title_part: str, on_status=None, on_frame=None,
                 fps_limit: float = 30.0, follow_mouse: bool = True,
                 enable_ocr: bool = False, max_lost_frames: int = 8,
                 acquire_frames: int = 4, enable_events: bool = True,
                 log_directory=None, cursor_smoothing: float = 0.0,
                 cursor_deadband_px: float = 1.5,
                 cursor_max_speed_px_s: float = 0.0):
        from platform_io.window_capture import GameWindowCapture
        from platform_io.win32_mouse import Win32Mouse
        from tracking.csrt_tracker import create_csrt
        try:
            create_csrt()
        except Exception as exc:
            raise RuntimeError(
                "当前 OpenCV 未提供 CSRT。请安装 opencv-contrib-python，"
                "并卸载 opencv-python/opencv-python-headless 后重试。"
            ) from exc
        self.capture = GameWindowCapture(title_part)
        self.mouse = Win32Mouse()
        self.follower = WhiteFigureFollower(max_lost_frames=max_lost_frames,
                                            acquire_frames=acquire_frames)
        self.event_detector = RoundEventDetector(enable_ocr=enable_ocr) if enable_events else None
        self.on_status = on_status or (lambda _message: None)
        self.on_frame = on_frame or (lambda _frame, _result, _event: None)
        self.frame_period = 1.0 / max(1.0, float(fps_limit))
        self.follow_mouse = bool(follow_mouse)
        self.enable_events = bool(enable_events)
        self.log_directory = log_directory
        self.cursor_controller = CursorFollowerController(
            smoothing=cursor_smoothing,
            deadband_px=cursor_deadband_px,
            max_speed_px_s=cursor_max_speed_px_s,
        )
        self.running = False
        self.waiting_next_round = False
        self.next_round_ready_at = 0.0
        self.round_index = 1
        self.frame_index = 0
        self.cursor_history = deque(maxlen=90)
        self.session_logger = None
        self.fps_ema = 0.0
        self.last_tick_started = None
        self.last_control_time = None
        self._was_foreground = True
        self._last_capture_warning = 0.0

    def stop(self):
        self.running = False
        self.cursor_controller.reset()

    def _attach_cursor_telemetry(self, result, rect):
        try:
            sx, sy = self.mouse.get_position()
            rx, ry = sx - rect[0], sy - rect[1]
            if 0 <= rx < rect[2] and 0 <= ry < rect[3]:
                cursor = (int(rx), int(ry))
                self.cursor_history.append(cursor)
                result.cursor_point = cursor
                result.cursor_trail = tuple(self.cursor_history)
                if result.point is not None:
                    result.cursor_error = math.hypot(result.point[0] - rx, result.point[1] - ry)
        except Exception:
            pass

    def _follow_target(self, result, rect, dt_seconds):
        if not self.follow_mouse or not result.valid or result.point is None:
            self.cursor_controller.reset()
            return None
        try:
            current = self.mouse.get_position()
            target = (rect[0] + result.point[0], rect[1] + result.point[1])
            command = self.cursor_controller.update(target, current, dt_seconds)
            if command is not None:
                self.mouse.move_to(*command.point)
            return command
        except Exception as exc:
            self.cursor_controller.reset()
            self.on_status(f"鼠标控制异常：{exc}")
            return None

    def run(self):
        self.running = True
        self.follower.reset()
        self.cursor_controller.reset()
        self.waiting_next_round = False
        self.next_round_ready_at = 0.0
        self.round_index = 1
        self.frame_index = 0
        self.cursor_history.clear()
        self.session_logger = SessionLogger(self.log_directory)
        self.on_status(f"已连接游戏窗口：{self.capture.window_title}；等待白色图形出现。日志：{self.session_logger.csv_path}")
        if self.event_detector is not None and not self.event_detector.available:
            self.on_status("轮次提示模板未加载；请检查 assets/round_templates 文件")
        elif self.event_detector is not None:
            self.on_status("成功/失败模板识别已启用；Tesseract OCR 为可选兜底")
        try:
            while self.running:
                started = time.perf_counter()
                if not self.capture.is_foreground():
                    if self._was_foreground:
                        self.follower.reset()
                        self.cursor_controller.reset()
                        self.cursor_history.clear()
                        self.last_tick_started = None
                        self.last_control_time = None
                        self.on_status("游戏窗口失去焦点或已最小化；已清除旧轨迹，暂停跟随")
                    self._was_foreground = False
                    time.sleep(0.08)
                    continue
                if not self._was_foreground:
                    self.follower.reset()
                    self.cursor_controller.reset()
                    if self.event_detector is not None:
                        self.event_detector.reset()
                    self._was_foreground = True
                    self.on_status("游戏窗口重新获得焦点；重新搜索目标")

                capture_started = time.perf_counter()
                try:
                    frame, rect = self.capture.grab()
                except RuntimeError as exc:
                    self.follower.reset()
                    self.cursor_controller.reset()
                    self.last_control_time = None
                    now = time.monotonic()
                    if now - self._last_capture_warning >= 1.0:
                        self.on_status(f"窗口捕获暂停：{exc}；窗口恢复后自动重连")
                        self._last_capture_warning = now
                    time.sleep(0.20)
                    continue
                capture_ms = (time.perf_counter() - capture_started) * 1000.0

                vision_started = time.perf_counter()
                event = self.event_detector.detect(frame) if self.event_detector else None
                if self.waiting_next_round and time.monotonic() < self.next_round_ready_at:
                    result = FollowerResult("waiting_next_round", False, None, None,
                                            "none", "unavailable", 0)
                else:
                    result = self.follower.update(frame)
                vision_ms = (time.perf_counter() - vision_started) * 1000.0

                log_round_index = self.round_index
                if event is not None and event.event == "success":
                    completed_round = self.round_index
                    log_round_index = completed_round
                    self.round_index += 1
                    self.waiting_next_round = True
                    self.next_round_ready_at = time.monotonic() + 1.0
                    self.follower.reset()
                    self.cursor_controller.reset()
                    self.cursor_history.clear()
                    self.on_status(f"检测到成功提示（{event.source} {event.score or 0:.3f}）；第 {completed_round} 轮完成，等待下一轮")
                    if self.session_logger:
                        self.session_logger.save_event_frame(frame, "success", self.frame_index)
                elif event is not None and event.event == "failure":
                    self.waiting_next_round = False
                    self.follower.reset()
                    self.cursor_controller.reset()
                    self.on_status(f"检测到失败提示（{event.source} {event.score or 0:.3f}）；已停止跟随")
                    if self.session_logger:
                        self.session_logger.save_event_frame(frame, "failure", self.frame_index)
                    self.running = False

                if self.last_control_time is None:
                    control_dt = self.frame_period
                else:
                    control_dt = max(0.001, min(0.25, started - self.last_control_time))
                self.last_control_time = started
                control_started = time.perf_counter()
                command = None
                if self.running and not self.waiting_next_round:
                    command = self._follow_target(result, rect, control_dt)
                else:
                    self.cursor_controller.reset()
                control_ms = (time.perf_counter() - control_started) * 1000.0
                self._attach_cursor_telemetry(result, rect)

                if self.last_tick_started is not None:
                    frame_interval = max(1e-6, started - self.last_tick_started)
                    inst_fps = 1.0 / frame_interval
                    self.fps_ema = inst_fps if self.fps_ema <= 0 else (0.15 * inst_fps + 0.85 * self.fps_ema)
                self.last_tick_started = started
                move_step = command.step_distance if command is not None else 0.0
                if self.session_logger:
                    self.session_logger.write(
                        frame_index=self.frame_index, round_index=log_round_index,
                        result=result, cursor_point=result.cursor_point,
                        cursor_error=result.cursor_error, event=event, fps=self.fps_ema,
                        capture_ms=capture_ms, vision_ms=vision_ms,
                        control_ms=control_ms, cursor_step_px=move_step,
                    )
                if event is None or event.event == "none":
                    if not self.waiting_next_round and result.valid and result.point is not None:
                        score_text = "" if result.score is None else f" | CSRT={result.score:.3f}"
                        error_text = "" if result.cursor_error is None else f" | 鼠标误差={result.cursor_error:.1f}px"
                        self.on_status(
                            f"第 {self.round_index} 轮 | {result.state} | {result.source} | "
                            f"目标={result.point} | 丢失={result.lost_frames}{score_text}{error_text} | "
                            f"捕获/识别={capture_ms:.1f}/{vision_ms:.1f}ms"
                        )
                    elif not self.waiting_next_round:
                        self.on_status(
                            f"{result.state} | 暂无可信位置，鼠标保持不动 | 丢失={result.lost_frames}"
                        )
                self.frame_index += 1
                self.on_frame(frame, result, event)
                delay = self.frame_period - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
        finally:
            self.cursor_controller.reset()
            if self.session_logger:
                log_path = self.session_logger.csv_path
                self.session_logger.close()
                self.on_status(f"已停止；会话日志已保存：{log_path}")
            self.capture.close()