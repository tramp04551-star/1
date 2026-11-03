"""MapleStory mini-game assistant with reference-inspired workflow UI."""
import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import deque
import ctypes
import sys
from ctypes import wintypes


def enable_dpi_awareness():
    """Keep Win32 window coordinates and MSS physical pixels in one coordinate space."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


enable_dpi_awareness()

import cv2
import mss
import numpy as np
import pyautogui

from tracker import WhiteTargetTracker
from result_dialog import find_result_dialog


GAME_ROI = (306, 119, 759, 563)


def map_game_roi(client_region, expected_client_size=None):
    """Map the fixed reference ROI into physical desktop coordinates."""
    if client_region is None:
        raise ValueError("游戏窗口客户区不可用")
    client_size = (int(client_region["width"]), int(client_region["height"]))
    if expected_client_size is not None and client_size != tuple(expected_client_size):
        raise ValueError(
            f"游戏客户区尺寸已变化：{client_size[0]}×{client_size[1]}，"
            f"预期 {expected_client_size[0]}×{expected_client_size[1]}"
        )
    x, y, width, height = GAME_ROI
    if client_size[0] < x + width or client_size[1] < y + height:
        raise ValueError(
            f"游戏客户区 {client_size[0]}×{client_size[1]} 不足以容纳固定 ROI "
            f"{x},{y},{width},{height}"
        )
    return {
        "left": int(client_region["left"]) + x,
        "top": int(client_region["top"]) + y,
        "width": width,
        "height": height,
    }


def _configure_user32(user32):
    """Set pointer-sized Win32 signatures; ctypes defaults HWND returns to c_int."""
    signatures = {
        "IsWindow": ([wintypes.HWND], wintypes.BOOL),
        "IsIconic": ([wintypes.HWND], wintypes.BOOL),
        "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
        "GetClientRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        "ClientToScreen": ([wintypes.HWND, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
        "GetForegroundWindow": ([], wintypes.HWND),
        "GetAncestor": ([wintypes.HWND, ctypes.c_uint], wintypes.HWND),
        "GetWindowTextLengthW": ([wintypes.HWND], ctypes.c_int),
        "GetWindowTextW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        try:
            function = getattr(user32, name)
            function.argtypes = argtypes
            function.restype = restype
        except (AttributeError, TypeError):
            # Test doubles are ordinary Python methods and do not expose ctypes
            # function metadata; real WinDLL functions always accept the types.
            pass
    return user32


def _handle_value(handle):
    if isinstance(handle, int):
        return handle
    value = ctypes.cast(handle, ctypes.c_void_p).value
    return int(value or 0)


def get_window_client_region(hwnd, user32=None):
    """Return the client-area screen rectangle, excluding title bar and borders."""
    if hwnd is None:
        return None
    user32 = _configure_user32(user32 or ctypes.windll.user32)
    try:
        handle = _handle_value(hwnd)
        if not user32.IsWindow(handle) or user32.IsIconic(handle):
            return None
        client = wintypes.RECT()
        if not user32.GetClientRect(handle, ctypes.byref(client)):
            return None
        origin = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(handle, ctypes.byref(origin)):
            return None
        width = int(client.right - client.left)
        height = int(client.bottom - client.top)
        if width < 1 or height < 1:
            return None
        return {
            "left": int(origin.x),
            "top": int(origin.y),
            "width": width,
            "height": height,
        }
    except Exception:
        return None


def is_window_foreground(target_hwnd, user32=None):
    """Only allow input while the selected game window (or its child) is foreground."""
    if target_hwnd is None:
        return False
    if user32 is None:
        if sys.platform != "win32":
            return False
        user32 = ctypes.windll.user32
    user32 = _configure_user32(user32)
    try:
        target = _handle_value(target_hwnd)
        if not user32.IsWindow(target):
            return False
        foreground = user32.GetForegroundWindow()
        if not foreground:
            return False
        # GA_ROOT = 2: child controls belong to the selected top-level game window.
        root = user32.GetAncestor(foreground, 2) or foreground
        return _handle_value(root) == target
    except Exception:
        return False


def enumerate_game_windows():
    """Return visible Windows top-level windows with usable client rectangles."""
    if sys.platform != "win32":
        return []
    user32 = _configure_user32(ctypes.windll.user32)
    found = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    try:
        user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
        user32.EnumWindows.restype = wintypes.BOOL
    except (AttributeError, TypeError):
        pass

    def callback(hwnd, _):
        handle = _handle_value(hwnd)
        if not user32.IsWindowVisible(handle):
            return True
        length = user32.GetWindowTextLengthW(handle)
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, title, length + 1)
        text = title.value.strip()
        if not text or "冒险岛小游戏助手" in text:
            return True
        region = get_window_client_region(handle, user32)
        if region and region["width"] >= 320 and region["height"] >= 240:
            found.append((
                text, region["left"], region["top"],
                region["width"], region["height"], handle
            ))
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    found.sort(key=lambda item: (
        not ("maple" in item[0].lower() or "冒险岛" in item[0]),
        -item[3] * item[4],
    ))
    return found

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def success_banner(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # The supplied recordings show the completion word in bright yellow/gold.
    mask = cv2.inRange(hsv, np.array([15, 105, 205]), np.array([42, 255, 255]))
    h, w = mask.shape
    roi = mask[int(h * .25):int(h * .78), int(w * .18):int(w * .82)]
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glyphs = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 12]
    return len(glyphs) >= 2 and sum(glyphs) > 260


def confirm_button(frame):
    """Return the center of the confirmed result-dialog button, never a color blob."""
    match = find_result_dialog(frame)
    return match.confirm_point if match is not None else None


class Assistant:
    def __init__(self, root):
        self.root = root
        root.title("冒险岛小游戏助手")
        root.geometry("760x560")
        root.minsize(720, 520)
        self.tracker = WhiteTargetTracker()
        self.stop_event = threading.Event()
        self.running = False
        self.worker = None
        self.threshold = tk.IntVar(value=215)
        self.area = tk.IntVar(value=500)
        self.smooth = tk.DoubleVar(value=.62)
        self.follow = tk.BooleanVar(value=True)
        self.auto_confirm = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="待机：等待开始")
        self.backend = tk.StringVar(value=self.tracker.backend_name)
        self.target_state = tk.StringVar(value="未锁定")
        self.rounds = tk.IntVar(value=0)
        self.round_count = 0
        self.events = deque(maxlen=100)
        self.state = "monitor"
        self.success_hits = 0
        self.button_hits = 0
        self.last_button_point = None
        self.dialog_started_at = 0.0
        self.after_confirm_clear_hits = 0
        self.last_backend = None
        self.window_choices = []
        self.selected_window = tk.StringVar(value="自动：桌面监控（仅监控）")
        self.capture_window_selection = "自动：桌面监控（仅监控）"
        self.capture_window_choices = []
        self.target_hwnd = None
        self.capture_client_size = None
        self.dialog_detect_hits = 0
        self.confirm_blocked_notice = False
        self._closing = False
        self.build()
        self._snapshot = (self.threshold.get(), self.area.get(), self.smooth.get(),
                          self.follow.get(), self.auto_confirm.get())
        self.root.after(100, self._refresh_snapshot)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def build(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("Sub.TLabel", foreground="#65748b")
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))
        header = ttk.Frame(self.root, padding=(18, 14))
        header.pack(fill="x")
        ttk.Label(header, text="冒险岛小游戏助手", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="目标锁定  /  透明期保持  /  完成确认  /  自动等待下一轮",
                  style="Sub.TLabel").pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ttk.Frame(body)
        right.pack(side="right", fill="y")

        detect = ttk.LabelFrame(left, text="目标识别", padding=12)
        detect.pack(fill="x", pady=(0, 10))
        self._slider(detect, "白色阈值", self.threshold, 125, 255)
        self._slider(detect, "最小目标面积", self.area, 120, 2500)
        mouse = ttk.LabelFrame(left, text="鼠标控制", padding=12)
        mouse.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(mouse, text="自动将鼠标保持在目标图形内部",
                        variable=self.follow).pack(anchor="w")
        self._slider(mouse, "跟随平滑度", self.smooth, .15, 1.0, .05)
        capture = ttk.LabelFrame(left, text="游戏窗口", padding=12)
        capture.pack(fill="x", pady=(0, 10))
        capture_row = ttk.Frame(capture)
        capture_row.pack(fill="x")
        self.window_combo = ttk.Combobox(
            capture_row, textvariable=self.selected_window,
            state="readonly", width=38
        )
        self.window_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(capture_row, text="刷新窗口", command=self.refresh_windows).pack(side="left")
        self.refresh_windows()

        flow = ttk.LabelFrame(left, text="自动流程", padding=12)
        flow.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(flow, text="完成后自动查找并点击确认，再等待下一轮",
                        variable=self.auto_confirm).pack(anchor="w")
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(5, 0))
        self.start_btn = ttk.Button(actions, text="开始监控", style="Accent.TButton",
                                    command=self.start)
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.stop_btn = ttk.Button(actions, text="停止", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True)

        status = ttk.LabelFrame(right, text="运行状态", padding=12)
        status.pack(fill="x", pady=(0, 10))
        ttk.Label(status, textvariable=self.status, style="Status.TLabel",
                  wraplength=220).pack(anchor="w", pady=(0, 8))
        ttk.Label(status, text="目标状态").pack(anchor="w")
        ttk.Label(status, textvariable=self.target_state, foreground="#16835d").pack(anchor="w")
        ttk.Label(status, text="掩膜计算后端").pack(anchor="w", pady=(8, 0))
        ttk.Label(status, textvariable=self.backend, wraplength=220).pack(anchor="w")
        ttk.Label(status, text="完成轮数").pack(anchor="w", pady=(8, 0))
        ttk.Label(status, textvariable=self.rounds, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        log = ttk.LabelFrame(right, text="操作提示", padding=12)
        log.pack(fill="both", expand=True)
        ttk.Label(log, text="• 鼠标移至左上角可紧急停止\n• 透明期光流辅助并显示预测置信度\n• 低置信度或失焦时暂停输入\n• 桌面模式仅监控，不发送输入",
                  wraplength=220, justify="left").pack(anchor="w")
        footer = ttk.Frame(self.root, padding=(18, 8))
        footer.pack(fill="x")
        ttk.Label(footer, text="OpenCL 掩膜加速 · CPU 回退",
                  style="Sub.TLabel").pack(side="left")
        ttk.Label(footer, text="本地屏幕处理", style="Sub.TLabel").pack(side="right")

    @staticmethod
    def _slider(parent, label, variable, low, high, step=1):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=15).pack(side="left")
        scale = tk.Scale(row, variable=variable, from_=low, to=high,
                         resolution=step, orient="horizontal",
                         showvalue=True, length=300, highlightthickness=0)
        scale.pack(side="left", fill="x", expand=True)

    def refresh_windows(self):
        self.window_choices = enumerate_game_windows()
        labels = ["自动：桌面监控（仅监控）"] + [
            f"{item[0]}  ({item[3]}×{item[4]})" for item in self.window_choices
        ]
        self.window_combo["values"] = labels
        if self.selected_window.get() not in labels:
            # Prefer a title containing MapleStory when present; otherwise keep
            # the primary-screen fallback and let the user choose the game window.
            maple = next((i for i, item in enumerate(self.window_choices)
                          if "maple" in item[0].lower() or "冒险岛" in item[0]), None)
            self.selected_window.set(labels[maple + 1] if maple is not None else labels[0])

    def _capture_region(self, capture):
        # The selected HWND is immutable for this run; resolve its current client
        # rectangle every two seconds so moving/resizing the game does not stale
        # the capture origin. Desktop mode remains monitor-only and cannot inject.
        if self.target_hwnd is not None:
            region = get_window_client_region(self.target_hwnd)
            if region is None:
                raise RuntimeError("所选游戏窗口已关闭或最小化")
            try:
                return map_game_roi(region, self.capture_client_size)
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
        mon = capture.monitors[1]
        return {"left": mon["left"], "top": mon["top"],
                "width": mon["width"], "height": mon["height"]}

    def _game_window_has_focus(self):
        return is_window_foreground(self.target_hwnd)

    def _ui(self, callback, *args):
        self.root.after(0, lambda: callback(*args))

    def start(self):
        if self.running or (self.worker is not None and self.worker.is_alive()):
            return
        self._snapshot = (self.threshold.get(), self.area.get(), self.smooth.get(),
                          self.follow.get(), self.auto_confirm.get())
        self.capture_window_selection = self.selected_window.get()
        self.capture_window_choices = list(self.window_choices)
        self.target_hwnd = None
        if not self.capture_window_selection.startswith("自动："):
            for item in self.capture_window_choices:
                if self.capture_window_selection == f"{item[0]}  ({item[3]}×{item[4]})":
                    self.target_hwnd = item[5]
                    break
            if self.target_hwnd is None:
                self.status.set("所选窗口已失效：请刷新窗口列表后重新选择")
                return
        self.capture_client_size = None
        if self.target_hwnd is not None:
            client_region = get_window_client_region(self.target_hwnd)
            try:
                initial_roi = map_game_roi(client_region)
            except ValueError as exc:
                self.status.set(f"无法开始：{exc}")
                return
            self.capture_client_size = (
                client_region["width"], client_region["height"]
            )
        self.window_combo.configure(state="disabled")
        self.running = True
        self.stop_event.clear()
        self.tracker.reset()
        self.state = "monitor"
        self.success_hits = self.button_hits = 0
        self.last_button_point = None
        self.dialog_started_at = 0.0
        self.after_confirm_clear_hits = 0
        self.dialog_detect_hits = 0
        self.confirm_blocked_notice = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.set("监控中：等待白色目标出现")
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        self.stop_event.set()
        worker_alive = self.worker is not None and self.worker.is_alive()
        self.start_btn.configure(state="disabled" if worker_alive else "normal")
        self.stop_btn.configure(state="disabled")
        self.status.set("正在停止…" if worker_alive else "已停止")
        self.target_state.set("未锁定")

    def loop(self):
        try:
            with mss.mss() as capture:
                previous_frame_time = time.perf_counter()
                while not self.stop_event.is_set():
                    now = time.perf_counter()
                    dt_frames = max(0.20, min(4.0, (now - previous_frame_time) * 30.0))
                    previous_frame_time = now
                    # Resolve the client origin every frame; a moved game window
                    # must never leave a two-second stale mouse-coordinate mapping.
                    screen = self._capture_region(capture)
                    frame = np.asarray(capture.grab(screen))[:, :, :3][:, :, ::-1].copy()
                    # Tk variables are read on the UI thread only; snapshot values
                    # are refreshed by the UI before starting and via trace callbacks.
                    threshold = self.threshold_value
                    min_area = self.area_value
                    smoothing = self.smooth_value
                    follow = self.follow_value
                    auto_confirm = self.auto_value

                    if self.state == "monitor":
                        point = self.tracker.update(frame, threshold, min_area, dt=dt_frames)
                        if self.last_backend != self.tracker.backend_name:
                            self._ui(self.backend.set, self.tracker.backend_name)
                            self.last_backend = self.tracker.backend_name
                        if point is not None:
                            x, y = point
                            if self.tracker.last_detected:
                                state_label = "图形可见 · 安全内部点"
                            elif self.tracker.flow_confidence >= 0.20:
                                state_label = (
                                    f"透明期光流辅助 · 置信度 {self.tracker.last_confidence:.2f}"
                                )
                            else:
                                state_label = (
                                    f"轨迹预测 · 第 {self.tracker.missing} 帧 · "
                                    f"置信度 {self.tracker.last_confidence:.2f}"
                                )
                            self._ui(self.target_state.set, state_label)
                            if follow:
                                if self.tracker.last_confidence < 0.12:
                                    self._ui(
                                        self.target_state.set,
                                        f"预测置信度过低 ({self.tracker.last_confidence:.2f}) · 暂停输入"
                                    )
                                elif self._game_window_has_focus():
                                    latest_screen = self._capture_region(capture)
                                    if latest_screen != screen:
                                        self._ui(self.target_state.set, "窗口位置变化 · 重新捕获")
                                    else:
                                        px, py = pyautogui.position()
                                        factor = float(np.clip(smoothing, .15, 1.0))
                                        target_x = int(latest_screen["left"] + x)
                                        target_y = int(latest_screen["top"] + y)
                                        pyautogui.moveTo(int(px + (target_x-px)*factor),
                                                         int(py + (target_y-py)*factor), _pause=False)
                                else:
                                    self._ui(
                                        self.target_state.set,
                                        "桌面监控 · 未选择游戏窗口，输入已禁用"
                                        if self.target_hwnd is None
                                        else "游戏窗口未置前 · 暂停输入"
                                    )
                        else:
                            self._ui(self.target_state.set, "等待目标 / 下一轮")
                        result_match = find_result_dialog(frame) if auto_confirm else None
                        if result_match is not None:
                            self.dialog_detect_hits = min(self.dialog_detect_hits + 1, 8)
                        else:
                            self.dialog_detect_hits = 0

                        if auto_confirm and success_banner(frame):
                            self.success_hits = min(self.success_hits + 1, 8)
                        else:
                            self.success_hits = max(0, self.success_hits - 1)

                        # The complete dialog template is a direct completion
                        # signal; the banner is only a fallback for delayed dialogs.
                        completion_seen = (
                            self.dialog_detect_hits >= 3 or self.success_hits >= 3
                        )
                        if auto_confirm and completion_seen:
                            if self.target_hwnd is None:
                                if not self.confirm_blocked_notice:
                                    self._ui(
                                        self.status.set,
                                        "当前为桌面监控模式：请选择具体游戏窗口后才能自动跟随/确认"
                                    )
                                    self.confirm_blocked_notice = True
                                self.success_hits = 0
                                self.dialog_detect_hits = 0
                            else:
                                self.state = "dialog"
                                self.button_hits = 0
                                self.last_button_point = None
                                self.dialog_started_at = time.monotonic()
                                self.dialog_detect_hits = 0
                                self._ui(self.status.set, "检测到完成状态：等待确认按钮")
                    elif self.state == "dialog":
                        button = confirm_button(frame)
                        if button is None:
                            self.button_hits = 0
                            self.last_button_point = None
                        else:
                            if (self.last_button_point is not None
                                    and np.hypot(button[0] - self.last_button_point[0],
                                                 button[1] - self.last_button_point[1]) <= 4):
                                self.button_hits += 1
                            else:
                                self.button_hits = 1
                            self.last_button_point = button

                        if self.button_hits >= 3 and button:
                            # Never send input to a game window that has lost focus.
                            if not self._game_window_has_focus():
                                self.button_hits = 0
                                self.last_button_point = None
                                self._ui(self.status.set, "游戏窗口未置前：确认已暂停")
                            else:
                                # The complete dialog and its confirm button have both
                                # matched for three consecutive, spatially stable frames.
                                latest_screen = self._capture_region(capture)
                                if latest_screen != screen:
                                    self.button_hits = 0
                                    self.last_button_point = None
                                    self._ui(self.status.set, "窗口位置变化：重新识别确认按钮")
                                else:
                                    pyautogui.click(latest_screen["left"] + button[0],
                                                    latest_screen["top"] + button[1])
                                    self.state = "after_confirm"
                                    self.after_confirm_clear_hits = 0
                                    self.last_button_point = None
                                    self._ui(self.status.set, "已点击确认：验证结果对话框关闭")
                        elif time.monotonic() - self.dialog_started_at > 8.0:
                            self._ui(self.stop)
                            self._ui(self.status.set, "未能可靠识别确认按钮，已安全停止")
                    elif self.state == "after_confirm":
                        dialog_still_visible = confirm_button(frame) is not None
                        banner_still_visible = success_banner(frame)
                        if not dialog_still_visible and not banner_still_visible:
                            self.after_confirm_clear_hits += 1
                        else:
                            self.after_confirm_clear_hits = 0

                        # Count a round only after the dialog is absent in several
                        # consecutive frames. A click alone is not success.
                        if self.after_confirm_clear_hits >= 3:
                            self.round_count += 1
                            self._ui(self.rounds.set, self.round_count)
                            self.tracker.reset()
                            self.state = "monitor"
                            self.success_hits = self.button_hits = 0
                            self.last_button_point = None
                            self._ui(self.status.set, "已验证对话框关闭：等待下一轮")
                    time.sleep(.025)
        except pyautogui.FailSafeException:
            self.stop_event.set()
            worker_message = "已触发左上角紧急停止"
        except Exception as exc:
            self.stop_event.set()
            worker_message = f"运行错误：{type(exc).__name__}: {exc}"
        else:
            worker_message = None
        finally:
            self._ui(self._worker_finished, locals().get("worker_message"))

    def _worker_finished(self, worker_message=None):
        self.running = False
        if self._closing:
            return
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.window_combo.configure(state="readonly")
        if worker_message:
            self.status.set(worker_message)
        elif self.stop_event.is_set() and self.status.get() == "正在停止…":
            self.status.set("已停止")

    @property
    def threshold_value(self):
        return int(self._snapshot[0])

    @property
    def area_value(self):
        return int(self._snapshot[1])

    @property
    def smooth_value(self):
        return float(self._snapshot[2])

    @property
    def follow_value(self):
        return bool(self._snapshot[3])

    @property
    def auto_value(self):
        return bool(self._snapshot[4])

    def _refresh_snapshot(self):
        self._snapshot = (self.threshold.get(), self.area.get(), self.smooth.get(),
                          self.follow.get(), self.auto_confirm.get())
        self.root.after(100, self._refresh_snapshot)

    def close(self):
        if self._closing:
            return
        self._closing = True
        self.stop()
        self._wait_for_worker_exit()

    def _wait_for_worker_exit(self):
        if self.worker is not None and self.worker.is_alive():
            self.root.after(50, self._wait_for_worker_exit)
            return
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Assistant(root)
    root.mainloop()
