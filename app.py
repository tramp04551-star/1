"""MapleStory mini-game assistant with reference-inspired workflow UI."""
import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import deque
import ctypes
from ctypes import wintypes

import cv2
import mss
import numpy as np
import pyautogui

from tracker import WhiteTargetTracker

def enumerate_game_windows():
    """Return visible Windows top-level windows with usable client rectangles."""
    if __import__("sys").platform != "win32":
        return []
    user32 = ctypes.windll.user32
    found = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        text = title.value.strip()
        if not text or "冒险岛小游戏助手" in text:
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width >= 320 and height >= 240:
            found.append((text, int(rect.left), int(rect.top), int(width), int(height)))
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    found.sort(key=lambda item: (("maple" not in item[0].lower()), -item[3]*item[4]))
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([4, 90, 115]), np.array([32, 255, 255]))
    h, w = mask.shape
    # Dialog buttons are expected in the lower central region; do not click
    # arbitrary orange scenery or UI elsewhere on the desktop.
    x0, y0 = int(w * .30), int(h * .48)
    roi = mask[y0:, x0:int(w * .78)]
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 35 <= cw <= 300 and 14 <= ch <= 100 and area >= 160:
            candidates.append((area, x0 + x + cw // 2, y0 + y + ch // 2))
    if not candidates:
        return None
    _, x, y = max(candidates)
    return int(x), int(y)


class Assistant:
    def __init__(self, root):
        self.root = root
        root.title("冒险岛小游戏助手")
        root.geometry("760x560")
        root.minsize(720, 520)
        self.tracker = WhiteTargetTracker()
        self.stop_event = threading.Event()
        self.running = False
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
        self.last_backend = None
        self.window_choices = []
        self.selected_window = tk.StringVar(value="自动：主屏幕")
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
        ttk.Label(status, text="GPU 后端").pack(anchor="w", pady=(8, 0))
        ttk.Label(status, textvariable=self.backend, wraplength=220).pack(anchor="w")
        ttk.Label(status, text="完成轮数").pack(anchor="w", pady=(8, 0))
        ttk.Label(status, textvariable=self.rounds, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        log = ttk.LabelFrame(right, text="操作提示", padding=12)
        log.pack(fill="both", expand=True)
        ttk.Label(log, text="• 鼠标移至左上角可紧急停止\n• 透明期按最后轨迹预测\n• GPU 不可用时自动回退 CPU\n• 仅在检测到完成状态后确认",
                  wraplength=220, justify="left").pack(anchor="w")
        footer = ttk.Frame(self.root, padding=(18, 8))
        footer.pack(fill="x")
        ttk.Label(footer, text="OpenCL · NVIDIA / AMD / Intel · CPU fallback",
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
        labels = ["自动：主屏幕"] + [
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
        selected = self.selected_window.get()
        if selected.startswith("自动：") or not self.window_choices:
            mon = capture.monitors[1]
            return {"left": mon["left"], "top": mon["top"],
                    "width": mon["width"], "height": mon["height"]}
        for label, item in zip(
            [f"{v[0]}  ({v[3]}×{v[4]})" for v in self.window_choices],
            self.window_choices
        ):
            if selected == label:
                _, left, top, width, height = item
                return {"left": left, "top": top, "width": width, "height": height}
        mon = capture.monitors[1]
        return {"left": mon["left"], "top": mon["top"],
                "width": mon["width"], "height": mon["height"]}

    def _ui(self, callback, *args):
        self.root.after(0, lambda: callback(*args))

    def start(self):
        if self.running:
            return
        self._snapshot = (self.threshold.get(), self.area.get(), self.smooth.get(),
                          self.follow.get(), self.auto_confirm.get())
        self.running = True
        self.stop_event.clear()
        self.tracker.reset()
        self.state = "monitor"
        self.success_hits = self.button_hits = 0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.set("监控中：等待白色目标出现")
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.stop_event.set()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status.set("已停止")
        self.target_state.set("未锁定")

    def loop(self):
        try:
            with mss.mss() as capture:
                screen = self._capture_region(capture)
                while not self.stop_event.is_set():
                    # Refresh the selected window rectangle occasionally so a
                    # moved/resized game window does not leave stale coordinates.
                    if int(time.monotonic() * 2) % 10 == 0:
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
                        point = self.tracker.update(frame, threshold, min_area)
                        if self.last_backend != self.tracker.backend_name:
                            self._ui(self.backend.set, self.tracker.backend_name)
                            self.last_backend = self.tracker.backend_name
                        if point is not None:
                            x, y = point
                            self._ui(self.target_state.set,
                                     "图形可见 · 内部点" if self.tracker.last_detected
                                     else f"透明期预测 · {self.tracker.missing} 帧")
                            if follow:
                                px, py = pyautogui.position()
                                factor = float(np.clip(smoothing, .15, 1.0))
                                target_x = int(screen["left"] + x)
                                target_y = int(screen["top"] + y)
                                pyautogui.moveTo(int(px + (target_x-px)*factor),
                                                 int(py + (target_y-py)*factor), _pause=False)
                        else:
                            self._ui(self.target_state.set, "等待目标 / 下一轮")
                        if auto_confirm and success_banner(frame):
                            self.success_hits = min(self.success_hits + 1, 8)
                        else:
                            self.success_hits = max(0, self.success_hits - 1)
                        if auto_confirm and self.success_hits >= 3:
                            self.state = "dialog"
                            self.button_hits = 0
                            self._ui(self.status.set, "检测到成功标志：查找确认按钮")
                    elif self.state == "dialog":
                        button = confirm_button(frame)
                        self.button_hits = self.button_hits + 1 if button else 0
                        if self.button_hits >= 3 and button:
                            # Click only after repeated, spatially consistent detection.
                            pyautogui.click(screen["left"] + button[0],
                                            screen["top"] + button[1])
                            self.state = "after_confirm"
                            self.round_count += 1
                            self._ui(self.rounds.set, self.round_count)
                            self._ui(self.status.set, "已确认：等待游戏界面恢复")
                    elif self.state == "after_confirm":
                        if confirm_button(frame) is None and not success_banner(frame):
                            self.tracker.reset()
                            self.state = "monitor"
                            self.success_hits = self.button_hits = 0
                            self._ui(self.status.set, "已恢复监控：等待下一轮")
                    time.sleep(.025)
        except pyautogui.FailSafeException:
            self._ui(self.stop)
        except Exception as exc:
            self._ui(self.status.set, f"运行错误：{exc}")
            self._ui(self.stop)

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
        self.stop()
        self.root.after(100, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = Assistant(root)
    root.mainloop()
