"""White Target Assistant desktop UI."""
import threading
import time
import tkinter as tk
from tkinter import ttk
import mss
import numpy as np
import pyautogui
from tracker import WhiteTargetTracker

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class App:
    def __init__(self, root):
        self.root = root
        root.title("白色目标跟踪助手")
        root.geometry("420x510")
        root.resizable(False, False)
        self.tracker = WhiteTargetTracker()
        self.stop_event = threading.Event()
        self.running = False
        self.threshold = tk.IntVar(value=220)
        self.area = tk.IntVar(value=80)
        self.smooth = tk.DoubleVar(value=0.35)
        self.follow = tk.BooleanVar(value=False)
        self.auto = tk.BooleanVar(value=False)
        self.key = tk.StringVar(value="space")
        self.delay = tk.DoubleVar(value=0.8)
        self.status = tk.StringVar(value="状态：已停止")
        self.last_seen = time.monotonic()
        self.sent = False
        self.build()
        root.protocol("WM_DELETE_WINDOW", self.close)

    def build(self):
        ttk.Label(self.root, text="白色目标跟踪助手", font=("Microsoft YaHei UI", 17, "bold")).pack(pady=14)
        box = ttk.LabelFrame(self.root, text="识别设置")
        box.pack(fill="x", padx=16, pady=7)
        self.slider(box, "白色阈值", self.threshold, 150, 255)
        self.slider(box, "最小面积", self.area, 10, 2000)
        box = ttk.LabelFrame(self.root, text="鼠标跟随")
        box.pack(fill="x", padx=16, pady=7)
        ttk.Checkbutton(box, text="启用鼠标跟随", variable=self.follow).pack(anchor="w", padx=10)
        self.slider(box, "平滑度", self.smooth, 0.05, 1.0, 0.05)
        box = ttk.LabelFrame(self.root, text="自动确认（默认关闭）")
        box.pack(fill="x", padx=16, pady=7)
        ttk.Checkbutton(box, text="目标消失后发送确认键", variable=self.auto).pack(anchor="w", padx=10)
        row = ttk.Frame(box); row.pack(fill="x", padx=10, pady=5)
        ttk.Label(row, text="按键").pack(side="left")
        ttk.Entry(row, textvariable=self.key, width=10).pack(side="left", padx=6)
        ttk.Label(row, text="等待秒数").pack(side="left", padx=(12,4))
        ttk.Spinbox(row, from_=0.2, to=10, increment=0.1, textvariable=self.delay, width=6).pack(side="left")
        row = ttk.Frame(self.root); row.pack(fill="x", padx=16, pady=12)
        self.start_btn = ttk.Button(row, text="开始监控", command=self.start); self.start_btn.pack(side="left", expand=True, fill="x", padx=4)
        self.stop_btn = ttk.Button(row, text="停止", command=self.stop, state="disabled"); self.stop_btn.pack(side="left", expand=True, fill="x", padx=4)
        ttk.Label(self.root, textvariable=self.status, font=("Microsoft YaHei UI", 10, "bold")).pack(pady=8)
        ttk.Label(self.root, text="屏幕左上角为 PyAutoGUI 安全停止角。自动确认会向当前前台窗口发送按键。", wraplength=380, foreground="#955").pack(padx=16)

    def slider(self, parent, label, var, low, high, resolution=1):
        row = ttk.Frame(parent); row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text=label, width=12).pack(side="left")
        tk.Scale(row, variable=var, from_=low, to=high, resolution=resolution, orient="horizontal", length=250, highlightthickness=0).pack(side="left", fill="x", expand=True)

    def start(self):
        if self.running: return
        self.running = True; self.stop_event.clear()
        self.last_seen = time.monotonic(); self.sent = False
        self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self.status.set("状态：正在监控")
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False; self.stop_event.set()
        self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
        self.status.set("状态：已停止")

    def set_status(self, value):
        self.root.after(0, lambda: self.status.set(value))

    def loop(self):
        try:
            with mss.mss() as capture:
                screen = capture.monitors[1]
                while not self.stop_event.is_set():
                    rgb = np.asarray(capture.grab(screen))[:, :, :3][:, :, ::-1].copy()
                    target = self.tracker.update(rgb, self.threshold.get(), self.area.get())
                    now = time.monotonic()
                    if target is not None:
                        x, y = target
                        if self.tracker.last_detected:
                            self.last_seen = now
                            self.sent = False
                            self.set_status(f"状态：已识别目标 {x}, {y}")
                        else:
                            self.set_status("状态：目标短暂丢失，正在预测")
                        if self.follow.get():
                            px, py = pyautogui.position()
                            a = float(self.smooth.get())
                            pyautogui.moveTo(int(px+(x-px)*a), int(py+(y-py)*a), _pause=False)
                    else:
                        if self.auto.get() and not self.sent and now-self.last_seen >= float(self.delay.get()):
                            key = self.key.get().strip().lower()
                            if key:
                                pyautogui.press(key); self.sent = True
                                self.set_status("状态：已发送确认键，等待下一轮")
                        else:
                            self.set_status("状态：未发现目标")
                    time.sleep(0.02)
        except pyautogui.FailSafeException:
            self.root.after(0, self.stop)
        except Exception as exc:
            self.set_status(f"错误：{exc}")
            self.root.after(0, self.stop)

    def close(self):
        self.stop()
        self.root.after(100, self.root.destroy)


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()
