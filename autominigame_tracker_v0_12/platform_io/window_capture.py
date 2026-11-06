"""Win32 client-area enumeration and desktop capture via MSS.

The HWND is refreshed if the game process recreates its top-level window.
Client geometry is queried for every frame so resize/move/DPI changes are picked
up without restarting the application.
"""
from __future__ import annotations
import ctypes
import numpy as np
import cv2


def _set_dpi_awareness():
    if not hasattr(ctypes, "windll"):
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class GameWindowCapture:
    def __init__(self, title_part: str):
        if not title_part.strip():
            raise ValueError("窗口标题关键字不能为空")
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("窗口捕获仅支持 Windows")
        _set_dpi_awareness()
        try:
            import win32gui
        except ImportError as exc:
            raise RuntimeError("窗口捕获需要 pywin32，请执行 pip install pywin32") from exc
        self.win32gui = win32gui
        self.title_part = title_part.casefold()
        self.hwnd, self.window_title = self._find_window()
        self._mss = None
        self.reconnect_count = 0

    def _find_window(self):
        matches = []

        def enum(hwnd, _):
            if self.win32gui.IsWindowVisible(hwnd):
                title = self.win32gui.GetWindowText(hwnd) or ""
                if self.title_part in title.casefold():
                    try:
                        rect = self.win32gui.GetWindowRect(hwnd)
                        area = max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])
                        if area > 10000:
                            matches.append((area, hwnd, title))
                    except Exception:
                        pass
            return True

        self.win32gui.EnumWindows(enum, None)
        if not matches:
            raise RuntimeError(f"找不到可见游戏窗口：{self.title_part}")
        matches.sort(reverse=True, key=lambda item: item[0])
        _, hwnd, title = matches[0]
        return hwnd, title

    def refresh_window(self) -> bool:
        """Rebind to the largest visible matching window; return whether HWND changed."""
        hwnd, title = self._find_window()
        changed = hwnd != self.hwnd
        if changed:
            self.hwnd = hwnd
            self.window_title = title
            self.reconnect_count += 1
        else:
            self.window_title = title
        return changed

    def client_rect_on_screen(self):
        if not self.win32gui.IsWindow(self.hwnd):
            try:
                self.refresh_window()
            except RuntimeError as exc:
                raise RuntimeError("游戏窗口已关闭，等待重新出现") from exc
        if self.win32gui.IsIconic(self.hwnd):
            raise RuntimeError("游戏窗口已最小化")
        left, top, right, bottom = self.win32gui.GetClientRect(self.hwnd)
        sx, sy = self.win32gui.ClientToScreen(self.hwnd, (left, top))
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise RuntimeError("游戏窗口客户区尺寸无效")
        return int(sx), int(sy), int(width), int(height)

    def grab(self):
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError("缺少 mss，请执行 pip install mss") from exc
        if self._mss is None:
            self._mss = mss.mss()
        left, top, width, height = self.client_rect_on_screen()
        try:
            raw = np.asarray(self._mss.grab({
                "left": left, "top": top, "width": width, "height": height
            }))
        except Exception as exc:
            raise RuntimeError(f"MSS 截图失败：{exc}") from exc
        if raw.size == 0:
            raise RuntimeError("窗口捕获返回空画面")
        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        return frame, (left, top, width, height)

    def is_foreground(self):
        if not self.win32gui.IsWindow(self.hwnd):
            try:
                self.refresh_window()
            except RuntimeError:
                return False
            # Require one fresh foreground check after rebinding.
        if self.win32gui.IsIconic(self.hwnd):
            return False
        fg = self.win32gui.GetForegroundWindow()
        try:
            root = self.win32gui.GetAncestor(fg, self.win32gui.GA_ROOT)
        except Exception:
            root = fg
        return fg == self.hwnd or root == self.hwnd

    def close(self):
        if self._mss is not None:
            self._mss.close()
            self._mss = None