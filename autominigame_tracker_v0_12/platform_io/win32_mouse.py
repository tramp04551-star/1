"""Absolute mouse movement through Win32 SendInput (Windows only)."""
from __future__ import annotations
import ctypes
from ctypes import wintypes

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG), ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]
class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]
class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]

class Win32Mouse:
    def __init__(self):
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("Win32 鼠标控制仅支持 Windows")
        self.user32 = ctypes.windll.user32
        self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT

    def _send_mouse(self, dx: int, dy: int, flags: int):
        mi = MOUSEINPUT(dx, dy, 0, flags, 0, 0)
        inp = INPUT(type=INPUT_MOUSE, mi=mi)
        sent = self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if sent != 1:
            raise ctypes.WinError()

    def move_to(self, x: int, y: int):
        SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
        SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
        vx = self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vy = self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vw = max(1, self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
        vh = max(1, self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        ax = round((int(x) - vx) * 65535 / max(1, vw - 1))
        ay = round((int(y) - vy) * 65535 / max(1, vh - 1))
        self._send_mouse(ax, ay,
                         MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)

    def get_position(self):
        point = wintypes.POINT()
        if not self.user32.GetCursorPos(ctypes.byref(point)):
            raise ctypes.WinError()
        return int(point.x), int(point.y)

    def click_left(self):
        self._send_mouse(0, 0, MOUSEEVENTF_LEFTDOWN)
        self._send_mouse(0, 0, MOUSEEVENTF_LEFTUP)