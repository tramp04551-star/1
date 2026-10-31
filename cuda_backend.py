"""Optional PyTorch CUDA acceleration for RGB white-target mask extraction.

CUDA performs per-pixel RGB luminance/saturation tests. Connected-component
labeling, temporal background subtraction, and Kalman tracking remain OpenCV
CPU operations. A missing/broken CUDA runtime transparently falls back to CPU.
"""
import cv2
import numpy as np


class WhiteMaskProcessor:
    def __init__(self, prefer_cuda: bool = True):
        self.torch = None
        self.device = "cpu"
        self.cuda_active = False
        self.failure_reason = None
        if not prefer_cuda:
            return
        try:
            import torch

            self.torch = torch
            if torch.version.cuda is None:
                self.failure_reason = "安装的是 CPU 版 PyTorch"
                return
            if not torch.cuda.is_available():
                self.failure_reason = "未检测到可用 NVIDIA CUDA 设备或驱动"
                return
            torch.cuda.init()
            self.device = "cuda:0"
            torch.empty((1,), device=self.device)
            torch.cuda.synchronize()
            self.cuda_active = True
        except Exception as exc:
            self.failure_reason = f"{type(exc).__name__}: {exc}"
            self.device = "cpu"
            self.cuda_active = False

    @property
    def backend_name(self) -> str:
        if self.cuda_active:
            try:
                return f"CUDA GPU · {self.torch.cuda.get_device_name(0)}"
            except Exception:
                return "CUDA GPU · device 0"
        if self.failure_reason:
            return f"CPU fallback · {self.failure_reason}"
        return "CPU · CUDA disabled"

    def white_mask(self, frame: np.ndarray, threshold: int = 215) -> np.ndarray:
        """Return an 8-bit 0/255 mask from an RGB uint8 frame."""
        if self.cuda_active:
            try:
                return self._cuda_white_mask(frame, threshold)
            except Exception as exc:
                self.failure_reason = f"CUDA runtime error: {type(exc).__name__}: {exc}"
                self.cuda_active = False
                self.device = "cpu"
        return self._cpu_white_mask(frame, threshold)

    @staticmethod
    def _cpu_white_mask(frame: np.ndarray, threshold: int) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(gray, int(threshold), 255)
        mask &= cv2.inRange(hsv[:, :, 1], 0, 95)
        return mask

    def _cuda_white_mask(self, frame: np.ndarray, threshold: int) -> np.ndarray:
        torch = self.torch
        rgb = np.ascontiguousarray(frame)
        pixels = torch.from_numpy(rgb).to(
            device=self.device, dtype=torch.int32, non_blocking=False
        )
        red, green, blue = pixels.unbind(dim=2)
        maximum = torch.maximum(torch.maximum(red, green), blue)
        minimum = torch.minimum(torch.minimum(red, green), blue)
        luminance = (red * 299 + green * 587 + blue * 114 + 500) // 1000
        saturation_ok = (maximum == 0) | (
            (maximum - minimum) * 510 <= maximum * 191
        )
        mask = (luminance >= int(threshold)) & saturation_ok
        return (mask.to(torch.uint8) * 255).to("cpu").numpy()
