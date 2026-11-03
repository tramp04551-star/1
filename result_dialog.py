"""Reference-template based result-dialog detection.

Input frames are RGB uint8 arrays in the same coordinate space used for mouse
injection. Detection requires both the complete result-dialog template and the
confirm-button template to agree; orange pixels alone are never actionable.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


REFERENCE_SIZE = (759, 563)  # width, height
DIALOG_BUTTON_OFFSET = (233, 118)  # relative to the dialog template's top-left


@dataclass(frozen=True)
class ResultDialogMatch:
    dialog_rect: tuple[int, int, int, int]
    confirm_point: tuple[int, int]
    dialog_score: float
    button_score: float


@lru_cache(maxsize=1)
def _load_templates() -> tuple[np.ndarray, np.ndarray]:
    asset_dir = Path(__file__).resolve().parent / "assets"
    dialog = cv2.imread(str(asset_dir / "result_dialog_template.png"), cv2.IMREAD_GRAYSCALE)
    button = cv2.imread(str(asset_dir / "confirm_button_template.png"), cv2.IMREAD_GRAYSCALE)
    if dialog is None or button is None:
        raise FileNotFoundError(
            "Result-dialog templates are missing. Expected assets/"
            "result_dialog_template.png and assets/confirm_button_template.png"
        )
    if dialog.shape != (151, 287) or button.shape != (34, 55):
        raise ValueError(
            f"Unexpected result-dialog template sizes: {dialog.shape}, {button.shape}"
        )
    return dialog, button


def _resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 0.002:
        return template
    width = max(8, int(round(template.shape[1] * scale)))
    height = max(8, int(round(template.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(template, (width, height), interpolation=interpolation)


def find_result_dialog(
    frame_rgb: np.ndarray,
    *,
    min_dialog_score: float = 0.90,
    min_button_score: float = 0.92,
) -> ResultDialogMatch | None:
    """Find the reference result dialog and return its confirm-button center.

    The reference client uses a 759x563 capture. Proportional resizing is
    supported, but non-uniform scaling is rejected because it distorts template
    geometry and would make the returned click coordinate unsafe.
    """
    if (
        not isinstance(frame_rgb, np.ndarray)
        or frame_rgb.ndim != 3
        or frame_rgb.shape[2] < 3
        or frame_rgb.dtype != np.uint8
    ):
        return None

    height, width = frame_rgb.shape[:2]
    if width < 320 or height < 240:
        return None

    sx = width / REFERENCE_SIZE[0]
    sy = height / REFERENCE_SIZE[1]
    scale = (sx + sy) / 2.0
    if abs(sx - sy) / max(scale, 1e-6) > 0.025:
        return None

    dialog_template, button_template = _load_templates()
    dialog_template = _resize_template(dialog_template, scale)
    button_template = _resize_template(button_template, scale)

    gray = cv2.cvtColor(frame_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)

    # The result dialog is centered. Restricting the search to the central
    # portion prevents unrelated orange controls elsewhere in the game scene
    # from becoming click candidates.
    x0, x1 = int(width * 0.12), int(width * 0.90)
    y0, y1 = int(height * 0.16), int(height * 0.90)
    search = gray[y0:y1, x0:x1]
    th, tw = dialog_template.shape[:2]
    if search.shape[0] < th or search.shape[1] < tw:
        return None

    scores = cv2.matchTemplate(search, dialog_template, cv2.TM_CCOEFF_NORMED)
    _, dialog_score, _, local_top_left = cv2.minMaxLoc(scores)
    if dialog_score < min_dialog_score:
        return None

    dialog_x = x0 + int(local_top_left[0])
    dialog_y = y0 + int(local_top_left[1])

    # Verify the button independently at its known location inside the dialog.
    button_x = dialog_x + int(round(DIALOG_BUTTON_OFFSET[0] * scale))
    button_y = dialog_y + int(round(DIALOG_BUTTON_OFFSET[1] * scale))
    bh, bw = button_template.shape[:2]
    # Resizing the whole client image can shift a scaled sub-template by a
    # pixel because of independent rounding. Search only a tight neighbourhood
    # around the measured button offset, then use the matched button rectangle.
    margin = max(3, int(round(5 * scale)))
    search_x = max(0, button_x - margin)
    search_y = max(0, button_y - margin)
    button_search = gray[
        search_y:button_y + bh + margin,
        search_x:button_x + bw + margin,
    ]
    if (button_search.shape[0] < bh or button_search.shape[1] < bw):
        return None

    button_scores = cv2.matchTemplate(
        button_search, button_template, cv2.TM_CCOEFF_NORMED
    )
    _, button_score, _, button_local = cv2.minMaxLoc(button_scores)
    if button_score < min_button_score:
        return None

    matched_button_x = search_x + int(button_local[0])
    matched_button_y = search_y + int(button_local[1])
    if (abs(matched_button_x - button_x) > margin
            or abs(matched_button_y - button_y) > margin):
        return None

    button_cx = matched_button_x + (bw - 1) / 2.0
    button_cy = matched_button_y + (bh - 1) / 2.0
    return ResultDialogMatch(
        dialog_rect=(dialog_x, dialog_y, tw, th),
        confirm_point=(int(round(button_cx)), int(round(button_cy))),
        dialog_score=float(dialog_score),
        button_score=float(button_score),
    )


__all__ = ["ResultDialogMatch", "find_result_dialog"]
