from pathlib import Path

import cv2
import numpy as np

from result_dialog import find_result_dialog


ASSETS = Path(__file__).resolve().parents[1] / "assets"


def make_reference_dialog_frame():
    dialog = cv2.imread(
        str(ASSETS / "result_dialog_template.png"), cv2.IMREAD_GRAYSCALE
    )
    button = cv2.imread(
        str(ASSETS / "confirm_button_template.png"), cv2.IMREAD_GRAYSCALE
    )
    assert dialog is not None and button is not None
    gray = np.full((563, 759), 64, dtype=np.uint8)
    gray[220:220 + dialog.shape[0], 242:242 + dialog.shape[1]] = dialog
    # The measured button extends one row below the dialog crop boundary.
    gray[338:338 + button.shape[0], 475:475 + button.shape[1]] = button
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def test_result_dialog_and_confirm_button_match_reference_geometry():
    match = find_result_dialog(make_reference_dialog_frame())
    assert match is not None
    assert match.dialog_rect == (242, 220, 287, 151)
    assert match.confirm_point == (502, 355)
    assert match.dialog_score >= 0.99
    assert match.button_score >= 0.99



def test_missing_confirm_button_is_not_actionable():
    frame = make_reference_dialog_frame()
    # The confirm button sits at the measured offset inside the dialog.
    frame[220 + 118:220 + 118 + 34, 242 + 233:242 + 233 + 55] = 0
    assert find_result_dialog(frame) is None


def test_success_banner_alone_does_not_count_as_result_dialog():
    gray = np.full((563, 759), 110, dtype=np.uint8)
    cv2.putText(
        gray, "SUCCESS!", (180, 320), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, 245, 8, cv2.LINE_AA
    )
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    assert find_result_dialog(frame) is None


def test_non_uniform_capture_scaling_is_rejected():
    frame = np.full((563, 900, 3), 80, dtype=np.uint8)
    assert find_result_dialog(frame) is None


def test_uniform_capture_scaling_preserves_click_mapping():
    original = make_reference_dialog_frame()
    scaled = cv2.resize(original, (1139, 844), interpolation=cv2.INTER_LINEAR)
    match = find_result_dialog(scaled)
    assert match is not None
    assert abs(match.confirm_point[0] - 754) <= 2
    assert abs(match.confirm_point[1] - 532) <= 2
