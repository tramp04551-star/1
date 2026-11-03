import ctypes

from app import (
    get_window_client_region, is_window_foreground, map_game_roi,
    smoothed_mouse_position,
)


class FakeUser32:
    def __init__(self, *, valid=True, foreground=100, root=42, minimized=False):
        self.valid = valid
        self.foreground = foreground
        self.root = root
        self.minimized = minimized

    def IsWindow(self, _hwnd):
        return self.valid

    def IsIconic(self, _hwnd):
        return self.minimized

    def GetForegroundWindow(self):
        return self.foreground

    def GetAncestor(self, _hwnd, _flags):
        return self.root

    def GetClientRect(self, _hwnd, rect_ptr):
        rect = rect_ptr._obj
        rect.left, rect.top = 0, 0
        rect.right, rect.bottom = 760, 560
        return 1

    def ClientToScreen(self, _hwnd, point_ptr):
        point = point_ptr._obj
        point.x, point.y = 120, 80
        return 1


def test_foreground_gate_accepts_selected_window_and_child():
    assert is_window_foreground(42, FakeUser32(root=42))


def test_foreground_gate_rejects_another_application():
    assert not is_window_foreground(42, FakeUser32(root=99))


def test_foreground_gate_rejects_desktop_mode_without_selected_window():
    assert not is_window_foreground(None, FakeUser32(root=42))


def test_foreground_gate_rejects_destroyed_window():
    assert not is_window_foreground(42, FakeUser32(valid=False))


def test_client_capture_region_uses_client_origin_and_dimensions():
    region = get_window_client_region(42, FakeUser32())
    assert region == {
        "left": 120,
        "top": 80,
        "width": 760,
        "height": 560,
    }


def test_client_capture_region_rejects_minimized_window():
    assert get_window_client_region(42, FakeUser32(minimized=True)) is None



def test_fixed_roi_maps_from_client_origin_without_scaling():
    client = {"left": 120, "top": 80, "width": 1400, "height": 900}
    assert map_game_roi(client) == {
        "left": 426,
        "top": 199,
        "width": 759,
        "height": 563,
    }


def test_fixed_roi_rejects_client_area_that_is_too_small():
    client = {"left": 0, "top": 0, "width": 1000, "height": 800}
    try:
        map_game_roi(client)
    except ValueError as exc:
        assert "不足以容纳固定 ROI" in str(exc)
    else:
        raise AssertionError("undersized client area must be rejected")


def test_fixed_roi_rejects_client_resize_after_start():
    client = {"left": 0, "top": 0, "width": 1401, "height": 900}
    try:
        map_game_roi(client, expected_client_size=(1400, 900))
    except ValueError as exc:
        assert "尺寸已变化" in str(exc)
    else:
        raise AssertionError("client resize must invalidate the ROI mapping")



def test_cursor_dead_zone_suppresses_small_jitter():
    assert smoothed_mouse_position((100, 100), (105, 106), 0.85) is None


def test_cursor_smoothing_uses_reference_factor_outside_dead_zone():
    assert smoothed_mouse_position((100, 100), (120, 100), 0.85) == (117, 100)
