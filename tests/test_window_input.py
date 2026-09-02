import ctypes

import win32con

from core import window


class DpiVirtualizedUser32:
    def __init__(self):
        self.cursor = [120, 240]
        self.set_positions = []
        self.events = []

    def GetCursorPos(self, pointer):
        pointer._obj.x, pointer._obj.y = self.cursor
        return True

    def GetSystemMetrics(self, index):
        return {
            76: -2560,
            77: 0,
            78: 5120,
            79: 1600,
        }.get(index, 0)

    def mouse_event(self, flags, x, y, data, extra):
        self.events.append(flags)
        if flags & win32con.MOUSEEVENTF_MOVE:
            # Reproduce the 150% DPI mismatch seen in the live worker.
            self.cursor[:] = [853, 24]

    def SetCursorPos(self, x, y):
        self.cursor[:] = [int(x), int(y)]
        self.set_positions.append((int(x), int(y)))
        return True


class PhysicalCursorUser32(DpiVirtualizedUser32):
    def __init__(self):
        super().__init__()
        self.physical_cursor = [120, 240]
        self.physical_positions = []

    def GetPhysicalCursorPos(self, pointer):
        pointer._obj.x, pointer._obj.y = self.physical_cursor
        return True

    def SetPhysicalCursorPos(self, x, y):
        self.physical_cursor[:] = [int(x), int(y)]
        self.physical_positions.append((int(x), int(y)))
        return True


def test_physical_click_falls_back_to_verified_physical_coordinates(monkeypatch):
    user32 = DpiVirtualizedUser32()
    monkeypatch.setattr(window.ctypes.windll, "user32", user32)
    monkeypatch.setattr(window.time, "sleep", lambda _seconds: None)

    assert window.physical_click(1280, 36)

    assert (1280, 36) in user32.set_positions
    assert user32.events.count(win32con.MOUSEEVENTF_LEFTDOWN) == 1
    assert user32.events.count(win32con.MOUSEEVENTF_LEFTUP) == 1
    # The user's original cursor position is restored after the single click.
    assert user32.set_positions[-1] == (120, 240)


def test_physical_click_prefers_unvirtualized_cursor_apis(monkeypatch):
    user32 = PhysicalCursorUser32()
    monkeypatch.setattr(window.ctypes.windll, "user32", user32)
    monkeypatch.setattr(window.time, "sleep", lambda _seconds: None)

    assert window.physical_click(998, 976)

    assert user32.physical_positions == [(998, 976), (120, 240)]
    assert user32.set_positions == []
    assert user32.events.count(win32con.MOUSEEVENTF_LEFTDOWN) == 1
    assert user32.events.count(win32con.MOUSEEVENTF_LEFTUP) == 1


def test_successful_physical_click_acknowledges_automation_input(monkeypatch):
    user32 = PhysicalCursorUser32()
    acknowledgements = []
    monkeypatch.setattr(window.ctypes.windll, "user32", user32)
    monkeypatch.setattr(window.time, "sleep", lambda _seconds: None)
    window.set_automation_input_observer(lambda: acknowledgements.append(True))
    try:
        assert window.physical_click(998, 976)
    finally:
        window.set_automation_input_observer(None)

    assert acknowledgements == [True]


def test_physical_scroll_uses_same_unvirtualized_coordinates(monkeypatch):
    user32 = PhysicalCursorUser32()
    monkeypatch.setattr(window.ctypes.windll, "user32", user32)
    monkeypatch.setattr(window.time, "sleep", lambda _seconds: None)

    assert window.physical_scroll(1280, 1350, -6)

    assert user32.physical_positions == [(1280, 1350), (120, 240)]
    assert win32con.MOUSEEVENTF_WHEEL in user32.events

