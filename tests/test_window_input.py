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

