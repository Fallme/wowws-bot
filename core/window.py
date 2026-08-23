"""Windows game-window discovery and physical cursor operations."""

import ctypes
import ctypes.wintypes
import time

import win32con
import win32gui


def _enable_dpi_awareness():
    """Keep captures and clicks in the same physical-pixel coordinate space."""
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


_enable_dpi_awareness()


def activate_window(hwnd):
    """Restore and foreground a game window."""
    try:
        ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        time.sleep(0.1)
        ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)  # Alt down
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)  # Alt up
        time.sleep(0.2)
        return True
    except Exception:
        return False


def find_game_window():
    """Return visible matching windows as ``(hwnd, title, rect)`` tuples."""
    targets = ("world of warships", "战舰世界", "wows")
    result = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if any(target in title.lower() for target in targets):
            rect = get_window_rect(hwnd)
            result.append(
                (
                    hwnd,
                    title,
                    (rect["left"], rect["top"], rect["right"], rect["bottom"]),
                )
            )
        return True

    win32gui.EnumWindows(callback, None)
    return result


def get_window_rect(hwnd):
    """Return DPI-aware physical window coordinates."""
    rect = ctypes.wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    return {
        "left": rect.left,
        "top": rect.top,
        "right": rect.right,
        "bottom": rect.bottom,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }


def physical_click(screen_x, screen_y, extra_delay=0.0):
    """Click a physical screen coordinate and restore the original cursor."""
    original = ctypes.wintypes.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(original)):
        return False

    try:
        virtual_left = ctypes.windll.user32.GetSystemMetrics(76)
        virtual_top = ctypes.windll.user32.GetSystemMetrics(77)
        virtual_width = ctypes.windll.user32.GetSystemMetrics(78)
        virtual_height = ctypes.windll.user32.GetSystemMetrics(79)
        if virtual_width <= 1 or virtual_height <= 1:
            return False

        normalized_x = int(
            (screen_x - virtual_left) * 65535 / (virtual_width - 1)
        )
        normalized_y = int(
            (screen_y - virtual_top) * 65535 / (virtual_height - 1)
        )
        normalized_x = max(0, min(normalized_x, 65535))
        normalized_y = max(0, min(normalized_y, 65535))
        move_flags = (
            win32con.MOUSEEVENTF_MOVE
            | win32con.MOUSEEVENTF_ABSOLUTE
            | 0x4000  # MOUSEEVENTF_VIRTUALDESK
        )
        ctypes.windll.user32.mouse_event(
            move_flags, normalized_x, normalized_y, 0, 0
        )
        time.sleep(0.15 + max(0.0, extra_delay))
        moved = ctypes.wintypes.POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(moved)):
            return False
        if abs(moved.x - screen_x) > 3 or abs(moved.y - screen_y) > 3:
            return False
        ctypes.windll.user32.mouse_event(
            win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0
        )
        time.sleep(0.05)
        ctypes.windll.user32.mouse_event(
            win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0
        )
        time.sleep(0.1)
        return True
    finally:
        ctypes.windll.user32.SetCursorPos(original.x, original.y)


def physical_scroll(screen_x, screen_y, notches):
    """Scroll at a physical coordinate and restore the user's cursor."""
    original = ctypes.wintypes.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(original)):
        return False
    try:
        ctypes.windll.user32.SetCursorPos(int(screen_x), int(screen_y))
        time.sleep(0.10)
        ctypes.windll.user32.mouse_event(
            win32con.MOUSEEVENTF_WHEEL,
            0,
            0,
            int(notches) * 120,
            0,
        )
        time.sleep(0.35)
        return True
    finally:
        ctypes.windll.user32.SetCursorPos(original.x, original.y)


def click_center(hwnd):
    rect = get_window_rect(hwnd)
    return physical_click(
        rect["left"] + rect["width"] // 2,
        rect["top"] + rect["height"] // 2,
    )
