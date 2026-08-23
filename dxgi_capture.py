"""Occlusion-safe game-window capture with a desktop fallback."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import threading

import mss
import numpy as np
import win32gui
import win32ui


logger = logging.getLogger("capture")
PW_RENDERFULLCONTENT = 0x00000002


class ScreenCapture:
    """Capture the actual game window even when the web panel covers it.

    MSS captures pixels from the desktop. That means a browser placed above a
    borderless game is mistaken for the game frame. ``PrintWindow`` asks DWM
    for the target window itself and works with the current WoWS renderer. MSS
    remains as a fallback for display modes where PrintWindow is unavailable.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        self.last_backend = "uninitialized"
        self.last_error = ""

    @staticmethod
    def _window_size(hwnd):
        rect = ctypes.wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return None
        return rect.left, rect.top, width, height

    @staticmethod
    def _usable(image):
        return (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[2] >= 3
            and image.size > 0
            and float(image.mean()) > 3.0
            and float(image.std()) > 4.0
        )

    def _capture_print_window(self, hwnd, width, height):
        window_dc = win32gui.GetWindowDC(hwnd)
        if not window_dc:
            return None
        source_dc = win32ui.CreateDCFromHandle(window_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        try:
            bitmap.CreateCompatibleBitmap(source_dc, width, height)
            memory_dc.SelectObject(bitmap)
            rendered = ctypes.windll.user32.PrintWindow(
                hwnd,
                memory_dc.GetSafeHdc(),
                PW_RENDERFULLCONTENT,
            )
            if not rendered:
                return None
            raw = bitmap.GetBitmapBits(True)
            image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            return np.ascontiguousarray(image[:, :, :3])
        finally:
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
            memory_dc.DeleteDC()
            source_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, window_dc)

    def _mss(self):
        capture = getattr(self._thread_local, "mss", None)
        if capture is None:
            capture = mss.MSS()
            self._thread_local.mss = capture
        return capture

    def _capture_desktop(self, left, top, width, height):
        monitor = {"left": left, "top": top, "width": width, "height": height}
        raw = np.array(self._mss().grab(monitor))
        return np.ascontiguousarray(raw[:, :, :3])

    def capture_window(self, hwnd):
        """Return one BGR frame using the safest order for current focus.

        WoWS can leave ``PrintWindow`` frozen on an old battle frame even
        though the call succeeds. While the game is foreground, MSS is both
        unobstructed and live, so prefer it. When another window is foreground,
        retain the occlusion-safe PrintWindow path.
        """
        geometry = self._window_size(hwnd)
        if geometry is None:
            self.last_error = "window_rect_invalid"
            return None
        left, top, width, height = geometry

        with self._lock:
            errors = []
            game_is_foreground = int(
                ctypes.windll.user32.GetForegroundWindow() or 0
            ) == int(hwnd)
            backends = (
                (
                    ("mss_desktop_live", lambda: self._capture_desktop(left, top, width, height)),
                    ("print_window", lambda: self._capture_print_window(hwnd, width, height)),
                )
                if game_is_foreground
                else (
                    ("print_window", lambda: self._capture_print_window(hwnd, width, height)),
                    ("mss_desktop_fallback", lambda: self._capture_desktop(left, top, width, height)),
                )
            )
            for backend_name, capture in backends:
                try:
                    image = capture()
                    if self._usable(image):
                        self.last_backend = backend_name
                        self.last_error = ""
                        return image
                    errors.append(f"{backend_name}_blank")
                except Exception as error:
                    errors.append(f"{backend_name}_failed:{error}")

            self.last_backend = "failed"
            self.last_error = "; ".join(errors)
            logger.warning("游戏画面截取失败: %s", self.last_error)
            return None
