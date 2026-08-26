import numpy as np

import dxgi_capture
from dxgi_capture import ScreenCapture


def frame(value=80):
    image = np.full((40, 60, 3), value, dtype=np.uint8)
    image[:, ::2] = value + 20
    return image


def test_focused_game_client_capture_is_preferred(monkeypatch):
    capture = ScreenCapture()
    monkeypatch.setattr(dxgi_capture, "ensure_game_window_foreground", lambda _hwnd: True)
    capture._window_size = lambda _hwnd: (0, 0, 60, 40)
    capture._capture_desktop = lambda *_args: frame()
    capture._capture_print_window = lambda *_args: (_ for _ in ()).throw(
        AssertionError("PrintWindow should not run while game is focused")
    )

    image = capture.capture_window(1)

    assert image.shape == (40, 60, 3)
    assert capture.last_backend == "mss_game_client"


def test_printwindow_is_only_a_target_window_fallback(monkeypatch):
    capture = ScreenCapture()
    monkeypatch.setattr(dxgi_capture, "ensure_game_window_foreground", lambda _hwnd: True)
    capture._window_size = lambda _hwnd: (0, 0, 60, 40)
    capture._capture_print_window = lambda _hwnd, _w, _h: frame()
    capture._capture_desktop = lambda *_args: None

    image = capture.capture_window(1)

    assert image.shape == (40, 60, 3)
    assert capture.last_backend == "print_window"


def test_capture_refuses_to_read_when_game_cannot_be_foregrounded(monkeypatch):
    capture = ScreenCapture()
    monkeypatch.setattr(dxgi_capture, "ensure_game_window_foreground", lambda _hwnd: False)
    capture._capture_desktop = lambda *_args: (_ for _ in ()).throw(
        AssertionError("desktop pixels must not be read")
    )

    assert capture.capture_window(1) is None
    assert capture.last_error == "game_window_not_foreground"
