import numpy as np

from dxgi_capture import ScreenCapture


def frame(value=80):
    image = np.full((40, 60, 3), value, dtype=np.uint8)
    image[:, ::2] = value + 20
    return image


def test_window_capture_is_preferred_over_desktop_capture():
    capture = ScreenCapture()
    capture._window_size = lambda _hwnd: (0, 0, 60, 40)
    capture._capture_print_window = lambda _hwnd, _w, _h: frame()
    capture._capture_desktop = lambda *_args: (_ for _ in ()).throw(
        AssertionError("desktop fallback should not run")
    )

    image = capture.capture_window(1)

    assert image.shape == (40, 60, 3)
    assert capture.last_backend == "print_window"


def test_desktop_capture_is_only_a_fallback():
    capture = ScreenCapture()
    capture._window_size = lambda _hwnd: (0, 0, 60, 40)
    capture._capture_print_window = lambda _hwnd, _w, _h: None
    capture._capture_desktop = lambda *_args: frame()

    image = capture.capture_window(1)

    assert image.shape == (40, 60, 3)
    assert capture.last_backend == "mss_desktop_fallback"
