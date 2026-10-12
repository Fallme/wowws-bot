"""Verify torpedo aim/fire injection without touching Windows input devices."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import core.keyboard as keyboard


@pytest.fixture
def native_backend(monkeypatch):
    captured = []

    def accept(count, events, structure_size):
        captured.append((count, structure_size, [
            (event.type, event.mi.dx, event.mi.dy, event.mi.mouseData,
             event.mi.dwFlags, event.mi.time, event.mi.dwExtraInfo)
            for event in events
        ]))
        return count

    user32 = SimpleNamespace(
        SendInput=Mock(side_effect=accept),
        SetCursorPos=Mock(), SetPhysicalCursorPos=Mock(),
        GetCursorPos=Mock(), GetPhysicalCursorPos=Mock(),
    )
    kernel32 = SimpleNamespace(GetTickCount=Mock(return_value=4200))
    monkeypatch.setattr(keyboard.ctypes, "windll",
                        SimpleNamespace(user32=user32, kernel32=kernel32), raising=False)
    monkeypatch.setattr(keyboard.ctypes, "WinError",
                        lambda: OSError("injected input rejected"), raising=False)
    return keyboard.SendInputBackend(), user32, captured


def test_relative_move_uses_signed_delta_without_absolute_cursor_operations(native_backend):
    backend, user32, captured = native_backend
    assert backend.move_relative(-27, 13) is True
    assert captured == [(1, keyboard.ctypes.sizeof(keyboard._INPUT), [
        (keyboard.INPUT_MOUSE, -27, 13, 0, keyboard.MOUSEEVENTF_MOVE, 0, 0),
    ])]
    assert backend.injection_generation == 1
    assert backend.last_injected_tick_ms == 4200
    assert not backend.recent_injected_key_ticks_ms
    for name in ("SetCursorPos", "SetPhysicalCursorPos", "GetCursorPos", "GetPhysicalCursorPos"):
        getattr(user32, name).assert_not_called()


def test_relative_move_rejection_has_no_input_acknowledgement(native_backend):
    backend, user32, _captured = native_backend
    user32.SendInput.side_effect = None
    user32.SendInput.return_value = 0
    controller = keyboard.KeyboardController(backend, focus_guard=lambda: True)
    observer = Mock()
    controller.set_automation_observer(observer)
    with pytest.raises(OSError, match="rejected"):
        controller.aim_relative(20, -5)
    assert backend.injection_generation == 0
    assert backend.last_injected_tick_ms is None
    assert controller.dispatch_count == 0
    assert controller.last_dispatch is None
    observer.assert_not_called()


@pytest.mark.parametrize("action,args", [("aim_relative", (20, -5)), ("fire_torpedo", ())])
def test_torpedo_input_requires_focus_before_dispatch(native_backend, action, args):
    backend, user32, _captured = native_backend
    focus = Mock(return_value=False)
    controller = keyboard.KeyboardController(backend, focus_guard=focus)
    with pytest.raises(RuntimeError, match="前台"):
        getattr(controller, action)(*args)
    focus.assert_called_once()
    user32.SendInput.assert_not_called()
    assert backend.injection_generation == 0
    assert controller.last_dispatch is None


def test_aim_and_atomic_fire_each_acknowledge_automation_even_in_same_clock_tick(native_backend):
    backend, user32, captured = native_backend
    focus = Mock(return_value=True)
    controller = keyboard.KeyboardController(backend, focus_guard=focus)
    observed = []
    controller.set_automation_observer(lambda current: observed.append(current.last_dispatch.action))
    assert controller.aim_relative(18, -4) is True
    assert controller.fire_torpedo() is True
    assert observed == ["torpedo_aim", "torpedo_launch"]
    assert focus.call_count == 2
    assert backend.injection_generation == 2
    assert controller.dispatch_count == 2
    assert captured[1] == (2, keyboard.ctypes.sizeof(keyboard._INPUT), [
        (keyboard.INPUT_MOUSE, 0, 0, 0, keyboard.MOUSEEVENTF_LEFTDOWN, 0, 0),
        (keyboard.INPUT_MOUSE, 0, 0, 0, keyboard.MOUSEEVENTF_LEFTUP, 0, 0),
    ])
    user32.SetPhysicalCursorPos.assert_not_called()
    user32.SetCursorPos.assert_not_called()


@pytest.mark.parametrize("action,args,device_method", [
    ("aim_relative", (18, -4), "move_relative"),
    ("fire_torpedo", (), "left_click"),
])
def test_explicit_failed_backend_dispatch_is_not_recorded(action, args, device_method):
    backend = SimpleNamespace(**{device_method: Mock(return_value=False)})
    controller = keyboard.KeyboardController(backend, focus_guard=lambda: True)
    assert getattr(controller, action)(*args) is False
    assert controller.dispatch_count == 0
    assert controller.last_dispatch is None


def test_rejected_atomic_fire_does_not_acknowledge_launch(native_backend):
    backend, user32, _captured = native_backend
    user32.SendInput.side_effect = None
    user32.SendInput.return_value = 0
    controller = keyboard.KeyboardController(backend, focus_guard=lambda: True)
    observer = Mock()
    controller.set_automation_observer(observer)
    with pytest.raises(OSError, match="rejected"):
        controller.fire_torpedo()
    assert backend.injection_generation == 0
    assert controller.dispatch_count == 0
    observer.assert_not_called()
