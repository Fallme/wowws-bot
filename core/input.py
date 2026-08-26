"""Input-controller selection.

The native keyboard backend is the safe default because the game did not
accept the project's unconfigured virtual Xbox controller during live tests.
The old vgamepad path remains available through WOWS_INPUT_BACKEND=vgamepad.
"""

from __future__ import annotations

import os

from core.keyboard import KeyboardController


DEFAULT_INPUT_BACKEND = "windows_native_keyboard"
SUPPORTED_INPUT_BACKENDS = {DEFAULT_INPUT_BACKEND, "vgamepad_xbox360"}


def configured_input_backend() -> str:
    value = os.environ.get("WOWS_INPUT_BACKEND", DEFAULT_INPUT_BACKEND).strip().lower()
    aliases = {
        "keyboard": DEFAULT_INPUT_BACKEND,
        "sendinput": DEFAULT_INPUT_BACKEND,
        "windows_sendinput_keyboard": DEFAULT_INPUT_BACKEND,
        "gamepad": "vgamepad_xbox360",
        "vgamepad": "vgamepad_xbox360",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_INPUT_BACKENDS:
        raise ValueError(f"Unsupported input backend: {value}")
    return value


def create_input_controller(backend: str | None = None, *, hwnd=None):
    selected = backend or configured_input_backend()
    if selected == DEFAULT_INPUT_BACKEND:
        focus_guard = None
        if hwnd:
            from core.window import ensure_game_window_foreground

            focus_guard = lambda: ensure_game_window_foreground(hwnd)
        return KeyboardController(focus_guard=focus_guard)
    if selected == "vgamepad_xbox360":
        from core.gamepad import GamepadController

        return GamepadController()
    raise ValueError(f"Unsupported input backend: {selected}")
