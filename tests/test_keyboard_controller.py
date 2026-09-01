import pytest

from core.keyboard import KeyboardController, VK


class RecordingBackend:
    def __init__(self):
        self.events = []

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))

    def tap(self, key):
        self.events.append(("tap", key))

    def left_click(self):
        self.events.append(("click", "left"))


class TelegraphBackend(RecordingBackend):
    """Simulate the game's latched engine telegraph independently of the cache."""

    def __init__(self, actual_notch=0, actual_rudder=0):
        super().__init__()
        self.actual_notch = actual_notch
        self.actual_rudder = actual_rudder
        self.notch_history = [actual_notch]

    def tap(self, key):
        super().tap(key)
        if key == "w":
            self.actual_notch = min(4, self.actual_notch + 1)
        elif key == "s":
            self.actual_notch = max(-4, self.actual_notch - 1)
        elif key == "q":
            self.actual_rudder = max(-2, self.actual_rudder - 1)
        elif key == "e":
            self.actual_rudder = min(2, self.actual_rudder + 1)
        self.notch_history.append(self.actual_notch)


def test_throttle_is_forward_only_and_resynchronizes_before_reducing_speed():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.set_movement(1.0, 0.0)
    assert backend.events == [("tap", "w")] * 8

    controller.set_movement(1.0, 0.0)
    assert backend.events == [("tap", "w")] * 8

    controller.set_movement(0.5, 0.0)
    assert backend.events[-10:] == [("tap", "w")] * 8 + [("tap", "s")] * 2
    assert controller.last_dispatch.throttle_notch == 2

    controller.set_movement(-0.5, 0.0)
    assert backend.events[-12:] == [("tap", "w")] * 8 + [("tap", "s")] * 4
    assert controller.last_dispatch.throttle_notch == 0


def test_completed_native_dispatch_notifies_automation_observer_once():
    class TickBackend(RecordingBackend):
        last_injected_tick_ms = None

        def tap(self, key):
            super().tap(key)
            self.last_injected_tick_ms = (self.last_injected_tick_ms or 100) + 1

    backend = TickBackend()
    controller = KeyboardController(backend)
    observed = []
    controller.set_automation_observer(observed.append)

    controller.full_speed()

    assert observed == [controller]


@pytest.mark.parametrize("actual_notch", range(-4, 5))
def test_reduction_never_crosses_stop_when_cached_and_actual_telegraph_disagree(
    actual_notch,
):
    backend = TelegraphBackend(actual_notch)
    controller = KeyboardController(backend)
    controller._throttle_notch = 4  # Simulate a stale FULL-ahead controller cache.

    controller.set_movement(0.32, 0.0)

    assert backend.actual_notch == 1
    assert min(backend.notch_history[-4:]) >= 1
    assert controller.last_dispatch.throttle_notch == 1


@pytest.mark.parametrize("actual_notch", range(-4, 5))
def test_upshift_also_recovers_when_cached_and_actual_telegraph_disagree(actual_notch):
    backend = TelegraphBackend(actual_notch)
    controller = KeyboardController(backend)
    controller._throttle_notch = 1

    controller.set_movement(1.0, 0.0)

    assert backend.actual_notch == 4
    assert controller.last_dispatch.throttle_notch == 4


@pytest.mark.parametrize("actual_notch", range(-4, 5))
def test_negative_request_is_clamped_to_stop_without_entering_reverse(actual_notch):
    backend = TelegraphBackend(actual_notch)
    controller = KeyboardController(backend)
    controller._throttle_notch = 4

    controller.set_movement(-1.0, 0.0)

    assert backend.actual_notch == 0
    assert min(backend.notch_history[-5:]) >= 0
    assert controller.last_dispatch.throttle_notch == 0


@pytest.mark.parametrize("actual_notch", range(-4, 5))
def test_full_speed_reassert_recovers_from_every_possible_actual_notch(actual_notch):
    backend = TelegraphBackend(actual_notch)
    controller = KeyboardController(backend)
    controller._throttle_notch = 4

    controller.reassert_full_speed()

    assert backend.actual_notch == 4
    assert controller.last_dispatch.throttle_notch == 4


@pytest.mark.parametrize("actual_notch", range(-4, 5))
@pytest.mark.parametrize("actual_rudder", range(-2, 3))
def test_control_handoff_normalizes_every_engine_and_rudder_state(
    actual_notch,
    actual_rudder,
):
    backend = TelegraphBackend(actual_notch, actual_rudder)
    controller = KeyboardController(backend)
    controller._throttle_notch = 3
    controller._rudder_notch = -1

    controller.resynchronize_forward_controls()

    assert backend.actual_notch == 4
    assert backend.actual_rudder == 0
    assert controller.last_dispatch.throttle_notch == 4
    assert controller.last_dispatch.rudder_notch == 0


def test_rudder_holds_one_key_and_releases_before_switching():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.set_movement(0.0, -0.7)
    controller.set_movement(0.0, -0.4)
    controller.set_movement(0.0, 0.8)
    controller.set_movement(0.0, 0.0)

    assert backend.events == [
        ("tap", "q"),
        ("tap", "q"),
        ("tap", "e"),
        ("tap", "e"),
        ("tap", "e"),
        ("tap", "e"),
        ("tap", "q"),
        ("tap", "q"),
    ]


def test_stop_returns_telegraph_to_zero_and_releases_all_keys():
    backend = RecordingBackend()
    controller = KeyboardController(backend)
    controller.set_movement(0.75, 1.0)

    controller.stop()

    assert backend.events.count(("tap", "s")) == 5
    assert backend.events.count(("tap", "w")) == 16
    assert backend.events.count(("tap", "q")) >= 2
    for key in ("a", "d", "q", "e", "w", "s"):
        assert ("up", key) in backend.events


@pytest.mark.parametrize("actual_notch", range(-4, 5))
@pytest.mark.parametrize("actual_rudder", range(-2, 3))
def test_stop_normalizes_every_actual_state_even_when_both_caches_say_neutral(
    actual_notch,
    actual_rudder,
):
    backend = TelegraphBackend(actual_notch, actual_rudder)
    controller = KeyboardController(backend)
    controller._throttle_notch = 0
    controller._rudder_notch = 0

    controller.stop()

    assert backend.actual_notch == 0
    assert backend.actual_rudder == 0
    assert controller.last_dispatch.action == "neutral_stop"
    assert controller.last_dispatch.action == "neutral_stop"
    assert controller.last_dispatch.throttle_notch == 0


def test_action_bindings_use_native_keyboard_and_mouse():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.fire()
    controller.lock()
    controller.torpedo()
    controller.smoke()
    controller.damage_control()
    controller.heal()

    assert backend.events == [
        ("click", "left"),
        ("tap", "x"),
        ("tap", "3"),
        ("tap", "4"),
        ("tap", "r"),
        ("tap", "t"),
    ]


def test_escape_has_a_valid_windows_virtual_key_binding():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.escape()

    assert VK["esc"] == 0x1B
    assert backend.events == [("tap", "esc")]
    assert controller.last_dispatch.action == "escape"


def test_confirm_has_a_valid_windows_virtual_key_binding():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.confirm()

    assert VK["enter"] == 0x0D
    assert backend.events == [("tap", "enter")]
    assert controller.last_dispatch.action == "confirm"


def test_consumable_cycle_tries_every_common_ship_slot_once():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.use_consumable_cycle()

    assert backend.events == [
        ("tap", "r"),
        ("tap", "t"),
        ("tap", "u"),
        ("tap", "y"),
    ]
    assert controller.last_dispatch.action == "consumable_cycle"
    assert VK["u"] == 0x55
    assert VK["y"] == 0x59


def test_consumable_cycle_continues_when_one_slot_dispatch_fails():
    class OneBrokenSlotBackend(RecordingBackend):
        def tap(self, key):
            if key == "u":
                raise OSError("slot unavailable")
            super().tap(key)

    backend = OneBrokenSlotBackend()
    controller = KeyboardController(backend)

    controller.use_consumable_cycle()

    assert backend.events == [
        ("tap", "r"),
        ("tap", "t"),
        ("tap", "y"),
    ]
    assert controller.last_dispatch.action == "consumable_cycle"


def test_other_consumables_never_press_damage_control_slot():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.use_other_consumables()

    assert backend.events == [
        ("tap", "t"),
        ("tap", "u"),
        ("tap", "y"),
    ]
    assert controller.last_dispatch.action == "other_consumables"


def test_tactical_map_and_autopilot_takeover_use_native_keys():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.toggle_tactical_map()
    controller.takeover_from_autopilot()

    assert backend.events[0] == ("tap", "m")
    assert backend.events.count(("tap", "w")) == 8
    assert controller.last_dispatch.action == "full_speed_reassert"


def test_manual_pause_releases_rudder_but_preserves_engine_notch():
    backend = RecordingBackend()
    controller = KeyboardController(backend)
    controller.set_movement(1.0, 0.8)

    controller.pause_automation()

    assert backend.events.count(("tap", "s")) == 0
    assert backend.events.count(("tap", "q")) == 0
    assert controller.last_dispatch.action == "manual_intervention_pause"
    assert controller.last_dispatch.throttle_notch == 4


def test_reassert_full_speed_resends_w_when_cached_notch_is_already_full():
    backend = RecordingBackend()
    controller = KeyboardController(backend)
    controller.full_speed()
    first_count = backend.events.count(("tap", "w"))

    controller.reassert_full_speed()

    assert first_count == 8
    assert backend.events.count(("tap", "w")) == 16
    assert controller.last_dispatch.action == "full_speed_reassert"
    assert controller.last_dispatch.throttle_notch == 4


def test_focus_guard_runs_before_native_key_input():
    backend = RecordingBackend()
    focus_checks = []
    controller = KeyboardController(
        backend,
        focus_guard=lambda: focus_checks.append("checked") or True,
    )

    controller.full_speed()

    assert focus_checks == ["checked"]
    assert backend.events == [("tap", "w")] * 8


def test_focus_guard_blocks_native_key_input_when_game_cannot_activate():
    backend = RecordingBackend()
    controller = KeyboardController(backend, focus_guard=lambda: False)

    with pytest.raises(RuntimeError, match="游戏窗口不在前台"):
        controller.full_speed()

    assert backend.events == []
