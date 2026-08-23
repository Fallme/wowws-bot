from core.keyboard import KeyboardController


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


def test_throttle_uses_discrete_notches_only_when_target_changes():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.set_movement(1.0, 0.0)
    assert backend.events == [("tap", "w")] * 4

    controller.set_movement(1.0, 0.0)
    assert backend.events == [("tap", "w")] * 4

    controller.set_movement(-0.5, 0.0)
    assert backend.events[-6:] == [("tap", "s")] * 6
    assert controller.last_dispatch.throttle_notch == -2


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

    assert backend.events.count(("tap", "s")) == 3
    assert backend.events.count(("tap", "q")) >= 2
    for key in ("a", "d", "q", "e", "w", "s"):
        assert ("up", key) in backend.events
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


def test_tactical_map_and_autopilot_takeover_use_native_keys():
    backend = RecordingBackend()
    controller = KeyboardController(backend)

    controller.toggle_tactical_map()
    controller.takeover_from_autopilot()

    assert backend.events[0] == ("tap", "m")
    assert backend.events.count(("tap", "w")) == 4
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

    assert first_count == 4
    assert backend.events.count(("tap", "w")) == 8
    assert controller.last_dispatch.action == "full_speed_reassert"
    assert controller.last_dispatch.throttle_notch == 4
