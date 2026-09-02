from types import SimpleNamespace

from core.intervention import UserInterventionMonitor, _keyboard_activity


def test_keyboard_activity_ignores_held_high_bit_without_new_transition(monkeypatch):
    class User32:
        @staticmethod
        def GetAsyncKeyState(_virtual_key):
            return 0x8000

    monkeypatch.setattr("core.intervention.ctypes.windll.user32", User32())

    assert not _keyboard_activity()


def test_foreground_user_input_pauses_for_configured_window():
    ticks = iter([100, 200, 200])
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=4,
        input_tick_reader=lambda: next(ticks),
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=5000)

    monitor.reset()
    assert monitor.poll(controller, now=10)
    assert monitor.poll(controller, now=13)


def test_automation_tick_is_ignored_but_background_keyboard_still_pauses():
    current = [100]
    foreground = [7]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=4,
        input_tick_reader=lambda: current[0],
        foreground_reader=lambda: foreground[0],
    )
    controller = SimpleNamespace(last_injected_tick_ms=200)
    monitor.reset()

    current[0] = 200
    assert not monitor.poll(controller, now=10)

    controller.last_injected_tick_ms = 9000
    current[0] = 800
    foreground[0] = 9
    assert monitor.poll(controller, now=11)


def test_delayed_focus_injection_is_not_user_intervention():
    current = [1000]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=4,
        input_tick_reader=lambda: current[0],
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=1300)
    monitor.reset()

    # A just-injected key can reach GetLastInputInfo shortly before the
    # backend records its marker.
    current[0] = 1185
    assert not monitor.poll(controller, now=10)


def test_input_outside_automation_window_still_pauses():
    current = [1000]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=4,
        input_tick_reader=lambda: current[0],
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=2000)
    monitor.reset()

    current[0] = 1100
    assert monitor.poll(controller, now=10)


def test_mouse_input_never_pauses_automation():
    current = [1000]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        input_tick_reader=lambda: current[0],
        keyboard_activity_reader=lambda: False,
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    current[0] = 1100
    assert not monitor.poll(controller, now=10)
    assert not monitor.latched


def test_web_pause_latches_immediately_until_resume(tmp_path):
    current = [100]
    pause_path = tmp_path / "pause.request"
    resume_path = tmp_path / "resume.request"
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        latch_seconds=15,
        pause_path=pause_path,
        resume_path=resume_path,
        input_tick_reader=lambda: current[0],
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    pause_path.write_text("pause", encoding="utf-8")
    assert monitor.poll(controller, now=10)
    assert monitor.latched
    assert monitor.web_paused

    pause_path.unlink()
    resume_path.write_text("resume", encoding="utf-8")
    assert not monitor.poll(controller, now=11)
    assert monitor.resumed_from_web


def test_pause_state_can_be_checked_without_consuming_resume(tmp_path):
    pause_path = tmp_path / "pause.request"
    resume_path = tmp_path / "resume.request"
    monitor = UserInterventionMonitor(
        7,
        pause_path=pause_path,
        resume_path=resume_path,
        input_tick_reader=lambda: 100,
        foreground_reader=lambda: 7,
    )
    monitor.reset()

    pause_path.write_text("pause", encoding="utf-8")
    assert monitor.command_generation_paused(now=10)
    pause_path.unlink()
    resume_path.write_text("resume", encoding="utf-8")
    assert not monitor.command_generation_paused(now=10)
    assert resume_path.exists()


def test_continuous_user_input_latches_until_web_resume(tmp_path):
    current = [100]
    resume_path = tmp_path / "resume.request"
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        latch_seconds=15,
        resume_path=resume_path,
        input_tick_reader=lambda: current[0],
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    for now in (10, 15, 20, 25):
        current[0] += 10
        assert monitor.poll(controller, now=now)
    assert monitor.latched
    assert monitor.poll(controller, now=40)

    resume_path.write_text("resume", encoding="utf-8")
    assert not monitor.poll(controller, now=41)
    assert monitor.resumed_from_web
    assert not monitor.latched


def test_short_user_input_resumes_automatically_after_five_seconds():
    current = [100]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        latch_seconds=15,
        input_tick_reader=lambda: current[0],
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    current[0] = 110
    assert monitor.poll(controller, now=10)
    assert not monitor.poll(controller, now=15.1)
    assert not monitor.latched


def test_automation_ack_drains_old_key_before_later_mouse_movement():
    current = [1000]
    keyboard_transitions = iter([False, True, False])
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        input_tick_reader=lambda: current[0],
        keyboard_activity_reader=lambda: next(keyboard_transitions),
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(last_injected_tick_ms=1100)
    monitor.reset()

    # A known automated M press leaves one keyboard transition pending.
    current[0] = 1100
    monitor.acknowledge_automation(controller)

    # The subsequent cursor warp changes LASTINPUTINFO but is mouse-only and
    # must not inherit the old M transition or pause the task.
    current[0] = 1400
    assert not monitor.poll(controller, now=10)
    assert not monitor.latched


def test_switching_away_from_game_pauses_before_focus_can_be_stolen_back():
    current_tick = [100]
    foreground = [7]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        latch_seconds=20,
        input_tick_reader=lambda: current_tick[0],
        keyboard_activity_reader=lambda: False,
        foreground_reader=lambda: foreground[0],
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    assert not monitor.poll(controller, now=10)
    foreground[0] = 99
    assert monitor.poll(controller, now=10.1)
    assert monitor.pause_until == 15.1
    assert monitor.poll(controller, now=15.0)
    assert not monitor.poll(controller, now=15.2)


def test_mouse_activity_after_switching_away_never_extends_or_latches_pause():
    current_tick = [100]
    foreground = [7]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        latch_seconds=20,
        input_tick_reader=lambda: current_tick[0],
        keyboard_activity_reader=lambda: False,
        foreground_reader=lambda: foreground[0],
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    foreground[0] = 99
    assert monitor.poll(controller, now=10)
    assert monitor.last_trigger == "window_switch"

    # The explicit window switch owns one five-second pause. Mouse-only use in
    # the other app must not extend it or turn it into a manual-resume latch.
    current_tick[0] += 10
    assert monitor.poll(controller, now=14.0)
    for now in (15.1, 18.0, 22.0, 26.0, 30.0):
        current_tick[0] += 10
        assert not monitor.poll(controller, now=now)
    assert not monitor.latched
    assert monitor.last_trigger == "window_switch"


def test_any_tick_from_a_long_automation_batch_is_not_user_input():
    current_tick = [1000]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        input_tick_reader=lambda: current_tick[0],
        keyboard_activity_reader=lambda: True,
        foreground_reader=lambda: 7,
    )
    controller = SimpleNamespace(
        last_injected_tick_ms=1400,
        recent_injected_key_ticks_ms=(1100, 1200, 1300, 1400),
    )
    monitor.reset()

    current_tick[0] = 1200
    assert not monitor.poll(controller, now=10)
    assert not monitor.latched


def test_mouse_activity_inside_game_still_does_not_extend_expired_keyboard_pause():
    current_tick = [100]
    foreground = [7]
    keyboard = [False]
    monitor = UserInterventionMonitor(
        7,
        pause_seconds=5,
        latch_seconds=20,
        input_tick_reader=lambda: current_tick[0],
        keyboard_activity_reader=lambda: keyboard[0],
        foreground_reader=lambda: foreground[0],
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    keyboard[0] = True
    current_tick[0] = 110
    assert monitor.poll(controller, now=10)
    keyboard[0] = False
    current_tick[0] = 120
    assert not monitor.poll(controller, now=15.1)
    assert not monitor.latched


def test_initial_non_game_foreground_does_not_block_startup_focus():
    monitor = UserInterventionMonitor(
        7,
        input_tick_reader=lambda: 100,
        keyboard_activity_reader=lambda: False,
        foreground_reader=lambda: 99,
    )
    controller = SimpleNamespace(last_injected_tick_ms=9000)
    monitor.reset()

    assert not monitor.poll(controller, now=10)
