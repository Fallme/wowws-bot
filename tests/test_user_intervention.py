from types import SimpleNamespace

from core.intervention import UserInterventionMonitor


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
