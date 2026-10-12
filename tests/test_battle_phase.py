from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.battle_phase import BattlePhase, BattlePhaseTracker
from core.ui import ScreenState
from main import run_battle, wait_for_battle


def test_loading_preparation_and_active_are_distinct():
    tracker = BattlePhaseTracker()
    assert tracker.observe(ScreenState.LOADING, None, 0) == BattlePhase.LOADING
    for time, clock in enumerate([30, 29, 1200, 1200], start=1):
        assert tracker.observe(ScreenState.BATTLE, clock, time) == BattlePhase.PREPARING
    assert tracker.observe(ScreenState.BATTLE, 1199, 5) == BattlePhase.ACTIVE


def test_frozen_clock_and_missing_ocr_do_not_enable_driving():
    tracker = BattlePhaseTracker()
    for time, clock in enumerate([1199, 1199, None, 1198]):
        assert tracker.observe(ScreenState.BATTLE, clock, time) == BattlePhase.UNKNOWN


def test_new_match_resets_active_evidence():
    tracker = BattlePhaseTracker()
    tracker.observe(ScreenState.BATTLE, 1100, 0)
    assert tracker.observe(ScreenState.BATTLE, 1099, 1) == BattlePhase.ACTIVE
    assert tracker.observe(ScreenState.BATTLE, 59, 2) == BattlePhase.ACTIVE
    tracker.observe(ScreenState.LOADING, None, 3)
    assert tracker.observe(ScreenState.BATTLE, 1200, 4) == BattlePhase.PREPARING


@pytest.mark.parametrize("previous,current,elapsed", [(1200, 900, 1), (1199, 1198, 0), (1199, 1198, 20)])
def test_clock_jump_or_stale_sample_cannot_confirm_start(previous, current, elapsed):
    tracker = BattlePhaseTracker()
    tracker.observe(ScreenState.BATTLE, previous, 0)
    assert tracker.observe(ScreenState.BATTLE, current, elapsed) != BattlePhase.ACTIVE


@pytest.mark.parametrize("clock", [1200, 30, None])
def test_preparation_hud_never_sends_opening_commands(clock):
    vision = SimpleNamespace(
        grab=Mock(return_value=object()),
        read_battle_clock_seconds=Mock(return_value=clock),
    )
    bot = SimpleNamespace(hwnd=1, vision=vision, gamepad=Mock(),
                          distance_reader=SimpleNamespace(backend=object()))
    with patch("main.classify_battle_continuity_screen", return_value=ScreenState.BATTLE), \
         patch("main.ensure_capture_foreground", return_value=True), \
         patch("main.time.sleep"), \
         patch("main.time.monotonic", side_effect=(i * 0.1 for i in range(1000))), \
         patch("main.configure_opening_autopilot") as configure:
        assert not wait_for_battle(bot, timeout=2, loading_already_seen=True)
    configure.assert_not_called()
    assert not bot.gamepad.mock_calls
    assert not getattr(bot, "_opening_autopilot_attempted", False)


@pytest.mark.parametrize("resume", [False, True])
def test_preparation_blocks_initial_and_resumed_control(resume):
    bot = SimpleNamespace(
        hwnd=1, intervention=None, gamepad=Mock(), reset=Mock(),
        vision=SimpleNamespace(grab=Mock(return_value=object()),
                               read_battle_clock_seconds=Mock(return_value=1200)),
        distance_reader=SimpleNamespace(backend=object()),
    )
    with patch("main.ensure_bound_game_foreground", return_value=True), \
         patch("main.normalize_tactical_map_overlay", return_value=True), \
         patch("main.classify_battle_continuity_screen", return_value=ScreenState.BATTLE), \
         patch("main.loading_start_confirmed", return_value=False):
        assert run_battle(bot, resume_existing=resume) == "resume_state"
    bot.reset.assert_not_called()
    assert not bot.gamepad.mock_calls


def test_phase_is_persisted_to_runtime_status(tmp_path):
    from main import observe_battle_phase
    from runtime_control import RuntimeReporter
    import inspect
    # Use the real reporter contract so phase publishing cannot silently break
    # on a missing required message argument.
    reporter = SimpleNamespace(status=SimpleNamespace(state="preparing", message="waiting"))
    reporter.update = Mock(spec=RuntimeReporter.update)
    bot = SimpleNamespace(runtime_reporter=reporter)
    assert observe_battle_phase(bot, None, ScreenState.LOADING) == BattlePhase.LOADING
    args, kwargs = reporter.update.call_args
    inspect.signature(RuntimeReporter.update).bind(reporter, *args, **kwargs)
    assert kwargs["battle_phase"] == "loading"
