from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.battle_phase import BattlePhase, BattlePhaseTracker
from core.ui import ScreenState
from main import is_quick_round_already_counted, run_battle, wait_for_battle


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


def test_clock_anchors_match_start_before_control_handoff():
    from main import observe_battle_phase

    bot = SimpleNamespace(
        vision=SimpleNamespace(
            read_battle_clock_seconds=Mock(side_effect=[1200, 1198]),
        ),
        distance_reader=SimpleNamespace(backend=object()),
        _round_control_initialized=False,
        _battle_phase_assumed_active=False,
    )
    with patch("main.time.monotonic", side_effect=[1000.0, 1001.0]):
        assert observe_battle_phase(
            bot, object(), ScreenState.BATTLE
        ) == BattlePhase.PREPARING
        assert observe_battle_phase(
            bot, object(), ScreenState.BATTLE
        ) == BattlePhase.ACTIVE
    # 19:58 at observation time 1001 means the match started at 999, not
    # when a later loading/setup handoff happened to finish.
    assert bot._battle_started_monotonic == pytest.approx(999.0)
    assert bot._last_battle_clock_seconds == 1198


def test_quick_round_double_count_guard_keys_on_battle_identity():
    assert not is_quick_round_already_counted(
        counted=False, counted_start=None, current_start=1000.0
    )
    assert is_quick_round_already_counted(
        counted=True, counted_start=1000.0, current_start=1002.0
    )
    assert not is_quick_round_already_counted(
        counted=True, counted_start=1000.0, current_start=1010.0
    )
    # A re-observed death page after a reset has no start; skip the count.
    assert is_quick_round_already_counted(
        counted=True, counted_start=1000.0, current_start=None
    )
    # A genuinely new battle with no clock may supply its own HUD start later.
    assert not is_quick_round_already_counted(
        counted=True, counted_start=None, current_start=1010.0
    )


def test_quick_battle_uses_clock_derived_start_not_handoff_time():
    class BattleVision:
        @staticmethod
        def grab(_hwnd, *, allow_stale=False):
            return object()

        @staticmethod
        def classify_screen(_image):
            return ScreenState.BATTLE

    class QuickBot:
        hwnd = 1
        vision = BattleVision()
        gamepad = SimpleNamespace()
        intervention = None
        battle_start_time = 100.0
        _battle_started_monotonic = 0.0

        @staticmethod
        def reset(*_args, **_kwargs):
            return None

        @staticmethod
        def combat_tick():
            raise AssertionError("clock-derived five-minute deadline must stop combat ticks")

    with (
        patch("main.ensure_bound_game_foreground", return_value=True),
        patch("main.configure_opening_autopilot", return_value=True),
        patch("main.time.monotonic", return_value=301.0),
    ):
        assert run_battle(QuickBot(), quick_battle=True) == "quick_timeout"


def test_unreadable_clock_uses_first_hud_then_yields_to_clock_anchor():
    from main import observe_battle_phase

    bot = SimpleNamespace(
        vision=SimpleNamespace(
            read_battle_clock_seconds=Mock(side_effect=[None, 1198, 1197]),
        ),
        distance_reader=SimpleNamespace(backend=object()),
        _round_control_initialized=False,
        _battle_phase_assumed_active=True,
    )
    with patch("main.time.monotonic", side_effect=[1000.0, 1005.0, 1006.0]):
        assert observe_battle_phase(
            bot, object(), ScreenState.BATTLE
        ) == BattlePhase.ACTIVE
        assert bot._battle_start_source == "hud"
        assert bot._battle_started_monotonic == pytest.approx(1000.0)
        assert observe_battle_phase(
            bot, object(), ScreenState.BATTLE
        ) == BattlePhase.ACTIVE
        # A single readable clock frame is not enough to replace the HUD
        # fallback; it must agree with the next decrement.
        assert bot._battle_start_source == "hud"
        assert observe_battle_phase(
            bot, object(), ScreenState.BATTLE
        ) == BattlePhase.ACTIVE
        assert bot._battle_start_source == "clock"
        assert bot._battle_started_monotonic == pytest.approx(1003.0)


def test_one_noisy_clock_read_cannot_ratchet_match_start_earlier():
    from main import observe_battle_phase

    bot = SimpleNamespace(
        vision=SimpleNamespace(
            read_battle_clock_seconds=Mock(side_effect=[1200, 1198, 658, 657]),
        ),
        distance_reader=SimpleNamespace(backend=object()),
        _round_control_initialized=False,
        _battle_phase_assumed_active=False,
    )
    with patch("main.time.monotonic", side_effect=[1000.0, 1001.0, 1002.0, 1003.0]):
        for _ in range(4):
            observe_battle_phase(bot, object(), ScreenState.BATTLE)
    assert bot._battle_start_source == "clock"
    assert bot._battle_started_monotonic == pytest.approx(999.0)
