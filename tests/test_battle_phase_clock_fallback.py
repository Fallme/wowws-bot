"""Fallbacks for machines where the battle clock OCR is unreadable."""

from types import SimpleNamespace

import numpy as np

import main
from core.battle_phase import BattlePhase, BattlePhaseTracker
from core.ui import ScreenState


def _phase_bot(*, assumed=False):
    return SimpleNamespace(
        vision=SimpleNamespace(read_battle_clock_seconds=lambda *_: None),
        distance_reader=SimpleNamespace(backend=object()),
        _battle_phase_assumed_active=assumed,
        _battle_phase_tracker=BattlePhaseTracker(),
    )


def test_assumed_active_overrides_missing_clock_and_resets_off_battle():
    bot = _phase_bot(assumed=True)
    frame = np.zeros((90, 160, 3), np.uint8)
    assert main.observe_battle_phase(
        bot, frame, ScreenState.BATTLE
    ) == BattlePhase.ACTIVE
    assert main.observe_battle_phase(
        bot, frame, ScreenState.LOADING
    ) == BattlePhase.LOADING
    assert bot._battle_phase_assumed_active is False


def test_missing_clock_logs_ocr_duration_with_throttle(monkeypatch, caplog):
    bot = _phase_bot()
    frame = np.zeros((90, 160, 3), np.uint8)
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)
    with caplog.at_level("INFO", logger=main.logger.name):
        main.observe_battle_phase(bot, frame, ScreenState.BATTLE)
        main.observe_battle_phase(bot, frame, ScreenState.BATTLE)
    messages = [record.message for record in caplog.records
                if "对局计时 OCR 未识别，耗时" in record.message]
    assert len(messages) == 1
    assert "ms" in messages[0]


def _fallback_bot(*, pose=True, health=1.0):
    return SimpleNamespace(
        vision=SimpleNamespace(
            find_minimap=lambda image: np.zeros((60, 60, 3), np.uint8),
            find_player_pose_on_minimap=lambda minimap: object() if pose else None,
            read_health_fraction=lambda *_: health,
        ),
        distance_reader=SimpleNamespace(backend=object()),
    )


def test_clock_fallback_requires_live_player_evidence(monkeypatch):
    monkeypatch.setattr(main, "recovery_escape_hint", lambda *args: None)
    frame = np.zeros((90, 160, 3), np.uint8)
    assert main.battle_phase_clock_fallback_ready(
        _fallback_bot(), frame
    ) == (True, "")
    assert main.battle_phase_clock_fallback_ready(
        _fallback_bot(pose=False, health=None), frame
    ) == (False, "player_marker_unavailable")
    assert main.battle_phase_clock_fallback_ready(
        _fallback_bot(pose=False, health=1.0), frame
    ) == (True, "")
    assert main.battle_phase_clock_fallback_ready(
        _fallback_bot(health=0.0), frame
    ) == (False, "ship_dead")


def test_clock_fallback_rejects_corner_escape_overlay(monkeypatch):
    monkeypatch.setattr(main, "recovery_escape_hint", lambda *args: (10, 20))
    assert main.battle_phase_clock_fallback_ready(
        _fallback_bot(), np.zeros((90, 160, 3), np.uint8)
    ) == (False, "corner_escape_hint")


def test_wait_for_battle_uses_fallback_instead_of_waiting_180_seconds(
    monkeypatch,
):
    frame = np.zeros((90, 160, 3), np.uint8)
    bot = SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(
            grab=lambda *args, **kwargs: frame,
            read_battle_clock_seconds=lambda *_: None,
        ),
        distance_reader=SimpleNamespace(backend=object()),
        gamepad=SimpleNamespace(),
        intervention=None,
        strategy={},
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(main.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        main.time, "sleep", lambda seconds: clock.__setitem__(
            "now", clock["now"] + float(seconds)
        )
    )
    monkeypatch.setattr(main, "ensure_capture_foreground", lambda bot: True)
    monkeypatch.setattr(
        main, "classify_battle_continuity_screen",
        lambda *args: ScreenState.BATTLE,
    )
    monkeypatch.setattr(main, "loading_start_confirmed", lambda *args: False)
    monkeypatch.setattr(
        main, "battle_phase_clock_fallback_ready", lambda *args: (True, "")
    )

    assert main.wait_for_battle(
        bot, timeout=60.0, loading_already_seen=True
    )
    assert clock["now"] >= 45.0
    assert bot._battle_phase_assumed_active is True


def test_autopilot_clock_timeout_clears_waiting_flag(monkeypatch):
    frame = np.zeros((90, 160, 3), np.uint8)
    bot = SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(
            grab=lambda *args, **kwargs: frame,
            read_battle_clock_seconds=lambda *_: 1200,
        ),
        distance_reader=SimpleNamespace(backend=object()),
        gamepad=SimpleNamespace(toggle_tactical_map=lambda: None),
        intervention=None,
        enable_opening_autopilot=lambda *args, **kwargs: None,
        _opening_autopilot_retry_pending=False,
        _tactical_map_attempted_this_battle=False,
    )
    monkeypatch.setattr(main, "operation_paused", lambda bot: False)
    monkeypatch.setattr(main, "normalize_tactical_map_overlay", lambda bot: True)
    monkeypatch.setattr(
        main, "classify_battle_continuity_screen",
        lambda *args: ScreenState.BATTLE,
    )
    monkeypatch.setattr(
        main, "observe_battle_phase",
        lambda *args: main.BattlePhase.PREPARING,
    )
    times = iter(1000.0 + index * 0.5 for index in range(400))
    monkeypatch.setattr(main.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(main.time, "sleep", lambda _: None)

    assert main.configure_opening_autopilot(bot) is False
    assert bot._autopilot_waiting_for_start is False
    assert bot._opening_autopilot_attempted is False
