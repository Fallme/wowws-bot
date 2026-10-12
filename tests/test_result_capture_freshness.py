"""Settlement decisions must use current captures instead of battle-loop caches."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.frame_guard import CaptureFault
from core.results import BattleRewards
from core.ui import ScreenState
from main import collect_battle_rewards


BATTLE_PIXEL = 21
RESULT_PIXEL = 92


def frame(marker):
    return np.full((90, 160, 3), marker, np.uint8)


def is_battle(image):
    return int(image[0, 0, 0]) == BATTLE_PIXEL


def reward_reader():
    return SimpleNamespace(
        backend=SimpleNamespace(execution_provider="CPUExecutionProvider"),
        MINIMUM_CREDITS=1_000,
        _looks_like_port_reward_card=Mock(return_value=False),
        read=Mock(return_value=BattleRewards(
            credits=102_692, ship_xp=1_143, free_xp=136,
            recognized=True, provider="CPUExecutionProvider", outcome="victory",
        )),
    )


def make_bot(cached_image, current_frames):
    return SimpleNamespace(
        hwnd=1,
        distance_ocr_service=None,
        last_analysis=SimpleNamespace(image=cached_image),
        vision=SimpleNamespace(
            grab=Mock(side_effect=current_frames),
            classify_screen=Mock(side_effect=lambda image: ScreenState.BATTLE if is_battle(image) else ScreenState.RESULTS),
            _has_battle_hud=Mock(side_effect=is_battle),
        ),
    )


def capture_interlocks(*, paused=False, foreground=True):
    stack = ExitStack()
    stack.enter_context(patch("main.operation_paused", side_effect=paused if callable(paused) else None,
                              return_value=paused if not callable(paused) else False))
    stack.enter_context(patch("main.ensure_capture_foreground", return_value=foreground))
    stack.enter_context(patch("main.time.sleep"))
    return stack


def test_cached_final_battle_frame_does_not_hide_three_current_result_captures():
    cached = frame(BATTLE_PIXEL)
    current = [frame(RESULT_PIXEL) for _ in range(3)]
    bot = make_bot(cached, current)
    reader = reward_reader()

    with capture_interlocks():
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert confirmed
    assert state == ScreenState.RESULTS
    assert rewards.recognized
    assert rewards.resource_values() == {"credits": 102_692, "ship_xp": 1_143, "free_xp": 136}
    assert rewards.outcome == "victory"
    assert bot.vision.grab.call_count == 3
    assert [call.args[0] is image for call, image in zip(reader.read.call_args_list, current)] == [True] * 3
    assert all(call.args[0] is not cached for call in bot.vision.classify_screen.call_args_list)


def test_cached_result_frame_never_counts_or_runs_ocr_when_current_scene_is_battle():
    cached = frame(RESULT_PIXEL)
    live_battle = frame(BATTLE_PIXEL)
    bot = make_bot(cached, [live_battle])
    reader = reward_reader()

    with capture_interlocks():
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.BATTLE
    bot.vision.grab.assert_called_once()
    reader.read.assert_not_called()


def test_fresh_battle_hud_blocks_ocr_even_when_scene_colours_look_like_results():
    bot = make_bot(frame(RESULT_PIXEL), [frame(BATTLE_PIXEL)])
    bot.vision.classify_screen.side_effect = None
    bot.vision.classify_screen.return_value = ScreenState.RESULTS
    reader = reward_reader()

    with capture_interlocks():
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.BATTLE
    reader.read.assert_not_called()


def test_capture_fault_never_reads_the_cached_result_as_a_fallback():
    bot = make_bot(frame(RESULT_PIXEL), CaptureFault("capture unavailable"))
    reader = reward_reader()

    with capture_interlocks():
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.UNKNOWN
    assert bot.vision.grab.call_count == 3
    bot.vision.classify_screen.assert_not_called()
    reader.read.assert_not_called()


def test_current_result_captures_can_recover_after_a_capture_fault_without_using_battle_cache():
    cached = frame(BATTLE_PIXEL)
    current = [frame(RESULT_PIXEL) for _ in range(3)]
    bot = make_bot(cached, [CaptureFault("temporary capture failure"), *current])
    reader = reward_reader()

    with capture_interlocks():
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=4)

    assert confirmed
    assert rewards.recognized
    assert state == ScreenState.RESULTS
    assert bot.vision.grab.call_count == 4
    assert reader.read.call_count == 3
    assert all(call.args[0] is not cached for call in reader.read.call_args_list)


def test_pause_before_capture_never_classifies_or_reads_cached_results():
    bot = make_bot(frame(RESULT_PIXEL), [frame(RESULT_PIXEL)])
    reader = reward_reader()

    with capture_interlocks(paused=True):
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.UNKNOWN
    bot.vision.grab.assert_not_called()
    bot.vision.classify_screen.assert_not_called()
    reader.read.assert_not_called()


def test_pause_after_capture_fault_does_not_fall_back_to_cached_results():
    paused = [False]
    bot = make_bot(frame(RESULT_PIXEL), [])
    reader = reward_reader()

    def failed_capture(*_args, **_kwargs):
        paused[0] = True
        raise CaptureFault("window became unavailable while user took over")

    bot.vision.grab.side_effect = failed_capture
    with capture_interlocks(paused=lambda _bot: paused[0]):
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.UNKNOWN
    bot.vision.grab.assert_called_once()
    reader.read.assert_not_called()


def test_missing_foreground_never_uses_cached_results():
    bot = make_bot(frame(RESULT_PIXEL), [frame(RESULT_PIXEL)])
    reader = reward_reader()

    with capture_interlocks(foreground=False):
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=3)

    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.UNKNOWN
    bot.vision.grab.assert_not_called()
    reader.read.assert_not_called()


@pytest.mark.parametrize("attempts,recognized", [(2, False), (3, True)])
def test_unchanged_result_pixels_still_require_each_capture_and_numeric_consensus(attempts, recognized):
    # allow_stale means an unchanged menu can be captured repeatedly; it must
    # never mean reusing last_analysis.image without invoking the capture API.
    current = frame(RESULT_PIXEL)
    bot = make_bot(frame(BATTLE_PIXEL), [current] * attempts)
    reader = reward_reader()

    with capture_interlocks():
        confirmed, rewards, state = collect_battle_rewards(bot, reader, attempts=attempts)

    assert confirmed
    assert state == ScreenState.RESULTS
    assert rewards.recognized is recognized
    assert bot.vision.grab.call_count == attempts
    assert reader.read.call_count == attempts
    assert all(call.kwargs.get("allow_stale") is True for call in bot.vision.grab.call_args_list)
