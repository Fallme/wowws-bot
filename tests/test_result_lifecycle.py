from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.results import BattleRewards
from core.ui import ScreenState
from main import collect_battle_rewards, return_to_port


class SequenceVision:
    def __init__(self, states):
        self.states = iter(states)
        self.current = ScreenState.UNKNOWN

    def grab(self, _hwnd, *, allow_stale=False):
        return np.full((90, 160, 3), 80, dtype=np.uint8)

    def classify_screen(self, _image):
        self.current = next(self.states)
        return self.current


class FixedRewardReader:
    backend = SimpleNamespace(execution_provider="CUDAExecutionProvider")

    def read(self, _image):
        return BattleRewards(
            credits=102_692,
            ship_xp=1_143,
            free_xp=136,
            recognized=True,
            provider="CUDAExecutionProvider",
        )


def make_bot(states):
    image = np.full((90, 160, 3), 80, dtype=np.uint8)
    return SimpleNamespace(
        hwnd=1,
        vision=SequenceVision(states),
        last_analysis=SimpleNamespace(image=image),
        distance_ocr_service=None,
    )


def test_round_commits_only_after_consecutive_result_frames():
    confirmed, rewards, state = collect_battle_rewards(
        make_bot([ScreenState.RESULTS, ScreenState.RESULTS]),
        FixedRewardReader(),
        attempts=2,
    )

    assert confirmed
    assert state == ScreenState.RESULTS
    assert rewards.ship_xp == 1_143


def test_confirmed_defeat_survives_final_unknown_ocr_frame():
    from dataclasses import replace
    reader = FixedRewardReader()
    reward = reader.read(None)
    outcomes = iter(["defeat", "defeat", "unknown"])
    reader.read = lambda _: replace(reward, outcome=next(outcomes))
    with patch("main.time.sleep"):
        confirmed, rewards, state = collect_battle_rewards(
            make_bot([ScreenState.RESULTS] * 3), reader, attempts=3,
        )
    assert confirmed
    assert rewards.outcome == "defeat"


def test_one_defeat_reading_is_not_enough_to_count_a_loss():
    from dataclasses import replace
    reader = FixedRewardReader()
    reward = reader.read(None)
    outcomes = iter(["unknown", "defeat", "unknown"])
    reader.read = lambda _: replace(reward, outcome=next(outcomes))
    with patch("main.time.sleep"):
        confirmed, rewards, _ = collect_battle_rewards(
            make_bot([ScreenState.RESULTS] * 3), reader, attempts=3,
        )
    assert confirmed
    assert rewards.outcome == "unknown"


def test_single_false_result_frame_does_not_commit_round():
    confirmed, _rewards, state = collect_battle_rewards(
        make_bot([ScreenState.RESULTS, ScreenState.BATTLE]),
        FixedRewardReader(),
        attempts=2,
    )

    assert not confirmed
    assert state == ScreenState.BATTLE


def test_live_battle_hud_overrides_false_result_colours():
    bot = make_bot([ScreenState.RESULTS, ScreenState.RESULTS])
    bot.vision._has_battle_hud = lambda _image: True

    confirmed, rewards, state = collect_battle_rewards(
        bot,
        FixedRewardReader(),
        attempts=2,
    )

    assert not confirmed
    assert not rewards.recognized
    assert state == ScreenState.BATTLE


def test_return_to_port_never_sends_escape_while_battle_is_live():
    class Gamepad:
        escapes = 0

        def escape(self):
            self.escapes += 1

    bot = make_bot([ScreenState.BATTLE])
    bot.last_analysis = None
    bot.gamepad = Gamepad()

    assert not return_to_port(bot, attempts=1)
    assert bot.gamepad.escapes == 0


def test_reward_consensus_accepts_columns_confirmed_on_different_frames():
    class StaggeredReader:
        backend = SimpleNamespace(execution_provider="CUDAExecutionProvider")
        MINIMUM_CREDITS = 1_000

        def __init__(self):
            self.values = iter(
                [
                    BattleRewards(credits=102_692, ship_xp=0, free_xp=0),
                    BattleRewards(credits=102_692, ship_xp=1_143, free_xp=136),
                    BattleRewards(credits=102_692, ship_xp=1_143, free_xp=136),
                ]
            )

        def read(self, _image):
            return next(self.values)

    confirmed, rewards, state = collect_battle_rewards(
        make_bot([ScreenState.RESULTS, ScreenState.RESULTS, ScreenState.RESULTS]),
        StaggeredReader(),
        attempts=3,
    )

    assert confirmed
    assert state == ScreenState.RESULTS
    assert rewards.resource_values() == {
        "credits": 102_692,
        "ship_xp": 1_143,
        "free_xp": 136,
    }


def test_reward_ocr_failure_does_not_abort_confirmed_result_lifecycle():
    class BrokenReader:
        backend = SimpleNamespace(execution_provider="CPUExecutionProvider")

        @staticmethod
        def read(_image):
            raise RuntimeError("OCR provider unavailable")

    confirmed, rewards, state = collect_battle_rewards(
        make_bot([ScreenState.RESULTS, ScreenState.RESULTS]),
        BrokenReader(),
        attempts=2,
    )

    assert confirmed
    assert not rewards.recognized
    assert state == ScreenState.RESULTS


def test_return_to_port_escapes_leftover_battle_type_selector():
    """A residual battle-type selector is classified UNKNOWN and would spin
    forever; return_to_port must close it with Esc and re-check the scene."""
    image = np.full((90, 160, 3), 80, dtype=np.uint8)

    class Gamepad:
        escapes = 0

        def escape(self):
            self.escapes += 1

    bot = SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(grab=lambda *_a, **_k: image),
        last_analysis=None,
        gamepad=Gamepad(),
        distance_reader=SimpleNamespace(backend=Mock()),
    )

    with (
        patch("main.ensure_capture_foreground", return_value=True),
        patch("main.classify_runtime_screen", side_effect=[ScreenState.UNKNOWN, ScreenState.PORT]),
        patch("main.operation_paused", return_value=False),
        patch("main.in_battle_type_selector", return_value=True),
        patch("main.time.sleep", return_value=None),
    ):
        assert return_to_port(bot, attempts=2)

    assert bot.gamepad.escapes == 1


def test_return_to_port_tries_escape_three_times_for_unrecognized_pages():
    image = np.full((90, 160, 3), 80, dtype=np.uint8)

    class Gamepad:
        escapes = 0

        def escape(self):
            self.escapes += 1

    bot = SimpleNamespace(
        hwnd=1,
        vision=SimpleNamespace(grab=lambda *_a, **_k: image),
        last_analysis=None,
        gamepad=Gamepad(),
        distance_reader=SimpleNamespace(backend=Mock()),
        intervention=None,
    )

    with (
        patch("main.ensure_capture_foreground", return_value=True),
        patch(
            "main.classify_runtime_screen",
            side_effect=[
                ScreenState.UNKNOWN,
                ScreenState.UNKNOWN,
                ScreenState.UNKNOWN,
                ScreenState.PORT,
            ],
        ),
        patch("main.operation_paused", return_value=False),
        patch("main.in_battle_type_selector", return_value=False),
        patch("main.time.sleep", return_value=None),
    ):
        assert return_to_port(bot, attempts=4)

    assert bot.gamepad.escapes == 3


def test_same_battle_continuation_rebuilds_closed_distance_ocr():
    """An unconfirmed result page can send the loop back into the same
    battle; collect_battle_rewards already closed the async OCR service and
    the continuation skips bot.reset(), so rebuild_distance_ocr must return
    a service whose executor still accepts submissions."""
    import bot as bot_module
    from config_loader import load_ship_config

    with (
        patch.object(bot_module, "create_input_controller", return_value=Mock()),
        patch.object(bot_module, "Vision", return_value=Mock()),
        patch.object(bot_module, "TargetDistanceReader", return_value=Mock()),
    ):
        bot = bot_module.BattleBot(hwnd=1, ship_config=load_ship_config("napoli"))
    service = bot.distance_ocr_service
    service.close()
    bot.rebuild_distance_ocr()
    assert bot.distance_ocr_service is not service
    submitted = bot.distance_ocr_service.submit(
        np.full((10, 10, 3), 80, np.uint8),
        [((5, 5), "probe")],
        captured_at=0.0,
    )
    assert submitted
    bot.distance_ocr_service.close()
    bot.rebuild_distance_ocr()  # repeated close/rebuild must stay safe
    bot.distance_ocr_service.close()
