from types import SimpleNamespace

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
