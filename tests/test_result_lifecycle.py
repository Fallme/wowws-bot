from types import SimpleNamespace

import numpy as np

from core.results import BattleRewards
from core.ui import ScreenState
from main import collect_battle_rewards


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
