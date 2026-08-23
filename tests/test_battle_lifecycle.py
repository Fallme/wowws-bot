import time

import pytest

from bot import BattleAnalysis, BattleBot
from core.feedback import SafetyFault


class StopRecorder:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def make_lifecycle_bot(analysis):
    bot = BattleBot.__new__(BattleBot)
    bot.analyze = lambda: analysis
    bot.last_analysis = None
    bot.gamepad = StopRecorder()
    bot._unknown_since = None
    bot._post_battle_grace_seconds = 45.0
    bot._finish_tick = lambda current: None
    return bot


def test_missing_hud_waits_safely_for_results_transition():
    analysis = BattleAnalysis(image=None, width=0, height=0)
    bot = make_lifecycle_bot(analysis)

    assert bot.combat_tick() == "waiting"
    assert bot.gamepad.stop_calls == 1
    assert bot._unknown_since is not None


def test_missing_hud_still_faults_after_transition_grace():
    analysis = BattleAnalysis(image=None, width=0, height=0)
    bot = make_lifecycle_bot(analysis)
    bot._unknown_since = time.monotonic() - 46.0

    with pytest.raises(SafetyFault, match="无法确认战斗 HUD"):
        bot.combat_tick()
