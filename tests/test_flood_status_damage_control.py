from types import SimpleNamespace
from unittest.mock import Mock

import cv2

from bot import BattleBot
from core.vision import Vision
from test_battle_navigation import ConflictingHealthVision, FixtureDistanceReader


class VerifiedFloodVision(ConflictingHealthVision):
    # Keep the stock combined detector identity. Only its stronger, verified
    # timer hook is replaced here, so the legacy adapter path stays covered
    # by test_battle_navigation.
    def flooding_status_icon_visible(self, image):
        return True


def make_bot():
    image = cv2.imread("tests/fixtures/live_battle.png")
    bot = BattleBot(1, {"strategy": {}}, vision=VerifiedFloodVision(image),
                    gamepad=SimpleNamespace(damage_control=Mock()),
                    distance_reader=FixtureDistanceReader())
    bot.intervention = SimpleNamespace(poll=lambda *_: False, latched=False)
    return bot


def test_verified_flood_timer_uses_two_frames_before_hp_ticks_down():
    bot = make_bot()
    assert type(bot.vision).is_flooding is Vision.is_flooding
    first = bot.analyze()
    bot.tick = 1
    second = bot.analyze()
    assert first.health == second.health == 1.0
    assert not first.flooding
    assert second.flooding
    bot.last_damage_control = -100.0
    bot._execute_survival_consumables(second, 100.0)
    bot.gamepad.damage_control.assert_called_once()


def test_pause_does_not_reuse_pre_pause_hazard_confirmation():
    bot = make_bot()
    bot.analyze()
    bot.mark_manual_pause()
    bot.tick = 1
    assert not bot.analyze().flooding
    bot.tick = 2
    assert bot.analyze().flooding
