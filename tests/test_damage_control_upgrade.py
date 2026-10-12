"""Damage-control policy from confirmed hazards and fresh numeric HP loss."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2

from bot import BattleAnalysis, BattleBot
from test_battle_navigation import ConflictingHealthVision, FixtureDistanceReader


def make_bot(**strategy):
    gamepad = SimpleNamespace(
        damage_control=Mock(return_value=None), set_movement=Mock(), stop=Mock(),
    )
    bot = BattleBot(
        1, {"strategy": strategy}, vision=object(), gamepad=gamepad,
        distance_reader=SimpleNamespace(backend=None),
    )
    bot.intervention = SimpleNamespace(poll=Mock(return_value=False), latched=False)
    bot.last_damage_control = 0.0
    return bot


def analysis(health=.9, **flags):
    return BattleAnalysis(
        image=None, width=2560, height=1494, health=health,
        health_recognized=True, **flags,
    )


def execute(bot, snapshot, now):
    # Successful dispatch records its completion time, so both clocks must
    # follow the virtual control-frame timeline used by these policy tests.
    with patch("bot.time.monotonic", return_value=now):
        return bot._execute_survival_consumables(snapshot, now)


def observe(bot, health, now, **flags):
    bot._health_loss_tracker.observe(health, now)
    return execute(bot, analysis(health, **flags), now)


def test_either_confirmed_fire_or_flooding_tries_damage_control_without_waiting_for_loss():
    for hazard in ("on_fire", "flooding"):
        bot = make_bot()
        bot.opening_autopilot_active = True
        execute(bot, analysis(.8, **{hazard: True}), 100)
        bot.gamepad.damage_control.assert_called_once()
        assert bot.last_damage_control == 100
        bot.gamepad.set_movement.assert_not_called()


def test_large_loss_requires_a_second_fresh_low_hp_reading_before_damage_control():
    bot = make_bot()
    observe(bot, 1.0, 100)
    observe(bot, .9, 101)
    bot.gamepad.damage_control.assert_not_called()
    observe(bot, .9, 102)
    bot.gamepad.damage_control.assert_called_once()
    assert bot.last_damage_control == 102
    assert not bot._health_loss_tracker.samples


def test_single_ocr_collapse_followed_by_recovery_never_uses_damage_control():
    bot = make_bot()
    for now, health in ((100, 1.0), (101, .4), (102, 1.0), (103, 1.0)):
        observe(bot, health, now)
    bot.gamepad.damage_control.assert_not_called()
    assert bot.last_damage_control == 0


def test_three_real_decreases_with_stable_frames_and_two_percent_total_loss_trigger():
    bot = make_bot()
    samples = ((100, 1.0), (103, .994), (106, .994), (109, .988), (112, .975))
    for now, health in samples[:-1]:
        observe(bot, health, now)
        bot.gamepad.damage_control.assert_not_called()
    observe(bot, samples[-1][1], samples[-1][0])
    bot.gamepad.damage_control.assert_called_once()


def test_small_repeated_decreases_below_total_damage_threshold_do_not_trigger():
    bot = make_bot()
    for now, health in ((100, 1.0), (103, .998), (106, .996), (109, .994)):
        observe(bot, health, now)
    bot.gamepad.damage_control.assert_not_called()


def test_custom_burst_and_sustained_thresholds_are_used_instead_of_defaults():
    burst_bot = make_bot(damage_control_burst_threshold=.15)
    for now, health in ((100, 1.0), (101, .9), (102, .9)):
        observe(burst_bot, health, now)
    burst_bot.gamepad.damage_control.assert_not_called()
    observe(burst_bot, .8, 103)
    observe(burst_bot, .8, 104)
    burst_bot.gamepad.damage_control.assert_called_once()

    sustained_bot = make_bot(damage_control_sustained_threshold=.04)
    for now, health in ((100, 1.0), (103, .99), (106, .98), (109, .97)):
        observe(sustained_bot, health, now)
    sustained_bot.gamepad.damage_control.assert_not_called()
    observe(sustained_bot, .95, 112)
    sustained_bot.gamepad.damage_control.assert_called_once()


def test_cooldown_blocks_confirmed_hazards_until_the_configured_interval_expires():
    bot = make_bot(damage_control_cooldown_seconds=80)
    bot.last_damage_control = 100
    execute(bot, analysis(on_fire=True), 179.9)
    bot.gamepad.damage_control.assert_not_called()
    execute(bot, analysis(on_fire=True), 180)
    bot.gamepad.damage_control.assert_called_once()
    assert bot.last_damage_control == 180


def test_remaining_at_low_hp_does_not_repeat_damage_control_after_cooldown():
    bot = make_bot()
    execute(bot, analysis(.04, on_fire=True), 100)
    for now in (180, 183, 186, 189, 270):
        observe(bot, .04, now)
    bot.gamepad.damage_control.assert_called_once()
    assert bot.last_damage_control == 100


def test_failed_input_preserves_timer_and_damage_evidence_and_retries_after_backoff():
    for failure in (RuntimeError("input blocked"), OSError("dispatch failed"), False):
        bot = make_bot()
        for now, health in ((100, 1.0), (101, .9), (102, .9)):
            bot._health_loss_tracker.observe(health, now)
        original = list(bot._health_loss_tracker.samples)
        bot.gamepad.damage_control.side_effect = [failure, None]
        assert execute(bot, analysis(), 102) is False
        assert bot.last_damage_control == 0
        assert list(bot._health_loss_tracker.samples) == original
        execute(bot, analysis(), 103)
        assert bot.gamepad.damage_control.call_count == 1
        execute(bot, analysis(), 104)
        assert bot.gamepad.damage_control.call_count == 2
        assert bot.last_damage_control == 104
        assert not bot._health_loss_tracker.samples


def test_pause_clears_loss_history_so_resuming_at_lower_hp_is_not_a_false_burst():
    bot = make_bot()
    for now, health in ((100, 1.0), (101, .9), (102, .9)):
        bot._health_loss_tracker.observe(health, now)
    bot.intervention.poll.return_value = True
    execute(bot, analysis(), 102)
    bot.gamepad.damage_control.assert_not_called()
    assert not bot._health_loss_tracker.samples
    assert bot.last_damage_control == 0

    bot.intervention.poll.return_value = False
    observe(bot, .9, 105)
    observe(bot, .9, 106)
    bot.gamepad.damage_control.assert_not_called()


def test_confirmed_death_blocks_hazards_and_previously_confirmed_damage():
    bot = make_bot()
    for now, health in ((100, 1.0), (101, .9), (102, .9)):
        bot._health_loss_tracker.observe(health, now)
    execute(bot, analysis(0, on_fire=True, flooding=True), 102)
    bot.gamepad.damage_control.assert_not_called()
    assert bot.last_damage_control == 0


def test_analyze_observes_only_fresh_numeric_hp_not_cached_control_ticks():
    class HealthSequenceVision(ConflictingHealthVision):
        def __init__(self, image):
            super().__init__(image)
            self.health = iter((1.0, .9, None))

        def read_health_fraction(self, _image, _backend):
            return next(self.health)

    image = cv2.imread("tests/fixtures/live_battle.png")
    bot = BattleBot(
        1, {"strategy": {}}, vision=HealthSequenceVision(image),
        gamepad=SimpleNamespace(damage_control=Mock(return_value=None)),
        distance_reader=FixtureDistanceReader(),
    )
    tracker = bot._health_loss_tracker
    with patch.object(tracker, "observe", wraps=tracker.observe) as fresh_observe:
        for tick in range(7):
            bot.tick = tick
            bot.analyze()

    assert [item.args[0] for item in fresh_observe.call_args_list] == [1.0, .9, None]
    assert not tracker.samples
    bot.gamepad.damage_control.assert_not_called()
