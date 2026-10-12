"""Fresh HUD transitions must not inherit an old convergence/fire proof."""

from dataclasses import replace

import pytest

import bot as bot_module
from test_torpedo_hud import analysis, hud, policy


def alignment(error, *, bearing=True):
    return replace(
        hud(aligned=abs(error) <= 8, aim=(500, 305), lead=(500 + error, 305)),
        alignment_error_px=abs(error), bearing_available=bearing,
    )


def cancel_reason(policy):
    results = [call.kwargs for call in policy.bot._record_torpedo_frame.call_args_list
               if len(call.args) > 1 and call.args[1] == "result"]
    assert len(results) == 1
    return results[0]["cancel_reason"]


def test_intervening_improvement_resets_the_consecutive_worsening_count(policy):
    policy.states = [alignment(error) for error in (100, 110, 80, 90, 0, 0)]
    policy.states.append(hud(loaded=False, cooldown=90))

    policy.bot._try_torpedo_attack(analysis(), 1000)

    assert policy.bot.gamepad.aim_relative.call_count == 4
    policy.bot.gamepad.fire_torpedo.assert_called_once()
    assert cancel_reason(policy) is None


def test_two_consecutive_worsening_frames_still_cancel_without_firing(policy):
    policy.states = [alignment(error) for error in (100, 110, 120)]

    policy.bot._try_torpedo_attack(analysis(), 1000)

    assert policy.bot.gamepad.aim_relative.call_count == 2
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    assert cancel_reason(policy) == "alignment_not_converging"


def test_a_plateau_below_the_worsening_threshold_breaks_the_sequence(policy):
    policy.states = [alignment(error) for error in (100, 110, 112, 120, 0, 0)]
    policy.states.append(hud(loaded=False, cooldown=90))

    policy.bot._try_torpedo_attack(analysis(), 1000)

    policy.bot.gamepad.fire_torpedo.assert_called_once()
    assert cancel_reason(policy) is None


def test_orange_frame_clears_the_first_green_fire_proof(policy):
    policy.states = [alignment(0), alignment(0, bearing=False),
                     alignment(0), alignment(0), hud(loaded=False, cooldown=90)]
    firing_after_inspections = []
    normal_fire = policy.bot.gamepad.fire_torpedo.side_effect

    def fire():
        firing_after_inspections.append(policy.inspect.call_count)
        return normal_fire()

    policy.bot.gamepad.fire_torpedo.side_effect = fire
    policy.bot._try_torpedo_attack(analysis(), 1000)

    assert firing_after_inspections == [4]
    assert policy.inspect.call_count == 5
    assert cancel_reason(policy) is None


@pytest.mark.parametrize("bearing,reason", [
    (False, "bearing_unavailable"), (True, "alignment_observation_limit"),
])
def test_twelve_nonfiring_observations_do_not_report_a_valid_target_as_lost(policy, bearing, reason):
    # The axes can share one screen height while their sector directions still
    # differ, or a geometrically aligned bearing can remain orange/unavailable.
    status = alignment(0, bearing=bearing)
    if bearing:
        status = replace(status, alignment_error_degrees=3)
    policy.states = [status]

    policy.bot._try_torpedo_attack(analysis(), 1000)

    assert policy.inspect.call_count == 12
    policy.bot.gamepad.aim_relative.assert_not_called()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    assert cancel_reason(policy) == reason


def test_total_deadline_remains_independent_of_each_fresh_three_second_frame(policy, monkeypatch):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    normal_inspect = policy.inspect.side_effect

    def inspect(*args):
        clock["value"] += 3.1
        return normal_inspect(*args)

    policy.inspect.side_effect = inspect
    policy.states = [alignment(0, bearing=False)]

    policy.bot._try_torpedo_attack(analysis(), 1000)

    assert policy.inspect.call_count == 10
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    assert cancel_reason(policy) == "alignment_timeout"


def test_alignment_capture_propagates_actual_health_rejection(policy):
    reads = {"count": 0}

    def health(*_):
        reads["count"] += 1
        return .8 if reads["count"] <= 2 else None

    policy.bot.vision.read_health_fraction.side_effect = health
    policy.bot._try_torpedo_attack(analysis(), 1000)

    policy.bot.gamepad.torpedo.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    assert cancel_reason(policy) == "alive_health_unconfirmed"
