"""Input decisions use fresh HUD evidence before optional PNG encoding."""

from unittest.mock import Mock

import bot as bot_module
from test_torpedo_hud import analysis, hud, policy


def test_slow_png_encoding_cannot_consume_the_fresh_aligned_fire_window(policy, monkeypatch):
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    normal_inspect = policy.inspect.side_effect

    def inspect(*args):
        clock["value"] += 3.1
        return normal_inspect(*args)

    def save(*_args, **_kwargs):
        clock["value"] += 5

    policy.inspect.side_effect = inspect
    policy.bot._record_torpedo_frame.side_effect = save
    normal_fire = policy.bot.gamepad.fire_torpedo.side_effect
    fired_at = []

    def fire():
        fired_at.append(clock["value"])
        # Earlier observations may already be encoded; the current firing
        # frame must still be fresh and absent from the synchronous writer.
        assert clock["value"] - policy.bot._torpedo_frame_captured_at <= 4
        assert not any(call.args[0] is policy.frames[-1]
                       for call in policy.bot._record_torpedo_frame.call_args_list)
        return normal_fire()

    policy.bot.gamepad.fire_torpedo.side_effect = fire
    policy.bot._try_torpedo_attack(analysis(), 1000)

    policy.bot.gamepad.fire_torpedo.assert_called_once()
    assert len(fired_at) == 1
    assert 1016.19 <= fired_at[0] <= 1016.21
    assert any(call.kwargs.get("firing") for call in policy.bot._record_torpedo_frame.call_args_list)
    assert any(call.kwargs.get("confirmed") for call in policy.bot._record_torpedo_frame.call_args_list)


def test_slot_three_transition_gets_fresh_observation_before_being_rejected(policy):
    policy.states = [hud(weapon=None, selected=False, loaded=False),
                     hud(), hud(), hud(loaded=False, cooldown=90)]

    policy.bot._try_torpedo_attack(analysis(), 1000)

    policy.bot.gamepad.fire_torpedo.assert_called_once()
    assert policy.inspect.call_count == 4


def test_unknown_selected_weapon_still_stops_after_three_observations(policy):
    policy.states = [hud(weapon=None, selected=False, loaded=False)]

    policy.bot._try_torpedo_attack(analysis(), 1000)

    assert policy.inspect.call_count == 3
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.aim_relative.assert_not_called()


def test_explicit_reload_does_not_spend_a_selection_retry(policy):
    policy.states = [hud(cooldown=30)]

    policy.bot._try_torpedo_attack(analysis(), 1000)

    assert policy.inspect.call_count == 1
    policy.bot.gamepad.fire_torpedo.assert_not_called()


def test_one_target_lock_can_wait_for_prediction_then_dispatch_single_fire(policy):
    policy.states = [hud(lead=None), hud(), hud(), hud(loaded=False, cooldown=90)]

    policy.bot._try_torpedo_attack(analysis(), 1000)

    policy.bot.gamepad.lock.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_called_once()
    assert policy.inspect.call_count == 4


def test_survival_input_guard_can_cancel_torpedo_preparation(policy):
    policy.bot._check_survival_frame = Mock(return_value=False)

    policy.bot._try_torpedo_attack(analysis(), 1000)

    policy.bot._check_survival_frame.assert_called_once_with(
        policy.frames[0], .8, observed_at=1000.0)
    policy.bot.gamepad.torpedo.assert_not_called()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    reasons = [call.kwargs["reason"] for call in policy.bot.events.publish.call_args_list
               if call.args[0] == "torpedo.preparation_skipped"]
    assert reasons == ["survival_input_cancelled"]


def test_takeover_while_prior_frame_is_encoded_stops_following_fire(policy):
    checked_frames = set()

    def survival(frame, _health, **_kwargs):
        checked_frames.add(id(frame))
        return True

    def save(frame, phase, **_kwargs):
        assert id(frame) in checked_frames
        if phase == "observation":
            policy.paused["value"] = True

    policy.bot._check_survival_frame.side_effect = survival
    policy.bot._record_torpedo_frame.side_effect = save
    policy.bot._try_torpedo_attack(analysis(), 1000)

    policy.bot.gamepad.torpedo.assert_called_once()
    policy.bot.gamepad.fire_torpedo.assert_not_called()
    policy.bot.gamepad.main_gun.assert_not_called()
    # The encoded image must not be counted as another independent HP read.
    assert policy.bot._check_survival_frame.call_count == 3
