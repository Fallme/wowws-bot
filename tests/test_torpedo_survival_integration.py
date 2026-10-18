"""Real survival policy stays active inside the multi-frame torpedo loop."""

from unittest.mock import Mock

import pytest

import bot as bot_module
from test_damage_control_upgrade import make_bot
from test_torpedo_hud import analysis, policy


def connect_real_survival(policy):
    instance = make_bot()
    # Reuse only the no-input replay adapters. Keep the real constructor's
    # health tracker/hazard history and the real bound survival methods.
    instance.__dict__.update({name: value for name, value in vars(policy.bot).items()
                              if name != "_check_survival_frame"})
    instance.vision.is_on_fire = Mock(return_value=False)
    instance.vision.is_flooding = Mock(return_value=False)

    def damage_control():
        policy.calls.append(("damage_control", ()))
        return True

    instance.gamepad.damage_control = Mock(side_effect=damage_control)
    policy.bot = instance
    return instance


def test_fresh_hp_loss_dispatches_r_before_the_single_aligned_left_click(policy, monkeypatch):
    instance = connect_real_survival(policy)
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    normal_grab = instance.vision.grab.side_effect

    def grab(*args, **kwargs):
        clock["value"] += 1
        return normal_grab(*args, **kwargs)

    health = iter((1.0, .9, .9))
    instance.vision.grab.side_effect = grab
    instance.vision.read_health_fraction.side_effect = lambda *_: next(health, .9)

    instance._try_torpedo_attack(analysis(), 1000)

    instance.gamepad.damage_control.assert_called_once()
    instance.gamepad.fire_torpedo.assert_called_once()
    assert [name for name, _ in policy.calls] == [
        "torpedo", "damage_control", "fire_torpedo", "main_gun",
    ]
    assert instance.last_damage_control == 1003
    assert instance._last_torpedo_release == 1004
    assert instance._cached_health == .9


def test_large_hp_drop_png_is_deferred_until_r_and_fresh_aligned_fire_can_run(policy, monkeypatch, tmp_path):
    instance = connect_real_survival(policy)
    instance.blockage_diagnostic_root = tmp_path
    instance.last_analysis = analysis()
    instance._cached_health = 1.0
    instance._health_ocr_valid = True
    instance._health_observed_at = 1000
    clock = {"value": 1000.0}
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock["value"])
    normal_grab = instance.vision.grab.side_effect

    def grab(*args, **kwargs):
        clock["value"] += 1
        return normal_grab(*args, **kwargs)

    # A second confirmed lower read can immediately try R while the first
    # large drop also produces an expensive pair of diagnostic PNGs.
    instance._health_loss_tracker.observe(1.0, 999)
    instance._health_loss_tracker.observe(.9, 1000)
    instance.vision.grab.side_effect = grab
    instance.vision.read_health_fraction.return_value = .9
    written_at = []

    def slow_write(_evidence):
        written_at.append(clock["value"])
        assert instance.gamepad.damage_control.call_count == 1
        clock["value"] += 5

    monkeypatch.setattr(bot_module.BattleBot, "_write_health_transition_evidence",
                        lambda _instance, evidence: slow_write(evidence))
    normal_fire = instance.gamepad.fire_torpedo.side_effect

    def fire():
        assert clock["value"] - instance._torpedo_frame_captured_at <= 4
        return normal_fire()

    instance.gamepad.fire_torpedo.side_effect = fire
    instance._try_torpedo_attack(analysis(), 1000)

    instance.gamepad.damage_control.assert_called_once()
    instance.gamepad.fire_torpedo.assert_called_once()
    assert [name for name, _ in policy.calls] == [
        "damage_control", "torpedo", "fire_torpedo", "main_gun",
    ]
    assert written_at
    assert not instance._pending_health_evidence


@pytest.mark.parametrize("gate", ["paused", "dead"])
def test_real_survival_connection_keeps_takeover_and_death_free_of_inputs(policy, gate):
    instance = connect_real_survival(policy)
    instance._health_loss_tracker.observe(1.0, 998)
    instance._health_loss_tracker.observe(.8, 999)
    if gate == "paused":
        policy.paused["value"] = True
    else:
        instance.vision.read_health_fraction.return_value = 0

    instance._try_torpedo_attack(analysis(), 1000)

    instance.gamepad.damage_control.assert_not_called()
    assert policy.calls == []
