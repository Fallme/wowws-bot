"""Six-kilometre opportunities must pass both initial and fresh distance gates."""
import pytest

from test_torpedo_hud import analysis, policy


def at_distance(policy, distance):
    snapshot = analysis()
    snapshot.target_distance_km = distance
    snapshot.minimap_distance_km = distance
    policy.observation.value_km = distance
    return snapshot


@pytest.mark.parametrize("distance", [3.01, 5., 5.9, 6.])
@pytest.mark.parametrize("custom", [False, True])
def test_default_attempts_and_confirms_launch_through_six_km(policy, distance, custom):
    if custom:
        policy.bot.ship = {"has_torpedoes": False, "type": "CUSTOM"}
    snapshot = at_distance(policy, distance)
    policy.bot._try_torpedo_attack(snapshot, 1000.)
    policy.bot.gamepad.fire_torpedo.assert_called_once()
    assert policy.detect.call_count == 2
    assert policy.bot.distance_reader.read.call_count >= 4
    assert [call.args[0] for call in policy.bot.events.publish.call_args_list] == [
        "torpedo.launch_attempt", "torpedo.launch_confirmed"]


@pytest.mark.parametrize("configured", [None, 6., 8.])
def test_over_six_km_cannot_select_or_fire_even_with_larger_config(policy, configured):
    policy.bot.ship = {"has_torpedoes": False, "type": "CUSTOM"}
    if configured is not None:
        policy.bot.strategy["torpedo_close_distance_km"] = configured
    policy.bot._try_torpedo_attack(at_distance(policy, 6.01), 1000.)
    assert policy.calls == []
    policy.detect.assert_not_called()


@pytest.mark.parametrize("distance,allowed", [(5.5, True), (5.51, False)])
def test_shorter_known_weapon_range_still_limits_attempts(policy, distance, allowed):
    policy.bot.ship["torpedo"]["range"] = 5.5
    policy.bot._try_torpedo_attack(at_distance(policy, distance), 1000.)
    assert policy.bot.gamepad.fire_torpedo.call_count == int(allowed)
    if not allowed:
        assert policy.calls == []


@pytest.mark.parametrize("fresh_distance,allowed", [(6., True), (6.01, False)])
def test_alignment_rechecks_the_six_km_boundary_in_fresh_frames(policy, fresh_distance, allowed):
    from types import SimpleNamespace
    snapshot = at_distance(policy, 5.8)
    updated = SimpleNamespace(accepted=True, value_km=fresh_distance, confidence=.95)
    policy.bot.distance_reader.read.side_effect = [
        policy.observation, policy.observation, updated, updated]
    policy.bot._try_torpedo_attack(snapshot, 1000.)
    policy.bot.gamepad.torpedo.assert_called_once()
    assert policy.bot.gamepad.fire_torpedo.call_count == int(allowed)
    if not allowed:
        policy.bot.gamepad.main_gun.assert_called_once()
        assert policy.bot._last_torpedo_release == 0


def test_explicit_smaller_attempt_distance_is_respected(policy):
    policy.bot.strategy["torpedo_close_distance_km"] = 4.
    policy.bot._try_torpedo_attack(at_distance(policy, 4.01), 1000.)
    assert policy.calls == []
