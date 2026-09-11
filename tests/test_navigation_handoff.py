import math
from types import SimpleNamespace

import pytest

from bot import BattleAnalysis, BattleBot
from strategy.helm_model import YawMotion
from strategy.native_navigation import NativeRouteWatchdog
from core.vision import Vision, PlayerPose


def test_stall_requires_continuous_low_speed_and_stationary_fresh_pose():
    watchdog = NativeRouteWatchdog()
    for t in range(20):
        assert not watchdog.stalled(t, (.4, .4), 0)
    assert watchdog.stalled(20, (.4, .4), 0)
    assert not watchdog.stalled(21, (.41, .4), 0)
    assert not watchdog.stalled(22, (.41, .4), None)
    assert not watchdog.stalled(23, (.41, .4), 25)


@pytest.mark.parametrize("kwargs", [{"cached": True}, {"grace": True}])
def test_unreliable_or_startup_observations_cannot_trigger_takeover(kwargs):
    watchdog = NativeRouteWatchdog()
    for t in range(60):
        assert not watchdog.stalled(t, (.4, .4), 0, **kwargs)


def test_pause_or_capture_gap_does_not_count_toward_stall():
    watchdog = NativeRouteWatchdog()
    assert not watchdog.stalled(0, (.4, .4), 0)
    assert not watchdog.stalled(100, (.4, .4), 0)


def test_yaw_wraparound_jump_rejection_and_expiry():
    yaw = YawMotion()
    heading = lambda d: (math.cos(math.radians(d)), math.sin(math.radians(d)))
    yaw.observe(heading(179), 0)
    yaw.observe(heading(-179), 1)
    assert yaw.fresh_rate(1) == pytest.approx(math.radians(2))
    assert yaw.fresh_rate(5) is None
    yaw.observe(heading(0), 2)
    assert yaw.fresh_rate(2) is None


@pytest.mark.parametrize("rate,side", [(0.06, -1), (-0.06, 1)])
def test_aligned_bow_with_residual_turn_receives_countersteer(rate, side):
    plan = Vision.plan_kinematic_rudder(
        (500, 500), PlayerPose((250, 250), (1., 0.)), (350, 250), [],
        speed_scale=5.22, initial_rudder=0., initial_yaw_rate=rate,
        yaw_response_seconds=6., horizon_seconds=30.,
    )
    assert plan.rudder * side > 0


class Controls:
    def __init__(self):
        self.takeovers = 0

    def resynchronize_forward_controls(self):
        self.takeovers += 1

    def stop(self):
        pass


def make_bot():
    controls = Controls()
    bot = BattleBot(1, {"strategy": {}}, vision=object(), gamepad=controls)
    bot.intervention = SimpleNamespace(poll=lambda *_: False)
    return bot, controls


def test_arrival_and_failure_establish_control_once_and_disable_native_rearm():
    bot, controls = make_bot()
    bot.enable_opening_autopilot("test")
    bot.enable_generic_center_route("arrival")
    bot.enable_generic_center_route("repeat")
    assert controls.takeovers == 1
    assert bot.native_autopilot_abandoned
    assert not bot.opening_autopilot_active
    bot.enable_opening_autopilot("next")
    bot.request_autopilot_retry("stalled")
    bot.request_autopilot_retry("repeated failure")
    bot.enable_generic_center_route("main loop")
    assert controls.takeovers == 2


def test_confirmed_green_hud_cannot_lock_a_proven_stationary_ship_forever():
    bot, controls = make_bot()
    bot.enable_opening_autopilot("test")
    bot._native_autopilot_confirmed = True
    analysis = BattleAnalysis(image=None, width=1000, height=1000,
                              speed_knots=0, player_position=(400, 400),
                              minimap_player_normalized=(.4, .4))
    for t in range(50):
        bot._execute_rules(analysis, float(t))
        assert bot.opening_autopilot_active
    bot._execute_rules(analysis, 50.)
    assert controls.takeovers == 1
    assert bot.autopilot_retry_pending
    assert not bot.opening_autopilot_active
