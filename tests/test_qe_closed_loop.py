from types import SimpleNamespace

import pytest

from bot import BattleAnalysis, BattleBot
from core.keyboard import KeyboardController
from strategy.helm_model import HelmModel, MapMotion
from strategy.secondary_movement import MovementCommand, MovementMode


class Controls:
    def __init__(self, fail=False):
        self.commands = []
        self.fail = fail

    def set_movement(self, throttle, rudder):
        if self.fail:
            raise RuntimeError("input failed")
        self.commands.append((throttle, rudder))

    def stop(self):
        pass


def setup_bot(fail=False):
    controls = Controls(fail)
    bot = BattleBot(1, {"strategy": {"rudder_minimum_hold_seconds": 4}},
                    vision=object(), gamepad=controls)
    bot.intervention = SimpleNamespace(poll=lambda *_: False)
    bot.movement.plan = lambda _: MovementCommand(MovementMode.APPROACH, .75, -.5, "test")
    bot.stuck_recovery = SimpleNamespace(reverse_until=None, update=lambda *a, **kw: None, cancel=lambda: None)
    bot._movement_feedback_update = lambda *a: None
    return bot, controls


def analysis(**kwargs):
    values = dict(image=None, width=1000, height=1000, in_battle=True,
                  player_position=(400, 400))
    values.update(kwargs)
    return BattleAnalysis(**values)


def test_input_failure_does_not_commit_proposed_helm():
    bot, _ = setup_bot(fail=True)
    with pytest.raises(RuntimeError, match="input failed"):
        bot._execute_rules(analysis(), 100)
    assert bot._last_applied_rudder == 0
    assert bot._rudder_commanded_at == 0
    assert bot.helm_model.order == 0


def test_predictive_countersteer_bypasses_ordinary_direction_hold():
    bot, controls = setup_bot()
    bot._record_applied_helm(.5, 99)
    bot.yaw_motion.rate = .03
    bot.yaw_motion.observed_at = 100
    bot._execute_rules(analysis(kinematic_rudder=-.5), 100)
    assert controls.commands == [(.75, -.5)]
    assert bot._last_applied_rudder == -.5


def test_cached_pose_never_escalates_stuck_recovery():
    bot, controls = setup_bot()
    bot.stuck_recovery.update = lambda *a, **kw: pytest.fail("cached pose used for recovery")
    bot._execute_rules(analysis(player_pose_cached=True), 100)
    assert controls.commands == []


def test_lost_pose_reduces_speed_and_centers_helm():
    bot, controls = setup_bot()
    bot._execute_rules(analysis(player_position=None), 100)
    assert controls.commands == [(.25, 0)]
    assert bot.helm_model.order == 0


def test_partial_rudder_send_retries_only_remaining_notches():
    class Backend:
        def __init__(self):
            self.calls = 0
            self.accepted = 0

        def tap(self, key):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("input failed")
            assert key == "e"
            self.accepted += 1

    backend = Backend()
    control = KeyboardController(backend)
    with pytest.raises(RuntimeError):
        control._set_rudder(1)
    assert control._rudder_notch == 1
    control._set_rudder(1)
    assert backend.accepted == 2
    assert control._rudder_notch == 2


def test_out_of_order_timestamp_cannot_advance_helm_twice():
    helm = HelmModel(15)
    helm.command(1, 10)
    helm.advance(20)
    before = helm.actual
    helm.advance(15)
    assert helm.advance(20) == before
    motion = MapMotion()
    motion.speed, motion.observed_at = .08, 20
    assert motion.fresh_speed(19) is None


def test_async_distance_result_after_enemy_disappears_is_not_adopted():
    bot, _ = setup_bot()
    bot._distance_ocr_async = True
    observation = SimpleNamespace(value_km=8., confidence=.9, accepted=True, raw_text="8.0")
    result = SimpleNamespace(point=(100, 100), observation=observation,
                             evidence=None, error=None)
    bot.distance_ocr_service = SimpleNamespace(
        poll=lambda: result, pending=False, submit=lambda *a, **kw: False,
    )
    track, nearest, accepted, _, stable = bot._update_target_distance(None, [], 100., 500.)
    assert track is nearest is accepted is stable is None
