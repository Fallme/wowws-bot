"""A transient map measurement failure uses the remaining bounded attempts."""
from collections import deque
from types import MethodType
from unittest.mock import Mock

import main
import pytest
from bot import BattleBot
from test_autopilot_interrupted_setup import setup


def test_next_fresh_map_can_recover_without_clicking_unmeasured_geometry(setup, monkeypatch):
    bot, state = setup.bot, setup.state
    state.interruption = None
    state.enable_after_click = 1
    geometry = main.detect_tactical_map_geometry(None)
    measure = Mock(side_effect=[None, geometry])
    monkeypatch.setattr(main, "detect_tactical_map_geometry", measure)

    assert main.configure_opening_autopilot(bot)
    assert measure.call_count == 2
    assert state.actions.count("M_open") == 2
    assert len(state.clicks) == 1 and state.clicks[0][0] == "physical"
    assert bot._tactical_map_attempts_used == 2
    assert bot.opening_autopilot_active


def test_three_unmeasured_maps_close_safely_and_never_send_a_waypoint(setup, monkeypatch):
    bot, state = setup.bot, setup.state
    state.interruption = None
    measure = Mock(return_value=None)
    monkeypatch.setattr(main, "detect_tactical_map_geometry", measure)

    assert main.configure_opening_autopilot(bot) is False
    assert measure.call_count == 3
    assert state.actions.count("M_open") == 3
    assert state.actions.count("M_close") == 3
    assert not state.map_open and not state.clicks
    assert main.configure_opening_autopilot(bot) is False
    assert state.actions.count("M_open") == 3


def seed_previous_battle_map(bot):
    bot._battle_map_islands = [{"points": [[.1, .1], [.3, .1], [.3, .9]], "area": .08}]
    bot._battle_capture_zone_layout = [
        {"label": label, "position": [.5, .5], "radius": .08}
        for label in "ABCD"
    ]
    bot._battle_capture_zones = [object() for _ in "ABCD"]
    bot._static_map_source = "previous_tactical_map"
    bot._island_layer_candidates = deque([list(bot._battle_map_islands)], maxlen=5)
    bot._zone_layer_candidates = deque([list(bot._battle_capture_zone_layout)], maxlen=5)
    bot.begin_tactical_map_static_capture = Mock(
        wraps=MethodType(BattleBot.begin_tactical_map_static_capture, bot)
    )


def test_all_grid_failures_cannot_leak_previous_four_point_map_into_new_battle(setup, monkeypatch):
    bot, state = setup.bot, setup.state
    state.interruption = None
    seed_previous_battle_map(bot)

    def unmeasured(_frame):
        # The geometry detector may reject every snow-covered M frame. The
        # old layout must already be invalidated on the first such frame.
        assert bot._battle_map_islands == []
        assert bot._battle_capture_zone_layout == []
        assert bot._battle_capture_zones == []
        return None

    monkeypatch.setattr(main, "detect_tactical_map_geometry", unmeasured)

    assert main.configure_opening_autopilot(bot) is False
    bot.begin_tactical_map_static_capture.assert_called_once()
    assert bot._tactical_map_attempts_used == 3
    assert not state.map_open and not state.clicks
    assert not bot._island_layer_candidates and not bot._zone_layer_candidates
    assert bot._static_map_source == ""

    # Production preserves the consumed M budget at this handoff. It now
    # preserves an empty layer, allowing current minimap evidence to rebuild
    # the route rather than steering/stopping against the previous map.
    assert main.run_battle(bot, resume_existing=False) is True
    assert bot.reset.call_args.kwargs["preserve_static_map"] is True
    assert bot._battle_map_islands == []
    assert bot._battle_capture_zone_layout == []
    bot.enable_generic_center_route.assert_called_once()


def test_grid_recovery_samples_new_static_layer_after_old_map_is_cleared(setup, monkeypatch):
    bot, state = setup.bot, setup.state
    state.interruption = None
    state.enable_after_click = 1
    geometry = main.detect_tactical_map_geometry(None)
    seed_previous_battle_map(bot)
    new_islands = [{"points": [[.6, .2], [.8, .2], [.8, .4]], "area": .02}]
    new_zones = [{"label": label, "position": [.4, .4], "radius": .08} for label in "ABC"]

    def sample(_frame):
        bot._battle_map_islands = new_islands
        bot._battle_capture_zone_layout = new_zones
        return True

    bot.capture_tactical_map_static_layer = Mock(side_effect=sample)
    monkeypatch.setattr(main, "detect_tactical_map_geometry", Mock(side_effect=[None, geometry]))

    assert main.configure_opening_autopilot(bot)
    bot.begin_tactical_map_static_capture.assert_called_once()
    assert bot.capture_tactical_map_static_layer.called
    assert bot._battle_map_islands == new_islands
    assert [zone["label"] for zone in bot._battle_capture_zone_layout] == list("ABC")
    assert bot._tactical_map_attempts_used == 2


@pytest.mark.parametrize("early_exit", ["missing_player", "non_battle_scene"])
def test_confirmed_new_round_clears_old_map_before_any_setup_early_exit(setup, monkeypatch, early_exit):
    bot, state = setup.bot, setup.state
    state.interruption = None
    seed_previous_battle_map(bot)
    main.reset_opening_autopilot_attempts(bot)
    if early_exit == "missing_player":
        bot.vision.find_player_pose_on_minimap = lambda _: None
    else:
        monkeypatch.setattr(main, "classify_battle_continuity_screen", lambda *_: main.ScreenState.LOADING)

    assert main.configure_opening_autopilot(bot) is False
    assert not state.actions and not state.clicks
    assert not bot._battle_map_islands and not bot._battle_capture_zone_layout
    bot.begin_tactical_map_static_capture.assert_called_once()


def test_same_round_budget_resume_does_not_clear_newly_sampled_map(setup):
    bot, _ = setup.bot, setup.state
    seed_previous_battle_map(bot)
    main.reset_opening_autopilot_attempts(bot)
    current_map = [{"points": [[.6, .2], [.8, .2], [.8, .4]], "area": .02}]
    bot._battle_map_islands = current_map

    main.reset_opening_autopilot_attempts(bot)

    assert bot._battle_map_islands is current_map
    bot.begin_tactical_map_static_capture.assert_called_once()
