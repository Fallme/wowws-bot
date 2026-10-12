"""A transient map measurement failure uses the remaining bounded attempts."""
from unittest.mock import Mock
import main
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
