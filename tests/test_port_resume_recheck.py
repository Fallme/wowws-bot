from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from main import prepare_battle, timed_port_action
from core.ui import ScreenState


@pytest.mark.parametrize("ship_ok,mode_ok,queued", [
    (True, True, True), (False, True, False), (True, False, False),
])
def test_resume_overrides_ship_lock_and_blocks_queue_until_verified(ship_ok, mode_ok, queued):
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    bot = SimpleNamespace(hwnd=1, _port_recheck_required=True,
                          vision=SimpleNamespace(grab=lambda *a, **kw: frame))
    with (
        patch("main.operation_paused", return_value=False),
        patch("main.wait_while_loading", return_value=frame),
        patch("main.classify_runtime_screen", return_value=ScreenState.PORT),
        patch("main.time.sleep"),
        patch("main.select_requested_ship", return_value=ship_ok) as ship,
        patch("main.ensure_selected_ship_commander", return_value=True),
        patch("main.ensure_requested_mode", return_value=mode_ok) as mode,
        patch("main.enter_battle", return_value=True) as enter,
    ):
        assert prepare_battle(bot, configure_port=False) == queued
    ship.assert_called_once()
    assert mode.call_count == int(ship_ok)
    assert enter.call_count == int(queued)
    assert bot._port_recheck_required == (not queued)


def test_operation_duration_is_on_its_own_log_line(caplog):
    with patch("main.time.perf_counter", side_effect=[1., 1.02]), caplog.at_level("INFO"):
        assert timed_port_action("模式复核", lambda: True)
    assert "模式复核完成   20ms" in caplog.text
