import numpy as np
import json
from unittest.mock import Mock, patch

from core.ocr import OcrToken
from core.ui import ScreenState
from core.blockage_diagnostics import BlockageTrace
from port_navigator import (
    _is_early_exit_confirmation,
    force_quick_battle_return_to_port,
    queue_next_battle,
    quick_death_action_point,
    quick_death_leave_hint_visible,
    quick_exit_menu_action_point,
)


def token(text, left, top, right, bottom, confidence=0.98):
    return OcrToken(
        text,
        confidence,
        ((left, top), (right, top), (right, bottom), (left, bottom)),
    )


class StaticBackend:
    def __init__(self, tokens):
        self.tokens = tokens

    def recognize(self, _image):
        return list(self.tokens)


class BrokenBackend:
    @staticmethod
    def recognize(_image):
        raise RuntimeError("OCR unavailable")


def test_quick_exit_finds_leave_battle_and_rejects_exit_game():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)
    backend = StaticBackend(
        [
            token("退出游戏", 1150, 520, 1390, 570),
            token("离开战斗", 1160, 590, 1400, 640),
            token("回到战斗", 1160, 700, 1400, 750),
        ]
    )

    assert quick_exit_menu_action_point(image, backend) == (1280, 615)
    assert (
        quick_exit_menu_action_point(
            image,
            StaticBackend([token("退出游戏", 1150, 520, 1390, 570)]),
        )
        is None
    )


def test_early_exit_second_page_requires_both_text_anchors():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)
    confirmed = StaticBackend(
        [
            token("提前退出战斗", 1170, 390, 1390, 440),
            token("离开战斗?", 1190, 830, 1370, 880),
            token("是", 1160, 900, 1210, 945),
            token("否", 1350, 900, 1400, 945),
        ]
    )
    missing_heading = StaticBackend(
        [token("离开战斗?", 1190, 830, 1370, 880)]
    )

    assert _is_early_exit_confirmation(image, confirmed)
    assert not _is_early_exit_confirmation(image, missing_heading)


def test_early_exit_ocr_failure_never_authorizes_confirmation_click():
    image = np.zeros((1494, 2560, 3), dtype=np.uint8)

    assert not _is_early_exit_confirmation(image, BrokenBackend())


def quick_frame(marker):
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    image[0, 0, 0] = marker
    return image


def test_sunk_quick_battle_waits_through_loading_then_exits_verified_menu(tmp_path):
    frames = [quick_frame(n) for n in (1, 1, 2, 3, 4, 5, 5)]
    states = {
        1: ScreenState.LOADING,
        2: ScreenState.BATTLE,
        3: ScreenState.ESCAPE_MENU,
        4: ScreenState.EXIT_CONFIRMATION,
        5: ScreenState.PORT,
    }

    class Vision:
        @staticmethod
        def classify_screen(image):
            return states[image[0, 0, 0]]

    class Backend:
        @staticmethod
        def recognize(image):
            marker = image[0, 0, 0]
            if marker == 3:
                return [token("离开战斗", 450, 280, 550, 310)]
            if marker == 4:
                return [
                    token("提前退出战斗", 400, 150, 600, 180),
                    token("离开战斗?", 400, 275, 600, 305),
                    token("是", 450, 340, 480, 370),
                ]
            return []

    with (
        patch("port_navigator._capture", side_effect=frames),
        patch("port_navigator._click_local", return_value=True) as click,
        patch("port_navigator.time.sleep"),
    ):
        escape = Mock()
        result = force_quick_battle_return_to_port(
            1, vision=Vision(), backend=Backend(), open_menu=escape,
            attempts=len(frames), diagnostic_dir=tmp_path,
        )

    assert result == ScreenState.PORT
    escape.assert_called_once()
    assert click.call_count == 2
    assert click.call_args_list[0].args[1] == (500, 295)
    yes_x, yes_y = click.call_args_list[1].args[1]
    assert 440 <= yes_x <= 490 and 330 <= yes_y <= 390
    traces = list(tmp_path.glob("quick_exit_blocked_*/observations.jsonl"))
    assert len(traces) == 1
    assert json.loads(traces[0].read_text(encoding="utf-8").splitlines()[0])["state"] == "loading"


def test_sunk_quick_battle_esc_finds_continue_without_port_or_results():
    frames = [quick_frame(n) for n in (1, 1, 2, 2)]
    states = {
        1: ScreenState.LOADING,
        2: ScreenState.UNKNOWN,
    }

    class Vision:
        @staticmethod
        def classify_screen(image):
            return states[image[0, 0, 0]]

    current = {"marker": 0}

    def capture(_hwnd):
        image = frames.pop(0)
        current["marker"] = image[0, 0, 0]
        return image

    class Backend:
        @staticmethod
        def recognize(image):
            marker = current["marker"]
            if marker == 1:
                return [token("您的战舰已被玩家摧毁", 40, 65, 260, 95),
                        token("离开战斗", 95, 125, 190, 155),
                        token("继续战斗", 260, 196, 380, 226)]
            if marker == 2:
                return [token("继续战斗", 260, 196, 380, 226)]
            return []

    with (
        patch("port_navigator._capture", side_effect=capture),
        patch("port_navigator._click_local", return_value=True) as click,
        patch("port_navigator.time.sleep"),
    ):
        escape = Mock()
        assert force_quick_battle_return_to_port(
            1, vision=Vision(), backend=Backend(), open_menu=escape,
            ship_sunk=True, attempts=len(frames),
        ) == ScreenState.QUICK_REQUEUE

    click.assert_not_called()
    escape.assert_called_once()


def test_post_death_continue_click_queues_next_battle_directly():
    frames = [quick_frame(2), quick_frame(3)]

    class Vision:
        @staticmethod
        def classify_screen(image):
            return (ScreenState.LOADING if image[0, 0, 0] == 3
                    else ScreenState.UNKNOWN)

    with (
        patch("port_navigator._capture", side_effect=frames),
        patch("port_navigator._click_local", return_value=True) as click,
        patch("port_navigator.time.sleep"),
    ):
        assert queue_next_battle(
            1, vision=Vision(), backend=StaticBackend([
                token("继续战斗", 260, 196, 380, 226),
            ]), source=ScreenState.QUICK_REQUEUE,
        )
    click.assert_called_once_with(1, (500, 295))


def test_death_action_finds_continue_and_ignores_return_port_or_game_exit():
    image = quick_frame(1)
    assert quick_death_action_point(image, StaticBackend([
        token("返回港口", 220, 148, 340, 178),
        token("继续战斗", 260, 196, 380, 226),
    ])) == (500, 295)
    assert quick_death_action_point(image, StaticBackend([
        token("退出游戏", 220, 148, 340, 178),
    ])) is None


def test_sunk_quick_battle_can_still_observe_natural_results_after_esc():
    frames = [quick_frame(n) for n in (1, 1, 5, 5)]

    class Vision:
        @staticmethod
        def classify_screen(image):
            return ScreenState.RESULTS if image[0, 0, 0] == 5 else ScreenState.LOADING

    backend = StaticBackend([
        token("您的战舰已被玩家摧毁", 40, 65, 260, 95),
        token("ESC", 40, 125, 85, 155),
        token("离开战斗", 95, 125, 190, 155),
    ])
    assert quick_death_leave_hint_visible(quick_frame(1), backend)
    assert not quick_death_leave_hint_visible(quick_frame(1), StaticBackend([
        token("离开战斗", 95, 125, 190, 155),
    ]))
    with (
        patch("port_navigator._capture", side_effect=frames),
        patch("port_navigator._click_local") as click,
        patch("port_navigator.time.sleep"),
    ):
        escape = Mock()
        assert force_quick_battle_return_to_port(
            1, vision=Vision(), backend=backend, open_menu=escape,
            ship_sunk=True, attempts=len(frames),
        ) == ScreenState.RESULTS
    escape.assert_called_once()
    click.assert_not_called()


def test_quick_exit_does_not_consume_click_when_dispatch_fails():
    frames = [quick_frame(n) for n in (2, 3, 3, 4, 5, 5)]

    class Vision:
        @staticmethod
        def classify_screen(image):
            return ScreenState.PORT if image[0, 0, 0] == 5 else ScreenState.BATTLE

    class Backend:
        @staticmethod
        def recognize(image):
            marker = image[0, 0, 0]
            if marker == 3:
                return [token("离开战斗", 450, 280, 550, 310)]
            if marker == 4:
                return [token("提前退出战斗", 400, 150, 600, 180),
                        token("离开战斗?", 400, 275, 600, 305),
                        token("是", 450, 340, 480, 370)]
            return []

    with (
        patch("port_navigator._capture", side_effect=frames),
        patch("port_navigator._click_local", side_effect=[False, True, True]) as click,
        patch("port_navigator.time.sleep"),
    ):
        assert force_quick_battle_return_to_port(
            1, vision=Vision(), backend=Backend(), open_menu=Mock(),
            attempts=len(frames),
        ) == ScreenState.PORT
    assert click.call_count == 3


def test_ambiguous_quick_exit_never_sends_escape_or_click(tmp_path):
    image = quick_frame(1)
    vision = Mock(classify_screen=Mock(return_value=ScreenState.LOADING))
    escape = Mock()
    with (
        patch("port_navigator._capture", return_value=image),
        patch("port_navigator.time.sleep"),
        patch("port_navigator._click_local") as click,
    ):
        assert force_quick_battle_return_to_port(
            1, vision=vision, backend=StaticBackend([]), open_menu=escape,
            attempts=5, diagnostic_dir=tmp_path,
        ) == ScreenState.UNKNOWN
    escape.assert_not_called()
    click.assert_not_called()
    assert list(tmp_path.glob("quick_exit_blocked_*/frame_*.jpg"))


def test_quick_exit_pause_during_wait_prevents_next_capture():
    paused = [False]
    image = quick_frame(1)
    with (
        patch("port_navigator._capture", return_value=image) as capture,
        patch("port_navigator.time.sleep", side_effect=lambda _delay: paused.__setitem__(0, True)),
    ):
        assert force_quick_battle_return_to_port(
            1, vision=Mock(classify_screen=Mock(return_value=ScreenState.LOADING)),
            backend=StaticBackend([]), open_menu=Mock(),
            should_abort=lambda: paused[0], attempts=5,
        ) == ScreenState.UNKNOWN
    assert capture.call_count == 1


def test_blockage_trace_retains_three_spaced_frames_with_decisions(tmp_path, monkeypatch):
    instants = iter((0.0, 0.2, 0.8, 1.6, 2.4, 3.2))
    monkeypatch.setattr("core.blockage_diagnostics.time.monotonic", lambda: next(instants))
    trace = BlockageTrace(tmp_path, "quick_exit_blocked", interval=0.7)
    for attempt in range(4):
        trace.record(quick_frame(attempt), state=ScreenState.LOADING,
                     action="await_battle_or_result", attempt=attempt)
    trace.record(quick_frame(4), state=ScreenState.EXIT_CONFIRMATION,
                 action="exit_confirmation", attempt=4)
    trace.record(quick_frame(5), state=ScreenState.EXIT_CONFIRMATION,
                 action="exit_confirmation", attempt=5)
    assert trace.count == 5
    files = sorted(trace.folder.glob("frame_*.jpg"))
    assert len(files) == 5
    observations = [json.loads(line) for line in
                    (trace.folder / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["attempt"] for item in observations] == [0, 2, 3, 4, 5]
