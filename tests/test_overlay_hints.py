"""Corner close cues must not confuse a battle-exit prompt with an overlay."""

from unittest.mock import Mock

import numpy as np
import pytest

from core.ocr import OcrToken
from core.overlay_hints import escape_close_hint


def token(text, *, x=10, y=10, width=30, height=18, confidence=.99):
    return OcrToken(text, confidence, ((x, y), (x + width, y), (x + width, y + height), (x, y + height)))


def image():
    return np.full((1000, 1600, 3), 60, np.uint8)


@pytest.mark.parametrize("corner,offset", [("left", 0), ("right", 1152)])
def test_separate_esc_and_cross_return_a_point_in_the_correct_corner(corner, offset):
    cue = [token("Esc"), token("×", x=45, width=10)]
    backend = Mock()
    backend.recognize.side_effect = [cue, []] if corner == "left" else [[], cue]
    assert escape_close_hint(image(), backend) == (offset + 50, 19)
    assert [call.args[0].shape for call in backend.recognize.call_args_list] == [(140, 416, 3), (140, 448, 3)]


@pytest.mark.parametrize("text", ["ESC ×", "Esc X", "ESC 关闭", "ESC 關閉", "Esc返回", "Esc取消", "ESC Back", "Close ESC"])
def test_combined_explicit_close_hint_is_supported(text):
    backend = Mock(recognize=Mock(side_effect=[[token(text, width=70)], []]))
    assert escape_close_hint(image(), backend) == (45, 19)


@pytest.mark.parametrize("label", ["X", "关闭", "返回", "取消", "Back", "Close"])
def test_adjacent_close_word_is_supported(label):
    backend = Mock(recognize=Mock(side_effect=[[token("ESC"), token(label, x=45, width=40)], []]))
    assert escape_close_hint(image(), backend) == (65, 19)


@pytest.mark.parametrize("phrase", ["ESC 离开战斗", "ESC 退出战斗", "ESC 菜单", "ESC 设置", "ESC Leave Battle", "ESC Exit Battle", "ESC Menu", "ESC Settings", "Escape"])
def test_battle_exit_or_menu_prompts_never_authorize_overlay_escape(phrase):
    backend = Mock(recognize=Mock(side_effect=[[token(phrase, width=150)], []]))
    assert escape_close_hint(image(), backend) is None


@pytest.mark.parametrize("phrase", ["离开战斗", "退出游戏", "菜单", "设置", "LEAVE BATTLE", "SETTINGS"])
def test_adjacent_forbidden_text_vetoes_even_an_esc_cross_pair(phrase):
    cue = [token("ESC"), token("×", x=45, width=10), token(phrase, x=60, width=80)]
    backend = Mock(recognize=Mock(side_effect=[cue, []]))
    assert escape_close_hint(image(), backend) is None


@pytest.mark.parametrize("cue", [[token("ESC")], [token("×")], [token("Close")]])
def test_key_or_close_semantics_alone_are_insufficient(cue):
    backend = Mock(recognize=Mock(side_effect=[cue, []]))
    assert escape_close_hint(image(), backend) is None


@pytest.mark.parametrize("close", [token("×", x=180, width=10), token("×", x=45, y=80, width=10)])
def test_close_glyph_must_be_adjacent_and_on_the_same_line(close):
    backend = Mock(recognize=Mock(side_effect=[[token("ESC"), close], []]))
    assert escape_close_hint(image(), backend) is None


@pytest.mark.parametrize("key_confidence,close_confidence", [(.74, .99), (.99, .74), (float("nan"), .99)])
def test_low_or_invalid_confidence_is_rejected(key_confidence, close_confidence):
    cue = [token("ESC", confidence=key_confidence), token("×", x=45, width=10, confidence=close_confidence)]
    backend = Mock(recognize=Mock(side_effect=[cue, []]))
    assert escape_close_hint(image(), backend) is None


def test_minimum_confidence_is_accepted_with_complete_geometric_evidence():
    backend = Mock(recognize=Mock(side_effect=[[token("ESC ×", width=70, confidence=.75)], []]))
    assert escape_close_hint(image(), backend) == (45, 19)


@pytest.mark.parametrize("box", [(), ((1, 1), (5, 5)), ((-1, 10), (20, 10), (20, 28), (-1, 28)),
                               ((10, 200), (40, 200), (40, 218), (10, 218)),
                               ((10, 10), (float("nan"), 10), (40, 28), (10, 28))])
def test_missing_or_out_of_corner_boxes_are_not_close_evidence(box):
    backend = Mock(recognize=Mock(side_effect=[[OcrToken("ESC ×", .99, box)], []]))
    assert escape_close_hint(image(), backend) is None


def test_two_corner_close_hints_are_ambiguous():
    cue = [token("ESC ×", width=70)]
    backend = Mock(recognize=Mock(side_effect=[cue, cue]))
    assert escape_close_hint(image(), backend) is None


def test_backend_failure_is_closed_even_after_an_earlier_valid_corner():
    backend = Mock(recognize=Mock(side_effect=[[token("ESC ×", width=70)], RuntimeError("OCR unavailable")]))
    assert escape_close_hint(image(), backend) is None


def test_missing_backend_and_invalid_images_do_not_start_ocr():
    assert escape_close_hint(image(), None) is None
    backend = Mock()
    for invalid in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8)):
        assert escape_close_hint(invalid, backend) is None
    backend.recognize.assert_not_called()
