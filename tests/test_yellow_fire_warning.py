"""The paired yellow flame warnings missed in the final Rupprecht trial."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from core.vision import Vision


def warning_frame(width=1920, height=1080, ui_scale=1., *, kind="fire", missing=None):
    image = np.full((height, width, 3), 35, np.uint8)
    template = cv2.imread(str(Path(__file__).parent.parent /
                               "assets/ui/fire_warning_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert template is not None
    glyph = template.copy()
    if kind in {"outline", "exclamation"}:
        glyph[13:35, 10:31] = 0
        if kind == "exclamation":
            cv2.rectangle(glyph, (18, 16), (21, 27), 255, -1)
            cv2.rectangle(glyph, (18, 30), (21, 33), 255, -1)
    elif kind == "solid":
        glyph[:] = 255
    elif kind == "orange_blob":
        glyph[:] = 0
        cv2.ellipse(glyph, (20, 20), (17, 14), 0, 0, 360, 255, -1)
    side = round(40 * ui_scale)
    glyph = cv2.resize(glyph, (side, side), interpolation=cv2.INTER_NEAREST)
    rects = {}
    for name, (cx, cy) in {
        "health_text": (width * .15, height * .76),
        "fire_ship": (width * .57, height * .70),
    }.items():
        x, y = round(cx), round(cy)
        rects[name] = (x - 10, y - 10, x + side + 10, y + side + 10)
        if missing != name:
            image[y:y + side, x:x + side][glyph > 0] = (0, 170, 200)
    rects["fire_status"] = (0, 0, 50, 50)
    rects["fire_dial"] = (50, 0, 100, 50)
    layout = SimpleNamespace(
        crop=lambda frame, name: frame[rects[name][1]:rects[name][3],
                                       rects[name][0]:rects[name][2]],
        scale=lambda _name: ui_scale,
    )
    return image, layout


@pytest.mark.parametrize("width,height,scale", [
    (1280, 720, .5), (1920, 1080, 1.25), (3440, 1440, .75), (3840, 2160, 1.5),
])
def test_paired_flame_icons_follow_two_moved_regions_and_independent_ui_size(width, height, scale):
    image, layout = warning_frame(width, height, scale)
    with patch("core.vision.detect_hud_layout", return_value=layout):
        vision = Vision(screen_capture=object())
        assert vision.fire_status_icon_visible(image)
        assert vision.is_on_fire(image)


@pytest.mark.parametrize("missing", ["health_text", "fire_ship"])
def test_one_warning_cannot_authorize_fire(missing):
    image, layout = warning_frame(missing=missing)
    assert not Vision._yellow_fire_warning_pair(image, layout)


@pytest.mark.parametrize("kind", ["outline", "exclamation", "solid", "orange_blob"])
def test_warning_outline_other_warning_symbol_and_deck_shapes_are_not_flames(kind):
    image, layout = warning_frame(kind=kind)
    assert not Vision._yellow_fire_warning_pair(image, layout)


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("scale", [.5, .75, 1, 1.25])
def test_retained_live_fire_warnings_are_recognized_without_numeric_hp_evidence(phase, scale):
    folder = Path("E:/aimemo/docs/screenshots/wows_blockages/health_reading_20261005_180808_1791194888360806000")
    source = folder / ("frame_0001_before.png" if phase == "before" else "frame_0002_after.png")
    if not source.exists():
        pytest.skip("Optional retained live fire capture unavailable")
    image = cv2.imread(str(source))
    assert image is not None
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    vision = Vision(screen_capture=object())
    assert vision.fire_status_icon_visible(image)
    assert vision.is_on_fire(image)


def test_two_confirmed_warning_frames_trigger_damage_control_at_full_numeric_hp(tmp_path):
    from bot import BattleBot
    from test_battle_navigation import ConflictingHealthVision, FixtureDistanceReader

    class WarningVision(ConflictingHealthVision):
        def classify_screen(self, _image):
            from core.ui import ScreenState
            return ScreenState.BATTLE

        fire_anchor_bits = Vision.fire_anchor_bits
        is_on_fire = Vision.is_on_fire
        _yellow_fire_warning_pair = staticmethod(Vision._yellow_fire_warning_pair)

        def fire_status_icon_visible(self, frame):
            return Vision._yellow_fire_warning_pair(frame, layout)

    from core.hud_layout import detect_hud_layout
    image = cv2.imread(str(Path(__file__).parent / "fixtures/live_battle.png"))
    assert image is not None
    layout = detect_hud_layout(image)
    template = cv2.imread(str(Path(__file__).parent.parent /
                               "assets/ui/fire_warning_mask.png"), cv2.IMREAD_GRAYSCALE)
    for name in ("health_text", "fire_ship"):
        x1, y1, x2, y2 = layout.rect(name)
        side = round(40 * layout.scale(name))
        x, y = (x1 + x2 - side) // 2, (y1 + y2 - side) // 2
        glyph = cv2.resize(template, (side, side), interpolation=cv2.INTER_NEAREST)
        image[y - 3:y + side + 3, x - 3:x + side + 3] = 35
        image[y:y + side, x:x + side][glyph > 0] = (0, 170, 200)
    assert Vision._yellow_fire_warning_pair(image, layout)
    gamepad = SimpleNamespace(damage_control=Mock(return_value=None),
                              set_movement=Mock(), stop=Mock())
    bot = BattleBot(1, {"strategy": {}}, vision=WarningVision(image),
                    gamepad=gamepad, distance_reader=FixtureDistanceReader())
    bot.last_damage_control = 0.
    bot.intervention = SimpleNamespace(poll=Mock(return_value=False), latched=False)
    with patch("bot.DEFAULT_DEBUG_ROOT", tmp_path), \
         patch("bot.time.monotonic", return_value=100):
        first = bot.analyze()
        assert first.health == 1 and not first.on_fire
        bot._execute_survival_consumables(first, 100)
        gamepad.damage_control.assert_not_called()
        bot.tick = 1
        second = bot.analyze()
        assert second.health == 1 and second.on_fire
        bot._execute_survival_consumables(second, 100)
    gamepad.damage_control.assert_called_once()
