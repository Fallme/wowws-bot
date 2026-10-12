"""Conservative screenshot evidence for a close-range torpedo opportunity.

The game supplies the lead cone and loaded (green) firing arc. Neither a
distance estimate nor the nominal YAML reload duration authorizes a shot.
"""

from __future__ import annotations

import re
import math
from functools import lru_cache, partial
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.hud_layout import detect_hud_layout


@dataclass(frozen=True)
class TorpedoHud:
    cooldown_seconds: int | None
    loaded_arc: bool
    lead_point: tuple[int, int] | None
    weapon_is_torpedo: bool | None = None
    selected: bool = False
    aim_point: tuple[int, int] | None = None
    alignment_error_px: float | None = None
    alignment_tolerance_px: float | None = None
    alignment_error_degrees: float | None = None
    reticle_point: tuple[int, int] | None = None
    slot_rect: tuple[int, int, int, int] | None = None
    slot_scale: float = 1.0
    frame_shape: tuple[int, int] | None = None
    bearing_available: bool = True

    @property
    def ready_to_align(self) -> bool:
        return bool(self.weapon_is_torpedo is True and self.selected
                    and self.loaded_arc and self.lead_point is not None
                    and self.aim_point is not None and not (
                        self.cooldown_seconds is not None and self.cooldown_seconds > 0))

    @property
    def aligned(self) -> bool:
        values = (self.alignment_error_px, self.alignment_tolerance_px,
                  self.alignment_error_degrees)
        if any(value is None or not math.isfinite(value) or value < 0 for value in values):
            return False
        return bool(self.alignment_error_px is not None
                    and self.alignment_tolerance_px is not None
                    and self.alignment_error_px <= self.alignment_tolerance_px
                    and self.alignment_error_degrees is not None
                    and self.alignment_error_degrees <= 2.0)

    @property
    def safe_to_fire(self) -> bool:
        return self.bearing_available is True and self.ready_to_align and self.aligned


@lru_cache(maxsize=1)
def _weapon_template():
    image = cv2.imread(str(Path(__file__).resolve().parent.parent /
                          "assets" / "ui" / "torpedo_weapon.png"), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return None
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return gray, image[:, :, 3]


@dataclass(frozen=True)
class _SlotEvidence:
    weapon: bool | None
    selected: bool
    rect: tuple[int, int, int, int] | None = None
    scale: float = 1.0

    def __iter__(self):
        yield self.weapon
        yield self.selected


def _valid_tokens(crop, backend, *, minimum_confidence=.75):
    if backend is None or crop.size == 0:
        return []
    try:
        tokens = backend.recognize(crop) or ()
    except Exception:
        return []
    result = []
    for token in tokens:
        try:
            box = np.asarray(getattr(token, "box", ()), np.float32)
            confidence = float(getattr(token, "confidence", 0))
        except (TypeError, ValueError):
            continue
        if (not math.isfinite(confidence) or confidence < minimum_confidence
                or box.shape != (4, 2) or not np.all(np.isfinite(box))
                or box.min() < 0 or box[:, 0].max() > crop.shape[1]
                or box[:, 1].max() > crop.shape[0]):
            continue
        result.append(token)
    return result


def _weapon_slot_evidence(image, backend):
    """Match the actual torpedo body plus its own labelled [3] keycap."""
    if image is None or image.size == 0:
        return _SlotEvidence(None, False)
    template = _weapon_template()
    if template is None:
        return _SlotEvidence(None, False)
    height, width = image.shape[:2]
    layout = detect_hud_layout(image)
    bounds = layout.map_rect((.28 * 2560, .84 * 1494, .73 * 2560, 1494), "center")
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return _SlotEvidence(None, False)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 35, 100)
    source, alpha = template
    measured = layout.scale("consumables")
    scales = sorted(set(round(measured * factor, 3) for factor in (.72, .82, .90, 1, 1.10, 1.22, 1.38)))
    if layout.source("consumables") != "anchor":
        scales = sorted(set(scales + [.4, .55, .7, .85, 1, 1.25, 1.5, 1.8, 2.2]))
    matches = []
    for scale in scales:
        tw, th = round(source.shape[1] * scale), round(source.shape[0] * scale)
        if min(tw, th) < 18 or tw >= gray.shape[1] or th + 8 >= gray.shape[0]:
            continue
        sample = cv2.resize(source, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        mask = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
        # Alpha excludes the selected green sector and its border, allowing
        # the same weapon identity to be read before and after selection.
        sample_edges = cv2.Canny(sample, 35, 100)
        if np.count_nonzero(sample_edges[mask > 0]) < 12:
            continue
        scores = cv2.matchTemplate(edge, sample_edges, cv2.TM_CCORR_NORMED, mask=mask)
        scores[~np.isfinite(scores)] = 0
        for _ in range(3):
            _, score, _, point = cv2.minMaxLoc(scores)
            if score < .48:
                break
            x, y = point
            body = gray[y:y + th, x:x + tw]
            correlation = cv2.matchTemplate(body, sample, cv2.TM_CCORR_NORMED, mask=mask)[0, 0]
            if np.isfinite(correlation) and correlation >= .90:
                matches.append((float(score), left + x, top + y, tw, th, scale))
            scores[max(0, y - th // 3):min(scores.shape[0], y + th // 3 + 1),
                   max(0, x - tw // 3):min(scores.shape[1], x + tw // 3 + 1)] = 0
    matches.sort(reverse=True)
    for score, x, y, tw, th, scale in matches[:12]:
        if score < .58:
            continue
        key_left, key_top = max(0, x - round(tw * .25)), y + round(th * .88)
        key_right, key_bottom = min(width, x + round(tw * 1.25)), min(height, y + th + round(40 * scale))
        key_crop = image[key_top:key_bottom, key_left:key_right]
        key_tokens = [(token, 1) for token in _valid_tokens(key_crop, backend)]
        if scale < .7 and key_crop.size:
            enlarged = cv2.resize(key_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            key_tokens.extend((token, 3) for token in _valid_tokens(enlarged, backend))
        for token, enlargement in key_tokens:
            text = re.sub(r"\s", "", str(getattr(token, "text", ""))).replace("３", "3")
            if text not in {"3", "[3]"}:
                continue
            box = np.asarray(token.box, np.float32) / enlargement
            cx, cy = box.mean(axis=0)
            cx, cy = cx + key_left, cy + key_top
            if not (abs(cx - (x + tw / 2)) <= tw * .45 and y + th * .93 <= cy <= y + th + 32 * scale):
                continue
            kx1, ky1 = np.floor(box.min(axis=0)).astype(int)
            kx2, ky2 = np.ceil(box.max(axis=0)).astype(int)
            padding = max(2, round(3 * scale))
            key = key_crop[max(0, ky1 - padding):min(key_crop.shape[0], ky2 + padding),
                           max(0, kx1 - padding):min(key_crop.shape[1], kx2 + padding)]
            if key.size == 0 or max(key.shape[:2]) > 45 * scale:
                continue
            hsv = cv2.cvtColor(key, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, np.array([35, 90, 80]), np.array([95, 255, 255]))
            return _SlotEvidence(True, bool(np.mean(green > 0) >= .18),
                                 (int(x), int(y), int(x + tw), int(y + th)), float(scale))
    # A failed image match does not prove the absence of torpedoes. Optional
    # weapon words are accepted only inside the actual bottom weapon strip.
    for token in _valid_tokens(crop, backend):
        text = str(getattr(token, "text", ""))
        if re.fullmatch(r"3\s*[:：]?\s*(?:主炮|副炮|深水炸弹|空袭|Main guns|Airstrike)", text, re.I):
            return _SlotEvidence(False, False)
    return _SlotEvidence(None, False)


def detect_torpedo_weapon(image, backend=None):
    """Return True only for a matched torpedo icon with its own slot-3 label."""
    return _weapon_slot_evidence(image, backend).weapon


def _selected_weapon_frame(image, backend=None):
    """Read the native green key badge attached to a selection frame.

    This reads the main battery's own native key, independently of the slot-3
    icon, which can be hidden by the torpedo reload counter. A green number
    without the rectangular selection frame above it is insufficient.
    """
    if image is None or image.size == 0 or backend is None:
        return None
    layout = detect_hud_layout(image)
    height, width = image.shape[:2]
    left, top, right, bottom = layout.map_rect(
        (.28 * 2560, .84 * 1494, .73 * 2560, 1494), "center")
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    scale = max(.3, layout.scale("consumables"))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 90, 80]), np.array([95, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bounds = sorted(set(cv2.boundingRect(contour) for contour in contours),
                    key=lambda rect: rect[2] * rect[3], reverse=True)
    for x, y, w, h in bounds:
        if not (35 * scale <= w <= 150 * scale and 45 * scale <= h <= 180 * scale
                and .55 <= w / h <= 1.35):
            continue
        component = mask[y:y + h, x:x + w] > 0
        edge_width = max(2, round(4 * scale))
        body_height = max(1, round(h * .70))
        # The selected slot has two vertical borders and a full upper edge;
        # isolated countdown badges, green icons and curved fans do not.
        if (np.mean(np.any(component[:body_height, :edge_width], axis=1)) < .65
                or np.mean(np.any(component[:body_height, -edge_width:], axis=1)) < .65
                or np.mean(np.any(component[:edge_width], axis=0)) < .65
                or np.mean(component[edge_width:body_height, edge_width:-edge_width]) > .60):
            continue
        key_top = max(0, y + round(h * .74))
        key_bottom = min(crop.shape[0], y + h + round(32 * scale))
        key_crop = crop[key_top:key_bottom, x:x + w]
        if key_crop.size == 0:
            continue
        enlarged = cv2.resize(key_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        for token in _valid_tokens(enlarged, backend):
            text = re.sub(r"\s", "", str(getattr(token, "text", "")))
            if text not in {"1", "2", "3", "[1]", "[2]", "[3]"}:
                continue
            box = np.asarray(token.box, np.float32) / 3
            cx, cy = box.mean(axis=0)
            cy += key_top - y
            if not (.22 * w <= cx <= .78 * w and .82 * h <= cy <= h + 32 * scale):
                continue
            kx1, ky1 = np.floor(box.min(axis=0)).astype(int)
            kx2, ky2 = np.ceil(box.max(axis=0)).astype(int)
            padding = max(2, round(3 * scale))
            key = key_crop[max(0, ky1 - padding):min(key_crop.shape[0], ky2 + padding),
                           max(0, kx1 - padding):min(key_crop.shape[1], kx2 + padding)]
            if key.size == 0 or max(key.shape[:2]) > 45 * scale:
                continue
            key_hsv = cv2.cvtColor(key, cv2.COLOR_BGR2HSV)
            selected = cv2.inRange(key_hsv, np.array([35, 90, 80]), np.array([95, 255, 255]))
            if np.mean(selected > 0) >= .18:
                return int(text.strip("[]")), (left + x, top + y, left + x + w, top + y + h)
    return None


def detect_main_gun_selected(image, backend=None):
    """Confirm slot 2 independently of a reload counter hiding the slot-3 icon."""
    selected = _selected_weapon_frame(image, backend)
    return None if selected is None else selected[0] == 2


def _parse_reload_time(text):
    match = re.fullmatch(r"(?:(\d{1,3})|(\d{1,2}):(\d{2}))(?:s|秒)?",
                         re.sub(r"\s", "", str(text)).lower())
    if match is None:
        return None
    if match.group(1) is not None:
        seconds = int(match.group(1))
    else:
        if int(match.group(3)) >= 60:
            return None
        seconds = int(match.group(2)) * 60 + int(match.group(3))
    return seconds if 0 <= seconds <= 240 else None


def _slot_body_cooldown(image, rect, backend):
    """Read only yellow numeric text inside this weapon's own icon body."""
    x1, y1, x2, y2 = rect
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    readings = []
    for token in _valid_tokens(enlarged, backend, minimum_confidence=.85):
        seconds = _parse_reload_time(token.text)
        if seconds is None:
            continue
        box = np.asarray(token.box, np.float32) / 3
        cx, cy = box.mean(axis=0)
        if not (0 < seconds <= 240 and .15 * crop.shape[1] <= cx <= .85 * crop.shape[1]
                and .15 * crop.shape[0] <= cy <= .85 * crop.shape[0]):
            continue
        bx1, by1 = np.floor(box.min(axis=0)).astype(int)
        bx2, by2 = np.ceil(box.max(axis=0)).astype(int)
        glyph = crop[max(0, by1):min(crop.shape[0], by2), max(0, bx1):min(crop.shape[1], bx2)]
        if glyph.size == 0:
            continue
        hsv = cv2.cvtColor(glyph, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([15, 45, 135]), np.array([42, 255, 255]))
        if np.mean(yellow > 0) >= .08:
            readings.append(seconds)
    return readings[0] if readings and len(set(readings)) == 1 else None


def confirm_torpedo_reload(image, prefire_hud, backend=None):
    """Read fresh reload feedback using this shot's verified slot geometry.

    The previous ready HUD may locate the counter, but cannot authorize a
    shot. The current frame must retain its own selected [3] frame, at the
    same location, and show a yellow countdown inside that weapon body.
    """
    if (image is None or image.size == 0 or backend is None
            or not isinstance(prefire_hud, TorpedoHud) or not prefire_hud.safe_to_fire
            or prefire_hud.frame_shape != image.shape[:2] or prefire_hud.slot_rect is None
            or not math.isfinite(prefire_hud.slot_scale) or not .25 <= prefire_hud.slot_scale <= 3.5):
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = prefire_hud.slot_rect
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    selected = _selected_weapon_frame(image, backend)
    if selected is None or selected[0] != 3:
        return None
    sx1, sy1, sx2, sy2 = selected[1]
    scale = prefire_hud.slot_scale
    if (abs((sx1 + sx2) / 2 - (x1 + x2) / 2) > max(8 * scale, (x2 - x1) * .22)
            or abs(sy1 - y1) > 20 * scale or sy2 < y2 - 10 * scale
            or sx1 > x1 + 8 * scale or sx2 < x2 - 8 * scale):
        return None
    return _slot_body_cooldown(image, prefire_hud.slot_rect, backend)


def _cooldown_seconds(image, backend, slot=None):
    """Read this verified slot's body counter, then its legacy upper counter."""
    if backend is None:
        return None
    if image is None or image.size == 0:
        return None
    slot = _weapon_slot_evidence(image, backend) if slot is None else slot
    if not isinstance(slot, _SlotEvidence) or slot.weapon is not True or slot.rect is None:
        return None
    body_countdown = _slot_body_cooldown(image, slot.rect, backend)
    if body_countdown is not None:
        return body_countdown
    height, width = image.shape[:2]
    x1, y1, x2, y2 = slot.rect
    padding = max(3, round(8 * slot.scale))
    left, right = max(0, x1 - padding), min(width, x2 + padding)
    top = max(0, y1 - round(36 * slot.scale))
    bottom = min(height, y1 + round((y2 - y1) * .10))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    readings = []
    for token in _valid_tokens(crop, backend, minimum_confidence=.70):
        seconds = _parse_reload_time(getattr(token, "text", ""))
        if seconds is None:
            continue
        box = token.box
        # Even a malformed/misplaced token may not reach the [3] keycap.
        center_y = sum(float(point[1]) for point in box) / len(box)
        if center_y > crop.shape[0] * .72:
            continue
        # The main-battery mount numbers can share this upper strip. A grey
        # "2" above the torpedo icon is not a reload timer, even at perfect
        # OCR confidence; require the countdown's actual yellow glyph pixels.
        bx1, by1 = np.floor(np.asarray(box).min(axis=0)).astype(int)
        bx2, by2 = np.ceil(np.asarray(box).max(axis=0)).astype(int)
        glyph = crop[max(0, by1):min(crop.shape[0], by2),
                     max(0, bx1):min(crop.shape[1], bx2)]
        if glyph.size == 0:
            continue
        glyph_hsv = cv2.cvtColor(glyph, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(glyph_hsv, np.array([15, 45, 135]),
                             np.array([42, 255, 255]))
        if np.mean(yellow > 0) < .08:
            continue
        readings.append(seconds)
    return max(readings) if readings else None


def _new_mask(current, previous, lo, hi, *, left, top, right, bottom):
    current_crop = current[top:bottom, left:right]
    previous_crop = previous[top:bottom, left:right]
    if current_crop.size == 0 or current_crop.shape != previous_crop.shape:
        return None
    current_hsv = cv2.cvtColor(current_crop, cv2.COLOR_BGR2HSV)
    previous_hsv = cv2.cvtColor(previous_crop, cv2.COLOR_BGR2HSV)
    current_mask = cv2.inRange(current_hsv, np.array(lo), np.array(hi))
    previous_mask = cv2.inRange(previous_hsv, np.array(lo), np.array(hi))
    previous_mask = cv2.dilate(previous_mask, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(current_mask, cv2.bitwise_not(previous_mask))


@dataclass(frozen=True)
class _Sector:
    apex: tuple[float, float]
    slopes: tuple[float, float]
    intercepts: tuple[float, float]
    top: float
    bottom: float
    score: float

    def axis_x(self, y):
        return float((sum(self.slopes) * y + sum(self.intercepts)) / 2)

    @property
    def angle(self):
        return float(np.degrees(np.arctan(sum(self.slopes) / 2)))


def _sectors(mask, offset, shape, scale, *, white=False):
    """Two long fan boundaries sharing an apex; short cursor strokes fail.

    Loaded broad arcs are deliberately excluded by the small opening angle.
    Pixel geometry comes from rendered lines, independent of UI layout crops.
    """
    if mask is None or mask.size == 0:
        return []
    height, width = shape[:2]
    minimum_length = max(35, 65 * scale, height * .07)
    # Boundaries, rather than arbitrary lines through a filled highlighted fan.
    edges = cv2.Canny(mask, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(18, round(22 * scale)),
                            minLineLength=minimum_length, maxLineGap=max(4, round(12 * scale)))
    if lines is None:
        return []
    models = []
    ox, oy = offset
    # Hough may return a short fragment before the full edge. Keep the longest
    # representative so scale-independent deduplication cannot erase the fan.
    ordered = sorted(lines.reshape(-1, 4),
                     key=lambda line: float(np.hypot(line[2] - line[0], line[3] - line[1])),
                     reverse=True)
    for x1, y1, x2, y2 in ordered:
        x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy
        if abs(y2 - y1) < minimum_length * .60:
            continue
        slope = (x2 - x1) / (y2 - y1)
        if abs(slope) > 1.6:
            continue
        intercept = x1 - slope * y1
        low, high = min(y1, y2), max(y1, y2)
        midpoint = slope * (height * .50) + intercept
        if any(abs(slope - item[0]) < .015 and abs(midpoint - item[5]) < max(4, 5 * scale)
               for item in models):
            continue
        models.append((slope, intercept, low, high, float(np.hypot(x2 - x1, y2 - y1)), midpoint))
    models = sorted(models, key=lambda item: item[4], reverse=True)[:28]
    sectors = []
    for index, first in enumerate(models):
        for second in models[index + 1:]:
            a, b = first[:2]
            c, d = second[:2]
            if abs(a - c) < .015:
                continue
            opening = abs(float(np.degrees(np.arctan(a) - np.arctan(c))))
            if not 1 <= opening <= (32 if white else 25):
                continue
            apex_y = (d - b) / (a - c)
            apex_x = a * apex_y + b
            shared_top, shared_bottom = max(first[2], second[2]), min(first[3], second[3])
            if not (height * .52 <= apex_y <= height * 1.35
                    and width * .10 <= apex_x <= width * .90
                    and shared_bottom - shared_top >= minimum_length * .6
                    and apex_y >= shared_bottom - height * .04):
                continue
            y = shared_top + (shared_bottom - shared_top) * .25
            l, r = sorted((a * y + b, c * y + d))
            if not max(6, 7 * scale) <= r - l <= width * .26:
                continue
            # Filled cursor arrowheads and opaque white explosions are not a
            # prediction sector. A true guide consists of thin fan boundaries.
            if white:
                yy1, yy2 = max(0, round(shared_top - oy)), min(mask.shape[0], round(shared_bottom - oy))
                xx1, xx2 = max(0, round(l - ox)), min(mask.shape[1], round(r - ox))
                local = mask[yy1:yy2, xx1:xx2]
                if local.size == 0 or np.mean(local > 0) > .30:
                    continue
            sectors.append(_Sector((float(apex_x), float(apex_y)), (float(a), float(c)),
                                   (float(b), float(d)), float(shared_top), float(shared_bottom),
                                   first[4] + second[4]))
    if not white:
        # A desired aiming line can run through the highlighted launch fan.
        # It splits that one fan into two narrower mathematical pairs, which
        # are not independent current launch directions. Only a filled outer
        # fan may absorb nested pairs sharing its perspective apex.
        support = {}
        for index, sector in enumerate(sectors):
            readings = []
            for yy in np.linspace(sector.top, sector.bottom, 12):
                x1, x2 = sorted(slope * yy + intercept for slope, intercept in
                                zip(sector.slopes, sector.intercepts))
                xa = max(0, round(x1 + (x2 - x1) * .2 - ox))
                xb = min(mask.shape[1], round(x2 - (x2 - x1) * .2 - ox))
                row = round(yy - oy)
                if 0 <= row < mask.shape[0] and xb > xa:
                    readings.append(float(np.mean(mask[row, xa:xb] > 0)))
            support[index] = float(np.mean(readings)) if readings else 0
        keep = []
        for index, inner in enumerate(sectors):
            nested = False
            for outer_index, outer in enumerate(sectors):
                if (index == outer_index or support[outer_index] < .2
                        or abs(inner.apex[0] - outer.apex[0]) > width * .07
                        or abs(inner.apex[1] - outer.apex[1]) > height * .12
                        or outer.score < inner.score * .85):
                    continue
                overlap_top = max(inner.top, outer.top)
                overlap_bottom = min(inner.bottom, outer.bottom)
                if overlap_bottom - overlap_top < minimum_length * .6:
                    continue
                yy = (overlap_top + overlap_bottom) / 2
                ia, ib = sorted(a * yy + b for a, b in zip(inner.slopes, inner.intercepts))
                oa, ob = sorted(a * yy + b for a, b in zip(outer.slopes, outer.intercepts))
                if (ob - oa >= (ib - ia) * 1.15
                        and oa - 5 * scale <= ia and ib <= ob + 5 * scale):
                    nested = True
                    break
            if not nested:
                keep.append(inner)
        sectors = keep
    return sectors


def _reticle_point(green, offset, shape, scale):
    if green is None or green.size == 0:
        return None
    height, width = shape[:2]
    ox, oy = offset
    xa, ya = max(0, round(width * .36 - ox)), max(0, round(height * .36 - oy))
    xb, yb = min(green.shape[1], round(width * .64 - ox)), min(green.shape[0], round(height * .64 - oy))
    green = green[ya:yb, xa:xb]
    if green.size == 0:
        return None
    ox, oy = ox + xa, oy + ya
    # The aiming circle is independent evidence, but gun mode has one too:
    # it never proves torpedo selection or an aligned launch direction.
    blur = cv2.GaussianBlur(green, (3, 3), 0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, max(12, 30 * scale),
                              param1=80, param2=8, minRadius=max(5, round(9 * scale)),
                              maxRadius=max(7, round(27 * scale)))
    if circles is None:
        return None
    candidates = []
    for x, y, radius in circles[0]:
        x1, y1 = max(0, round(x - radius)), max(0, round(y - radius))
        x2, y2 = min(green.shape[1], round(x + radius + 1)), min(green.shape[0], round(y + radius + 1))
        patch = green[y1:y2, x1:x2]
        yy, xx = np.ogrid[y1:y2, x1:x2]
        inner = (xx - x) ** 2 + (yy - y) ** 2 <= (radius * .48) ** 2
        if not np.any(inner) or np.mean(patch[inner] > 0) > .45:
            continue
        point = (round(x + ox), round(y + oy))
        distance = np.hypot((point[0] - width / 2) / width, (point[1] - height / 2) / height)
        if distance <= .12:
            candidates.append((float(distance), point))
    return min(candidates)[1] if candidates else None


def _exclude_ui(mask, bounds, image, layout):
    left, top, right, bottom = bounds
    for name in ("minimap", "health_text", "fire_dial"):
        x1, y1, x2, y2 = layout.rect(name)
        xa, ya, xb, yb = max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)
        if xa < xb and ya < yb:
            mask[ya - top:yb - top, xa - left:xb - left] = 0
    return mask


def _translucent_prediction_sectors(image, bounds, layout, launch, reference_y, *, movement_only=False):
    """Measure a textured white overlay only beside a verified launch sector.

    A translucent prediction band can have V below 190 and a filled interior.
    Its two long edges still converge at the launch apex. Each edge must show
    consistent whitening of the local background, with retained background
    variation, rather than an opaque pointer, explosion or cloud silhouette.
    This helper supplies geometry; only the caller's green/loaded gate can
    authorize use of that geometry.
    """
    height, width = image.shape[:2]
    left, top, right, bottom = bounds
    scale = max(.3, layout.ui_scale)
    distance_to_apex = launch.apex[1] - reference_y
    if not math.isfinite(distance_to_apex) or distance_to_apex < height * .08:
        return []
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 45)
    edges[hsv[:, :, 1] > 90] = 0
    edges = _exclude_ui(edges, bounds, image, layout)
    # No low-brightness mask is searched over the whole viewport. The expected
    # perspective apex bounds the local corridor, including an unaligned lead.
    yy, xx = np.ogrid[top:bottom, left:right]
    center = (sum(launch.slopes) * yy + sum(launch.intercepts)) / 2
    spread = np.clip((launch.apex[1] - yy) / distance_to_apex, .08, 1.8)
    corridor = ((np.abs(xx - center) <= width * .14 * spread)
                & (yy >= max(top, launch.top - height * .05))
                & (yy <= min(bottom, launch.apex[1] - height * .015)))
    edges[~corridor] = 0
    result = []
    for sector in _sectors(edges, (left, top), image.shape, scale, white=True):
        if (abs(sector.apex[0] - launch.apex[0]) > width * .05
                or abs(sector.apex[1] - launch.apex[1]) > height * .07
                or not sector.top - height * .05 <= reference_y <= sector.bottom):
            continue
        if _translucent_sector_background(image, sector, scale, movement_only=movement_only):
            result.append(sector)
    return result


def _translucent_sector_background(image, sector, scale, *, movement_only=False):
    """Require both outer edges to preserve and whiten the nearby sea texture."""
    height, width = image.shape[:2]
    foreground, background, left_gains, right_gains = [], [], [], []
    left_background, right_background, middle_foreground = [], [], []
    span = sector.bottom - sector.top
    for row in np.linspace(sector.top + span * .12, sector.bottom - span * .12, 17):
        xa, xb = sorted(slope * row + intercept for slope, intercept in
                        zip(sector.slopes, sector.intercepts))
        band_width = xb - xa
        sample_width = max(3, min(round(12 * scale), round(band_width * .10)))
        if band_width < max(10, 9 * scale) or sample_width < 3:
            continue
        # Sample close to each measured boundary. Sampling deep inside a
        # wider, incorrect pair could borrow the real white band's interior.
        centers = (xa + band_width * .12, xb - band_width * .12,
                   xa - band_width * .12, xb + band_width * .12,
                   (xa + xb) / 2)
        y1, y2 = max(0, round(row) - 1), min(height, round(row) + 2)
        values = []
        for center in centers:
            x1, x2 = round(center - sample_width / 2), round(center + sample_width / 2)
            if x1 < 0 or x2 > width or y1 >= y2 or x1 >= x2:
                break
            values.append(np.median(image[y1:y2, x1:x2].reshape(-1, 3), axis=0))
        if len(values) != 5:
            continue
        fi, fj, bi, bj, middle = np.asarray(values, np.float32)
        foreground.append((fi + fj) / 2)
        background.append((bi + bj) / 2)
        left_gains.append(fi - bi)
        right_gains.append(fj - bj)
        left_background.append(bi)
        right_background.append(bj)
        middle_foreground.append(middle)
    if len(foreground) < 8:
        return False
    fg, bg = np.asarray(foreground), np.asarray(background)
    lg, rg = np.asarray(left_gains), np.asarray(right_gains)
    # Two borders must independently brighten every colour channel. Neutral
    # paint with no original texture, or a single white wake line, is rejected.
    paired = (np.min(lg, axis=1) >= 3) & (np.min(rg, axis=1) >= 3)
    if np.mean(paired) < .60:
        return False
    left_alpha = np.median(lg[paired] / np.maximum(255 - np.asarray(left_background)[paired], 1), axis=0)
    right_alpha = np.median(rg[paired] / np.maximum(255 - np.asarray(right_background)[paired], 1), axis=0)
    # Both edges must independently show comparable neutral transparency.
    # A green boundary paired with one white boundary fails this check even
    # when changing sea texture makes both averaged strips slightly brighter.
    mismatch = float(np.max(np.abs(left_alpha - right_alpha)))
    # A green fan composited beside the white lead can reduce one edge's
    # measured whitening. An unavailable orange bearing may retain such a
    # pair strictly for steering, only when both edges remain strongly neutral.
    # A frame that can authorize firing keeps the original equal-alpha check.
    neutral_movement_pair = (movement_only
                             and np.ptp(left_alpha) <= .06 and np.ptp(right_alpha) <= .06)
    if (np.min(left_alpha) < .08 or np.min(right_alpha) < .08
            or np.max(left_alpha) > .55 or np.max(right_alpha) > .55
            or mismatch > (.16 if neutral_movement_pair else .12)
            or np.ptp(left_alpha) > .16 or np.ptp(right_alpha) > .16):
        return False
    # Separate white wedges can share an apex and yield a plausible outer
    # edge pair. Their unpainted gap must not become a central prediction.
    # Most rows need matching whitening through the middle of this band.
    middle_alpha = ((np.asarray(middle_foreground)[paired] - bg[paired])
                    / np.maximum(255 - bg[paired], 1))
    edge_alpha = (left_alpha + right_alpha) / 2
    continuous = ((np.min(middle_alpha, axis=1) >= .05)
                  & (np.max(middle_alpha, axis=1) <= .60)
                  & (np.max(np.abs(middle_alpha - edge_alpha), axis=1) <= .16))
    if np.mean(continuous) < .60:
        return False
    fg, bg = fg[paired], bg[paired]
    alpha = np.median((fg - bg) / np.maximum(255 - bg, 1), axis=0)
    if (np.min(alpha) < .05 or np.max(alpha) > .55
            or np.ptp(alpha) > .16 or np.median(np.max(fg, axis=1)) >= 233):
        return False
    saturation = (np.max(fg, axis=1) - np.min(fg, axis=1)) / np.maximum(np.max(fg, axis=1), 1)
    if np.median(saturation) > .29:
        return False
    fg_luma, bg_luma = np.mean(fg, axis=1), np.mean(bg, axis=1)
    # Isolated HUD pixels must not create false "texture" in a flat shape.
    # The middle half of samples needs variation in both the background and
    # the translucent foreground, rather than just one outlying green line.
    if (np.ptp(np.percentile(fg_luma, [25, 75])) < 3
            or np.ptp(np.percentile(bg_luma, [25, 75])) < 4):
        return False
    correlation = float(np.corrcoef(fg_luma, bg_luma)[0, 1])
    expected = bg + (255 - bg) * alpha
    if math.isfinite(correlation) and correlation < .45:
        # Shell trails/ship effects can contaminate a few boundary samples.
        # Use one fixed white-compositing model to reject those rows, never
        # search subsets for a favourable correlation. The remaining rows
        # must retain the same independent foreground/background texture.
        textured = np.max(np.abs(fg - expected), axis=1) <= 5
        if np.count_nonzero(textured) < 8 or np.mean(textured) < .60:
            return False
        robust_fg, robust_bg = fg_luma[textured], bg_luma[textured]
        if (np.ptp(np.percentile(robust_fg, [25, 75])) < 3
                or np.ptp(np.percentile(robust_bg, [25, 75])) < 4):
            return False
        correlation = float(np.corrcoef(robust_fg, robust_bg)[0, 1])
    return bool(math.isfinite(correlation) and correlation >= .45
                and np.median(np.abs(fg - expected)) <= 14)


def _loaded_arc(mask, offset, shape, reference, scale):
    """A broad, curved green sea overlay, separate from the narrow aim fan.

    Strip the middle aiming fan and all pixels above the crosshair. Fit each
    remaining thin side arc, so ship HP bars and bottom weapon lines cannot
    masquerade as loaded launchers. No configured reload time grants readiness.
    """
    height, width = shape[:2]
    ox, oy = offset
    rx, ry = reference
    candidate = mask.copy()
    candidate[:max(0, round(ry + height * .06 - oy))] = 0
    candidate[max(0, round(height * .88 - oy)):] = 0
    left = max(0, round(rx - width * .04 - ox))
    right = min(candidate.shape[1], round(rx + width * .04 - ox))
    candidate[:, left:right] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    for label, (x, y, cw, ch, area) in enumerate(stats[1:], 1):
        if (cw < width * .12 or ch < height * .025 or ch > height * .30
                or area / (cw * ch) > .25 or area < cw * .6
                or x + ox > rx + width * .15
                or x + cw + ox < rx - width * .15):
            continue
        component = labels[y:y + ch, x:x + cw] == label
        columns, medians = [], []
        for column in range(cw):
            rows = np.flatnonzero(component[:, column])
            if rows.size:
                columns.append(column)
                medians.append(float(np.median(rows)))
        if len(columns) < cw * .65:
            continue
        horizontal = np.asarray(columns, np.float64)
        vertical = np.asarray(medians, np.float64)
        # Antialiasing can join a short sea-colored or horizontal tail at an
        # arc endpoint. Each retained window still needs a broad actual span.
        for start, finish in ((0, 1), (0, .8), (.2, 1), (.1, .9)):
            span = (cw - 1) * (finish - start)
            # One side of a real broad arc can lose a short segment to its
            # translucent fan/sea boundary (314 px at 2560 px in a live frame).
            # Keep the measured curvature and thinness checks on the entire
            # retained span rather than discarding that remaining side.
            if span < width * .12:
                continue
            keep = (horizontal >= (cw - 1) * start) & (horizontal <= (cw - 1) * finish)
            if np.count_nonzero(keep) < span * .65:
                continue
            u = (horizontal[keep] - (cw - 1) * start) / span * 2 - 1
            v = vertical[keep]
            curved = np.polyfit(u, v, 2)
            straight = np.polyfit(u, v, 1)
            curved_error = float(np.mean(np.abs(np.polyval(curved, u) - v)))
            straight_error = float(np.mean(np.abs(np.polyval(straight, u) - v)))
            if (abs(curved[0]) >= max(4, height * .012)
                    and curved_error <= max(3 * scale, height * .012)
                    and curved_error <= straight_error * .85):
                return True
    return False


def inspect_torpedo_hud(image, before_switch, target_point, backend=None):
    """Keep weapon/selection evidence even when no target or pre-switch exists.

    A loaded broad green arc is only readiness. The actual small launch sector
    and thin white prediction sector must have independently measured axes.
    Their points are projected at the same height for relative mouse steering.
    """
    if image is None or image.size == 0:
        return TorpedoHud(None, False, None)
    slot = _weapon_slot_evidence(image, backend)
    weapon, selected = slot
    cooldown = _cooldown_seconds(image, backend, slot)
    hud = partial(TorpedoHud, slot_rect=slot.rect, slot_scale=slot.scale,
                  frame_shape=tuple(image.shape[:2]))
    if (before_switch is None or before_switch.shape != image.shape
            or target_point is None):
        return hud(cooldown, False, None, weapon_is_torpedo=weapon,
                          selected=selected)
    height, width = image.shape[:2]
    layout = detect_hud_layout(image)
    scale = max(.3, layout.ui_scale)
    target_x, target_y = (int(value) for value in target_point)
    if not (.10 * width < target_x < .90 * width and .18 * height < target_y < .87 * height):
        return hud(cooldown, False, None, weapon_is_torpedo=weapon,
                          selected=selected)
    left, top, right, bottom = round(width * .10), round(height * .18), round(width * .90), round(height * .90)
    bounds = (left, top, right, bottom)
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    green_current = _exclude_ui(cv2.inRange(hsv, np.array([37, 75, 80]),
                                           np.array([94, 255, 255])), bounds, image, layout)
    green_new = _exclude_ui(_new_mask(image, before_switch, (37, 75, 80), (94, 255, 255),
                                     left=left, top=top, right=right, bottom=bottom),
                           bounds, image, layout)
    # Selection can already be active before the bot inspects the frame.
    # Verified slot identity allows current evidence; delta is auxiliary only.
    green_ready = green_current if weapon is True and selected else green_new
    reticle = _reticle_point(green_current, (left, top), image.shape, scale)
    blocked_bearing = False
    if weapon is True and selected:
        # A loaded launcher aimed through the ship's bow has an orange circle
        # and line. It can locate the camera for movement towards the enemy;
        # green sector / white prediction geometry may still guide movement,
        # but the orange circle must independently prohibit firing.
        orange = _exclude_ui(cv2.inRange(hsv, np.array([9, 65, 90]),
                                         np.array([34, 255, 255])), bounds, image, layout)
        orange_reticle = _reticle_point(orange, (left, top), image.shape, scale)
        if orange_reticle is not None:
            def central_distance(point):
                return float(np.hypot((point[0] - width / 2) / width,
                                      (point[1] - height / 2) / height))
            # A green ally's label farther up the viewport must not hide the
            # actual central orange aiming circle. Prefer that closer circle
            # only near the camera centre, keeping unrelated warnings out.
            if central_distance(orange_reticle) <= .04:
                # Orange is independent blocking evidence even if a green
                # ring happens to share its centre. Camera-anchor selection
                # must not suppress a known unavailable firing bearing.
                blocked_bearing = True
                if (reticle is None
                        or central_distance(orange_reticle) + .004 < central_distance(reticle)):
                    reticle = orange_reticle
    hud = partial(hud, bearing_available=not blocked_bearing)
    reference_y = reticle[1] if reticle is not None else target_y
    loaded = _loaded_arc(green_ready, (left, top), image.shape,
                         (reticle[0] if reticle else target_x, reference_y), scale)
    if weapon is not True or not selected:
        return hud(cooldown, loaded, None, weapon_is_torpedo=weapon,
                          selected=selected, reticle_point=reticle)
    launch_sectors = _sectors(green_current, (left, top), image.shape, scale)
    launch_sectors = [sector for sector in launch_sectors if
                      sector.top - height * .10 <= reference_y <= sector.bottom
                      and abs(sector.axis_x(reference_y) - (reticle[0] if reticle else target_x)) <= width * .12]
    if not launch_sectors:
        return hud(cooldown, loaded, None, weapon_is_torpedo=weapon,
                          selected=selected, reticle_point=reticle)
    launch_sectors.sort(key=lambda sector: (
        abs(sector.axis_x(reference_y) - (reticle[0] if reticle else target_x)), -sector.score))
    launch = launch_sectors[0]
    if (len(launch_sectors) > 1
            and abs(launch_sectors[1].axis_x(reference_y) - launch.axis_x(reference_y)) > max(18, 20 * scale)
            and launch_sectors[1].score >= launch.score * .85):
        return hud(cooldown, loaded, None, weapon_is_torpedo=weapon,
                          selected=selected, reticle_point=reticle)
    aim = (round(launch.axis_x(reference_y)), int(reference_y))
    white = _exclude_ui(cv2.inRange(hsv, np.array([0, 0, 190]), np.array([179, 55, 255])),
                        bounds, image, layout)
    prediction_sectors = _sectors(white, (left, top), image.shape, scale, white=True)
    def prediction_candidates(sectors):
        matches = []
        for sector in sectors:
            # Native antialiasing can move the first visible white edge a
            # few pixels below the reticle. A small upward axis projection
            # preserves the shared reference height; all sector/background
            # and apex checks still apply before this geometry is used.
            if not sector.top - height * .05 <= reference_y <= sector.bottom:
                continue
            if (abs(sector.apex[0] - launch.apex[0]) > width * .07
                    or abs(sector.apex[1] - launch.apex[1]) > height * .10):
                continue
            prediction = (round(sector.axis_x(reference_y)), int(reference_y))
            if math.dist(prediction, (target_x, target_y)) > width * .18:
                continue
            matches.append((math.dist(prediction, (target_x, target_y)), -sector.score,
                            prediction, sector))
        return matches

    candidates = prediction_candidates(prediction_sectors)
    # Existing bright, thin prediction geometry has priority. Faint filled
    # geometry cannot substitute for a loaded green launcher or a cooldown.
    if not candidates and loaded and not (cooldown is not None and cooldown > 0):
        candidates = prediction_candidates(_translucent_prediction_sectors(
            image, bounds, layout, launch, reference_y, movement_only=blocked_bearing))
    if not candidates:
        return hud(cooldown, loaded, None, weapon_is_torpedo=weapon,
                          selected=selected, aim_point=aim, reticle_point=reticle)
    candidates.sort(key=lambda item: item[:2])
    if (len(candidates) > 1 and math.dist(candidates[0][2], candidates[1][2]) > max(18, 20 * scale)
            and -candidates[1][1] >= -candidates[0][1] * .85):
        return hud(cooldown, loaded, None, weapon_is_torpedo=weapon,
                          selected=selected, aim_point=aim, reticle_point=reticle)
    _, _, lead, prediction_sector = candidates[0]
    return hud(cooldown, loaded, lead, weapon_is_torpedo=weapon,
                      selected=selected, aim_point=aim,
                      alignment_error_px=float(abs(aim[0] - lead[0])),
                      alignment_tolerance_px=float(max(5, 8 * scale)),
                      alignment_error_degrees=float(abs(launch.angle - prediction_sector.angle)),
                      reticle_point=reticle)
