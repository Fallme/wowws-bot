"""Measure independent HUD groups; unverified groups keep their safe crops.

The speed dial and bottom key row have a pixel size independent of the client
resolution.  Their measured transforms must not be inferred from the sea,
viewport ships, or the minimap. One weakly held frame is cached for all readers.
"""

from dataclasses import dataclass, field
import math
import weakref

import cv2
import numpy as np

from core.minimap_geometry import minimap_grid_bounds


_REFERENCE_SIZE = (2560, 1494)
_RECTS = {
    "health_bar": (.014, .785, .1015, .817, "left"),
    "health_text": (0, .75, .19, .87, "left"),
    "speed": (.105, .83, .205, .96, "left"),
    "autopilot": (.10, .79, .22, .88, "left"),
    "autopilot_text": (0, .68, .28, .97, "left"),
    "fire_dial": (.020, .840, .120, .940, "left"),
    "flood_left": (.025, .790, .120, .875, "left"),
    "player": (0, .70, .25, 1, "left"),
    "fire_ship": (.15, .55, .73, .90, "viewport"),
    "fire_status": (.44, .76, .53, .84, "center"),
    "flood_center": (.47, .815, .54, .905, "center"),
    "rudder_q": (.42, .70, .49, .84, "center"),
    "rudder_e": (.51, .70, .58, .84, "center"),
    "consumables": (.34, .78, .68, 1, "center"),
    "clock": (.80, 0, 1, .10, "right_top"),
    "score_clock": (.42, .02, .58, .12, "right_top"),
    "minimap": (.73, .565, 1, 1, "right_bottom"),
}


@dataclass(frozen=True)
class HudLayout:
    width: int
    height: int
    transforms: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)

    @property
    def ui_scale(self):
        return self.scales.get("left", self.scales.get("center", min(
            self.width / 2560, self.height / 1494)))

    def source(self, name):
        group = _RECTS[name][4]
        if name == "clock" and name in self.overrides:
            return "inferred"
        return "anchor" if name in self.overrides or group in self.transforms else "fallback"

    def scale(self, name):
        group = _RECTS[name][4]
        return self.scales.get(group, self.ui_scale if self.transforms else math.sqrt(
            max(self.width * self.height / (2560 * 1494), .20)))

    def _clip(self, rect):
        x1, y1, x2, y2 = (round(value) for value in rect)
        return (max(0, min(self.width, x1)), max(0, min(self.height, y1)),
                max(0, min(self.width, x2)), max(0, min(self.height, y2)))

    def map_rect(self, rect, group="center"):
        """Map a reference 2560x1494 pixel rectangle into one HUD group."""
        if group in self.transforms:
            scale, tx, ty = self.transforms[group]
            x1, y1, x2, y2 = rect
            return self._clip((x1 * scale + tx, y1 * scale + ty,
                               x2 * scale + tx, y2 * scale + ty))
        return self._clip((rect[0] * self.width / 2560, rect[1] * self.height / 1494,
                           rect[2] * self.width / 2560, rect[3] * self.height / 1494))

    def rect(self, name):
        if name in self.overrides:
            return self._clip(self.overrides[name])
        left, top, right, bottom, group = _RECTS[name]
        return self.map_rect((left * 2560, top * 1494, right * 2560, bottom * 1494), group)

    def crop(self, image, name):
        x1, y1, x2, y2 = self.rect(name)
        return image[y1:y2, x1:x2]

    def diagnostics(self):
        return {"ui_scale": round(float(self.ui_scale), 3),
                "groups": {group: ("anchor" if group in self.transforms else
                           "inferred" if group == "right_top" and "clock" in self.overrides else "fallback")
                           for group in ("left", "center", "right_top", "right_bottom")},
                "sources": {name: self.source(name) for name in _RECTS},
                "rects": {name: self.rect(name) for name in _RECTS}}


def _hp_support(image, cx, cy, radius):
    """Require the ship status above the dial, rather than a circle in water."""
    height, width = image.shape[:2]
    crop = image[max(0, round(cy - 1.80 * radius)):max(0, round(cy - .90 * radius)),
                 max(0, round(cx - 1.55 * radius)):min(width, round(cx + 2 * radius))]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # The warm/green own-ship silhouette has a saturated fill and a stable
    # left edge about one dial radius left of its centre. Sea reflections and
    # HP digits alone must not corroborate a smaller circle in the same band.
    colored = cv2.inRange(hsv, np.array([0, 110, 100]), np.array([95, 255, 255]))
    _, _, stats, _ = cv2.connectedComponentsWithStats(colored)
    crop_left = max(0, round(cx - 1.55 * radius))
    return any(.55 <= (cx - (x + crop_left)) / radius <= 1.35
               and x > 2 and x + cw < crop.shape[1] - 2
               and cw >= .15 * radius and max(3, .06 * radius) <= ch <= .55 * radius
               and 1.3 <= cw / ch <= 20 and .18 <= area / (cw * ch) <= .95
               and area >= .003 * radius ** 2
               for x, _, cw, ch, area in stats[1:])


def _left_anchor(image):
    height, width = image.shape[:2]
    top, right = round(height * .60), round(width * .32)
    roi = image[top:, :right]
    if roi.size == 0 or min(roi.shape[:2]) < 40:
        return None
    ratio = min(1, 360 / max(roi.shape[:2]))
    small = cv2.resize(roi, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    if gray.std() < 8:
        return None
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2,
                              max(12, height * .045 * ratio), param1=90, param2=25,
                              minRadius=max(9, round(height * .025 * ratio)),
                              maxRadius=max(10, round(height * .15 * ratio)))
    if circles is None:
        return None
    edges = cv2.dilate(cv2.Canny(gray, 45, 90), np.ones((3, 3), np.uint8))
    angles = np.linspace(0, 2 * np.pi, 144, endpoint=False)
    accepted = []
    for x, y, r in circles[0][:80]:
        cx, cy, radius = x / ratio, y / ratio + top, r / ratio
        if not (.65 <= cx / radius <= 3.5 and cy >= height * .70
                and cy + radius <= height + .06 * radius):
            continue
        xs = np.clip(np.rint(x + r * np.cos(angles)).astype(int), 0, edges.shape[1] - 1)
        ys = np.clip(np.rint(y + r * np.sin(angles)).astype(int), 0, edges.shape[0] - 1)
        coverage = float(np.mean(edges[ys, xs] > 0))
        if coverage >= .86 and _hp_support(image, cx, cy, radius):
            accepted.append((coverage, radius, cx, cy))
    if not accepted:
        return None
    # Dial interiors/sea wakes yield smaller concentric candidates. Once the
    # own-ship silhouette agrees, the outer ring supplies the actual UI size.
    _, radius, cx, cy = max(accepted, key=lambda item: (item[1], item[0]))
    return cx, cy, radius


def _center_anchor(image, scale):
    """At least four equally sized rectangular keycaps on one spaced row."""
    height, width = image.shape[:2]
    top, left = round(height * .78), round(width * .25)
    roi = image[top:, left:round(width * .78)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cv2.Canny(gray, 80, 180), cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    # RETR_LIST supplies both edges of a keycap. Keep its outer border, whose
    # physical size is the scale anchor, rather than the first inner contour.
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3], reverse=True)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if not (7 <= cw <= min(80, width * .065)
                and 7 <= ch <= min(80, height * .095) and .80 <= cw / ch <= 1.25):
            continue
        polygon = cv2.approxPolyDP(contour, .035 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or cv2.contourArea(contour) / (cw * ch) < .65:
            continue
        box = (x + left, y + top, cw, ch)
        if not any(abs(box[0] - old[0]) <= 3 and abs(box[1] - old[1]) <= 3 for old in boxes):
            boxes.append(box)
    candidates = []
    for _, y, cw, ch in boxes:
        row = sorted((b for b in boxes if abs(b[1] - y) <= max(3, ch * .15)
                      and abs(b[2] - cw) <= cw * .20 and abs(b[3] - ch) <= ch * .20))
        if len(row) < 4:
            continue
        gaps = np.diff([b[0] for b in row])
        if not (np.all(gaps >= cw * 2) and np.all(gaps <= cw * 10)
                and (row[-1][0] - row[0][0]) >= cw * 8):
            continue
        # Irregular spacing is normal between weapon and consumable groups;
        # the keycaps themselves must still form one compact bottom HUD row.
        candidates.append((len(row), y, row))
    if not candidates:
        return None
    _, _, row = max(candidates, key=lambda item: (item[0], item[1]))
    key_size = float(np.median([max(b[2], b[3]) for b in row]))
    center_x = (row[0][0] + row[-1][0] + row[-1][2]) / 2
    center_y = float(np.median([b[1] + b[3] / 2 for b in row]))
    return center_x, center_y, key_size / 23


_cached_frame = None
_cached_layout = None


def _grid_search_worthwhile(image):
    """Cheap line evidence avoids combinatorial grid fitting on port artwork."""
    height, width = image.shape[:2]
    side = min(height, round(width * .32))
    if side < 40:
        return False
    crop = image[-side:, -side:]
    ratio = min(1, 300 / side)
    small = cv2.resize(crop, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 12, 36)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=35,
                            minLineLength=small.shape[0] * .30, maxLineGap=4)
    if lines is None:
        return False
    vertical = horizontal = 0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        vertical += abs(x2 - x1) <= 1
        horizontal += abs(y2 - y1) <= 1
    return vertical >= 4 and horizontal >= 4


def detect_hud_layout(image):
    """Return measured bounds; cached by image identity without retaining pixels."""
    global _cached_frame, _cached_layout
    if image is None or image.size == 0:
        return HudLayout(0, 0)
    if _cached_frame is not None and _cached_frame() is image:
        return _cached_layout
    height, width = image.shape[:2]
    transforms, overrides, scales = {}, {}, {}
    left = _left_anchor(image)
    if left is not None:
        cx, cy, radius = left
        scale = radius / 125
        transforms["left"] = (scale, cx - 150 * scale, cy - 1346 * scale)
        scales["left"] = scale
        overrides.update({
            "health_text": (cx - 1.25 * radius, cy - 1.8 * radius,
                            cx + 2.6 * radius, cy - .9 * radius),
            "health_bar": (cx - .90 * radius, cy - 1.7 * radius,
                           cx + .90 * radius, cy - 1.27 * radius),
            "speed": (cx + 1.20 * radius, cy - .75 * radius,
                      cx + 3.0 * radius, cy + .50 * radius),
            "fire_dial": (cx - .80 * radius, cy - .70 * radius,
                          cx + .85 * radius, cy + .70 * radius),
            "flood_left": (cx - .90 * radius, cy - 1.14 * radius,
                           cx + 1.4 * radius, cy - .28 * radius),
        })
    center = _center_anchor(image, scales.get("left", min(width / 2560, height / 1494)))
    if center is not None:
        cx, cy, scale = center
        transforms["center"] = (scale, cx - 1280 * scale, cy - 1477 * scale)
        scales["center"] = scale
    # Keep the inferred clock anchored to the right edge, outside capture/base
    # timers at screen centre. The source explicitly distinguishes inference.
    measured_scale = scales.get("left", scales.get("center"))
    if measured_scale is not None:
        overrides["clock"] = (max(math.ceil(width * .67), width - 512 * measured_scale), 0,
                              width, min(height * .18, 149 * measured_scale))
    bounds = minimap_grid_bounds(image) if (left is not None or center is not None
                                           or _grid_search_worthwhile(image)) else None
    if bounds is not None:
        overrides["minimap"] = bounds
        scales["right_bottom"] = (bounds[2] - bounds[0]) / 690
        transforms["right_bottom"] = (scales["right_bottom"],
                                      bounds[0] - 1869 * scales["right_bottom"],
                                      bounds[1] - 844 * scales["right_bottom"])
    if bounds is not None:
        overrides["fire_ship"] = (width * .15, height * .55,
                                  min(width * .73, bounds[0]), height * .90)
    layout = HudLayout(width, height, transforms, overrides, scales)
    _cached_frame, _cached_layout = weakref.ref(image), layout
    return layout
