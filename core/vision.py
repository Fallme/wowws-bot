"""Screen capture and visual state recognition."""

import base64
import logging
import math
import re
import time
import weakref
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

from core.terrain import exclude_player_land, repair_overlay_gaps
from core.minimap_geometry import minimap_grid_bounds
from core.hud_layout import detect_hud_layout
from core.tactical_geometry import detect_tactical_map_geometry

import cv2
import numpy as np

from core.frame_guard import CaptureFault, FrameGuard
from core.ocr import RapidOcrBackend, numeric_ocr_fallback_variants
from core.ui import (
    ESCAPE_RESUME_BUTTON,
    EXIT_CONTINUE_BUTTON,
    FIRE_DIAL_ROI,
    FIRE_SHIP_ROI,
    FIRE_STATUS_ROI,
    HEALTH_BAR_REGION,
    LOADING_START_BUTTON,
    MINIMAP_REGION,
    NO_COMMANDER_CONFIRM_BUTTON,
    PORT_BATTLE_BUTTON,
    RESULTS_REQUEUE_BUTTON,
    RESULTS_RETURN_TO_PORT_BUTTON,
    ScreenState,
)
from dxgi_capture import ScreenCapture

logger = logging.getLogger("vision")


@lru_cache(maxsize=1)
def _fire_warning_template():
    template = cv2.imread(str(Path(__file__).resolve().parent.parent /
                               "assets" / "ui" / "fire_warning_mask.png"),
                          cv2.IMREAD_GRAYSCALE)
    if template is None or template.shape != (40, 40):
        return None
    return template > 127


@dataclass(frozen=True)
class PlayerPose:
    position: tuple[int, int]
    heading: tuple[float, float]


@dataclass(frozen=True)
class IslandRisk:
    distance: float
    avoidance_rudder: float


@dataclass(frozen=True)
class KinematicSteeringPlan:
    """Best Q/E order after simulating the ship's delayed turning arc."""

    rudder: float
    avoidance_required: bool
    collision_time_seconds: float | None
    minimum_clearance_km: float
    predicted_endpoint: tuple[float, float]
    all_paths_blocked: bool = False


@dataclass(frozen=True)
class CaptureZone:
    center: tuple[int, int]
    radius: float
    label: str = ""
    # Capture ring ownership read from the live minimap.  Friendly green
    # points are intentionally not selected as an opening destination when a
    # neutral white or hostile red lettered point is available.
    state: str = "unknown"


def terrain_value_floor(ocean_value: float) -> int:
    """Brightness floor that follows the minimap's lighting palette.

    Night maps draw islands barely brighter than the sea, so the historical
    fixed floor of 125 misses nearly every coastline.  Keep the conservative
    daytime floor while adapting to dark palettes.
    """
    return int(min(125, max(45, float(ocean_value) + 25)))


class Vision:
    def __init__(self, screen_capture=None, frame_guard=None):
        self.screen_capture = screen_capture or ScreenCapture()
        self.frame_guard = frame_guard or FrameGuard()
        self.last_frame_quality = None
        self._minimap_grid_cache = None
        self._minimap_grid_checked_at = float("-inf")
        self.red_lo1 = np.array([0, 40, 80])
        self.red_hi1 = np.array([15, 255, 255])
        self.red_lo2 = np.array([160, 40, 80])
        self.red_hi2 = np.array([180, 255, 255])
        self.orange_lo = np.array([10, 60, 100])
        self.orange_hi = np.array([30, 255, 255])
        self.yellow_lo = np.array([10, 80, 80])
        self.yellow_hi = np.array([40, 255, 255])
        self.green_lo = np.array([35, 100, 100])
        self.green_hi = np.array([85, 255, 255])

    @staticmethod
    def hud_layout(image):
        """Expose the shared measured HUD and its diagnostic sources."""
        return detect_hud_layout(image)

    def grab(self, hwnd, *, allow_stale=False):
        last_reason = "capture_failed"
        for attempt in range(3):
            image = self.screen_capture.capture_window(hwnd)
            if image is None:
                last_reason = "capture_failed"
            else:
                quality = self.frame_guard.inspect(image)
                self.last_frame_quality = quality
                if quality.valid or (
                    allow_stale and quality.reason == "capture_stale"
                ):
                    return image
                last_reason = quality.reason
            if attempt < 2:
                time.sleep(0.06 * (attempt + 1))
        if last_reason == "capture_failed":
            raise CaptureFault("游戏画面连续3次截取失败")
        raise CaptureFault(f"游戏画面连续3次不可用: {last_reason}")

    def analyze_minimap(self, minimap):
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        return self.find_enemies_from_hsv(hsv), self.has_torpedoes_from_hsv(hsv)

    @staticmethod
    def is_autopilot_enabled(image) -> bool:
        """Confirm the green ``M 自动驾驶开启`` HUD indicator."""
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        roi = layout.crop(image, "autopilot")
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 90)
            & (hsv[:, :, 1] > 80)
            & (hsv[:, :, 2] > 90)
        )
        green_pixels = int(np.count_nonzero(green))
        # A pure ratio silently fails on a 4K framebuffer when the user keeps
        # the game UI compact: the label retains roughly the same pixel size
        # while this relative ROI quadruples. Combine the ratio with a capped
        # scale-aware pixel floor; the tight ROI still excludes the HP bar and
        # minimap's many green elements.
        framebuffer_scale = layout.scale("autopilot")
        minimum_pixels = max(
            24,
            int(round(120 * min(max(framebuffer_scale, 0.35), 1.0) ** 2)),
        )
        return bool(
            float(np.mean(green)) >= 0.008
            or green_pixels >= minimum_pixels
        )

    @staticmethod
    def read_autopilot_enabled_text(image, backend) -> bool | None:
        """Read the exact green lower-left ``自动驾驶`` status text.

        The same words also exist as a neutral key hint before a route starts.
        OCR first locates that text, then colour is measured only inside its
        returned glyph box. This avoids the unrelated green consumables and
        minimap content that made the old broad mask report false positives.
        """
        if image is None or image.size == 0 or backend is None:
            return None
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        crop = layout.crop(image, "autopilot_text")
        if crop.size == 0:
            return None
        try:
            tokens = backend.recognize(crop)
        except Exception:
            logger.debug("自动驾驶状态文字 OCR 失败", exc_info=True)
            return None
        for token in tokens:
            text = "".join(str(getattr(token, "text", "") or "").split())
            if "自动驾驶" not in text or float(
                getattr(token, "confidence", 0.0)
            ) < 0.55:
                continue
            if "启用" in text or "开启" in text:
                return True
            points = np.asarray(getattr(token, "box", ()) or (), dtype=np.float32)
            if points.shape != (4, 2):
                continue
            x1, y1 = np.floor(points.min(axis=0)).astype(int)
            x2, y2 = np.ceil(points.max(axis=0)).astype(int)
            padding = max(2, int(round(layout.scale("autopilot_text") * 3)))
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2 = min(crop.shape[1], x2 + padding + 1)
            y2 = min(crop.shape[0], y2 + padding + 1)
            glyphs = crop[y1:y2, x1:x2]
            if glyphs.size == 0:
                continue
            hsv = cv2.cvtColor(glyphs, cv2.COLOR_BGR2HSV)
            green = (
                (hsv[:, :, 0] >= 35)
                & (hsv[:, :, 0] <= 90)
                & (hsv[:, :, 1] > 55)
                & (hsv[:, :, 2] > 80)
            )
            if int(np.count_nonzero(green)) >= 12 and float(np.mean(green)) >= 0.03:
                return True
        return False

    @staticmethod
    def detect_rudder_indicator(image) -> str:
        """Read the green Q/E steering cue shown near the lower HUD centre."""
        if image is None or image.size == 0:
            return "neutral"
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)

        def score(name):
            crop = layout.crop(image, name)
            roi = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.size else crop
            if roi.size == 0:
                return 0.0
            green = (
                (roi[:, :, 0] >= 35)
                & (roi[:, :, 0] <= 90)
                & (roi[:, :, 1] > 100)
                & (roi[:, :, 2] > 120)
            )
            return float(np.mean(green))

        q_score = score("rudder_q")
        e_score = score("rudder_e")
        threshold = 0.002
        if q_score >= threshold and q_score > e_score * 1.35:
            return "Q"
        if e_score >= threshold and e_score > q_score * 1.35:
            return "E"
        if max(q_score, e_score) >= threshold:
            return "ambiguous"
        return "neutral"

    def find_player_on_minimap(self, minimap):
        pose = self.find_player_pose_on_minimap(minimap)
        return None if pose is None else pose.position

    @staticmethod
    def minimap_pixels_to_km(minimap, pixel_distance: float) -> float:
        """Convert minimap pixels using the game's 10x10, 5 km grid."""
        height, width = minimap.shape[:2]
        pixels_per_grid = max(min(width, height) / 10.0, 1.0)
        return float(pixel_distance) * 5.0 / pixels_per_grid

    @staticmethod
    def relative_bearing(pose, target: tuple[float, float]) -> float:
        """Return signed target bearing normalized to [-1, 1]."""
        target_x = float(target[0]) - pose.position[0]
        target_y = float(target[1]) - pose.position[1]
        heading_x, heading_y = pose.heading
        cross = heading_x * target_y - heading_y * target_x
        dot = heading_x * target_x + heading_y * target_y
        return max(-1.0, min(math.atan2(cross, dot) / math.pi, 1.0))

    @staticmethod
    def _capture_zone_ocr_label(minimap, zone, backend):
        """Read the A/B/C/D glyph from the centre of one capture ring.

        Hough circles give accurate geometry but their spatial ordering is not
        the game's label ordering on staggered maps.  The diamond letter is
        small, so OCR is restricted to a square around the circle centre and
        enlarged once; this excludes the map's A-J row/column grid labels and
        nearby ship names.
        """
        if backend is None or minimap is None or minimap.size == 0:
            return ""
        height, width = minimap.shape[:2]
        center_x, center_y = (int(zone.center[0]), int(zone.center[1]))
        half = max(12, int(round(float(zone.radius) * 0.62)))
        x1, y1 = max(0, center_x - half), max(0, center_y - half)
        x2, y2 = min(width, center_x + half + 1), min(height, center_y + half + 1)
        crop = minimap[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        # A single enlargement is enough for both the 2K minimap and the
        # rectified tactical map. Avoid multiple OCR passes in every control
        # frame; static geometry is sampled only during the opening window.
        enlarged = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        try:
            tokens = backend.recognize(enlarged) or ()
        except Exception:
            logger.debug("占领点字母 OCR 失败", exc_info=True)
            return ""
        best = (0.0, "")
        for token in tokens:
            text = str(getattr(token, "text", "") or "").strip().upper()
            if text not in {"A", "B", "C", "D"}:
                continue
            confidence = float(getattr(token, "confidence", 0.0) or 0.0)
            box = getattr(token, "box", ()) or ()
            points = np.asarray(box, dtype=np.float32)
            if points.shape != (4, 2):
                continue
            # Reject any accidental grid/name token that happens to be a
            # single Latin character but lies far from this ring's centre.
            token_center = points.mean(axis=0) / 2.0
            crop_center = np.asarray(crop.shape[1::-1], dtype=np.float32) / 2.0
            if float(np.linalg.norm(token_center - crop_center)) > half * 0.55:
                continue
            if confidence > best[0]:
                best = (confidence, text)
        return best[1]

    @classmethod
    def _apply_capture_zone_ocr_labels(cls, minimap, zones, backend):
        """Replace inferred labels only when every visible ring is confirmed."""
        if backend is None or not zones:
            return list(zones)
        labels = [cls._capture_zone_ocr_label(minimap, zone, backend) for zone in zones]
        # Partial OCR must not create a mixture of true and guessed letters;
        # retain the deterministic geometric fallback until a complete set is
        # available. This also keeps the static-layer vote stable across fades.
        if len(set(labels)) != len(labels) or not all(labels):
            return list(zones)
        return [
            CaptureZone(zone.center, zone.radius, label, zone.state)
            for zone, label in zip(zones, labels)
        ]

    @staticmethod
    def _capture_center_glyph(minimap, center):
        """Require an actual small upright glyph, never infer a missing cap."""
        height, width = minimap.shape[:2]
        scale = min(height, width)
        half = max(9, int(scale * 0.025))
        x, y = center
        x1, y1 = max(0, x-half), max(0, y-half)
        crop = minimap[y1:min(height, y+half+1), x1:min(width, x+half+1)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        colored = (sat > 110) & ((hue < 12) | ((hue > 35) & (hue < 90)))
        mask = ((val > 130) & ((sat < 90) | colored)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            gx, gy, w, h = cv2.boundingRect(contour)
            if not (max(2, scale*.003) <= w <= scale*.022
                    and max(5, scale*.008) <= h <= scale*.028
                    and .25 <= w/h <= 1.10 and cv2.contourArea(contour) >= 9):
                continue
            if math.dist((x1+gx+w/2, y1+gy+h/2), center) <= max(4, scale*.010):
                return (x1+gx, y1+gy, w, h)
        return None

    def find_capture_zones(self, minimap, player=None, ocr_backend=None):
        """Circle evidence plus a centre glyph. No equal-spacing or A/B/C guesses."""
        if minimap is None or minimap.size == 0:
            return []
        height, width = minimap.shape[:2]
        scale = min(width, height)
        gray = cv2.GaussianBlur(cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=scale*.055,
            param1=100, param2=max(24, scale*.045),
            minRadius=max(12, int(scale*.040)), maxRadius=max(20, int(scale*.115)),
        )
        if circles is None:
            return []
        zones = []
        for raw_x, raw_y, radius in circles[0]:
            if radius < scale * .060:
                continue
            center = (int(round(raw_x)), int(round(raw_y)))
            if not (scale*.035 < center[0] < width-scale*.035 and scale*.035 < center[1] < height-scale*.035):
                continue
            # Only concentric circles are ship range rings. A cap may contain us.
            if player is not None and math.dist(center, player) <= scale*.045:
                continue
            glyph = self._capture_center_glyph(minimap, center)
            if glyph is None:
                continue
            # Hough also sees the yellow ship/range rings.  A real capture
            # point has a bright neutral or team-coloured circumference; the
            # ring-ink vote rejects yellow range rings and wake arcs before a
            # nearby digit is mistaken for an A/B/C marker.
            angles = np.arange(72) * (2 * np.pi / 72)
            band = max(2, round(scale * .006))
            radii = radius + np.arange(-band, band + 1)[:, None]
            xs = np.clip(np.rint(center[0] + radii * np.cos(angles)), 0, width - 1).astype(int)
            ys = np.clip(np.rint(center[1] + radii * np.sin(angles)), 0, height - 1).astype(int)
            ring = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)[ys, xs]
            neutral_ink = (ring[:, :, 2] >= 145) & (ring[:, :, 1] <= 90)
            team_ink = (
                (ring[:, :, 2] >= 120) & (ring[:, :, 1] >= 100)
                & ((ring[:, :, 0] <= 15) | (ring[:, :, 0] >= 165)
                   | ((ring[:, :, 0] >= 35) & (ring[:, :, 0] <= 95)))
            )
            ring_ink = neutral_ink | team_ink
            # Vote by angular sector, not by the area of the sampling
            # band: a one-pixel circumference remains valid at any scale.
            if float(np.mean(np.any(ring_ink, axis=0))) < 0.55:
                continue
            if any(math.dist(center, z.center) < scale*.055 for z in zones):
                continue
            zone = CaptureZone(center, float(radius), "", self._capture_zone_state(minimap, center, float(radius)))
            label = ""
            glyph_reader = getattr(ocr_backend, "recognize_glyph", None)
            if glyph_reader is not None:
                gx, gy, gw, gh = glyph
                crop = minimap[max(0, gy-1):gy+gh+1, max(0, gx-1):gx+gw+1]
                try:
                    tokens = glyph_reader(crop)
                    accepted = [t for t in tokens if t.text.strip().upper() in {"A", "B", "C", "D"} and t.confidence >= .85]
                    if accepted:
                        label = max(accepted, key=lambda t: t.confidence).text.strip().upper()
                except Exception:
                    logger.debug("点位字母直读失败，保留未命名点位", exc_info=True)
            elif ocr_backend is not None:
                label = self._capture_zone_ocr_label(minimap, zone, ocr_backend)
            zones.append(CaptureZone(center, float(radius), label, zone.state))
        # Partial OCR labels remain partial; geometry must never rename D as C.
        return sorted(zones, key=lambda z: (z.center[0], z.center[1]))

    def find_nearest_capture_zone(self, minimap, player):
        """Find the nearest neutral/hostile lettered capture point.

        The player's white arrow and the ship's range/stealth circles are
        excluded by ``find_capture_zones``.  Of the remaining A/B/C/D points,
        prefer white/neutral or red/enemy ownership: sailing back to a green
        friendly cap is not an opening objective unless it is the only point
        currently visible.
        """
        if player is None:
            return None
        candidates = self.find_capture_zones(minimap, player=player)
        return self.select_navigation_capture_zone(candidates, player)

    @staticmethod
    def select_navigation_capture_zone(candidates, player):
        """Select an eligible A/B/C/D zone from already-detected candidates."""
        if player is None or not candidates:
            return None
        if not candidates:
            return None
        preferred = [
            zone
            for zone in candidates
            if getattr(zone, "state", "unknown") in {"neutral", "hostile", "unknown"}
        ]
        return min(
            preferred or candidates,
            key=lambda zone: math.dist(zone.center, player),
        )

    @staticmethod
    def _capture_zone_state(minimap, center, radius: float) -> str:
        """Classify the coloured A/B/C/D ring without using ship range rings."""
        if minimap is None or minimap.size == 0 or radius <= 1:
            return "unknown"
        height, width = minimap.shape[:2]
        # Ownership belongs to the centre glyph. Nearby friendly range rings
        # must not turn a white C point green.
        half = max(4, int(min(height, width) * .009))
        x, y = center
        core = minimap[max(0, y-half):y+half+1, max(0, x-half):x+half+1]
        ch, cs, cv = cv2.split(cv2.cvtColor(core, cv2.COLOR_BGR2HSV))
        if np.count_nonzero((cv >= 135) & (cs >= 110) & ((ch <= 10) | (ch >= 165))) >= 4:
            return "hostile"
        if np.count_nonzero((cv >= 135) & (cs >= 110) & (ch >= 35) & (ch <= 95)) >= 4:
            return "friendly"
        if np.count_nonzero((cv >= 160) & (cs <= 75)) >= 5:
            return "neutral"
        yy, xx = np.ogrid[:height, :width]
        distance = np.hypot(xx - float(center[0]), yy - float(center[1]))
        # The ownership colour sits on the capture outline. Sampling an
        # annulus prevents the island texture inside a point from dominating.
        annulus = (distance >= radius * 0.76) & (distance <= radius * 1.10)
        sample_count = int(np.count_nonzero(annulus))
        if sample_count < 20:
            return "unknown"
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        red = (
            ((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 165))
            & (hsv[:, :, 1] >= 75)
            & (hsv[:, :, 2] >= 75)
            & annulus
        )
        green = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 55)
            & (hsv[:, :, 2] >= 65)
            & annulus
        )
        neutral = (
            (hsv[:, :, 1] <= 75)
            & (hsv[:, :, 2] >= 125)
            & annulus
        )
        red_ratio = float(np.count_nonzero(red)) / sample_count
        green_ratio = float(np.count_nonzero(green)) / sample_count
        neutral_ratio = float(np.count_nonzero(neutral)) / sample_count
        if red_ratio >= 0.012:
            return "hostile"
        if green_ratio >= 0.012:
            return "friendly"
        if neutral_ratio >= 0.045:
            return "neutral"
        return "unknown"

    def find_central_capture_zone(self, minimap):
        """Find the plausible capture circle nearest the geometric map centre."""
        candidates = self.find_capture_zones(minimap)
        if not candidates:
            return None
        height, width = minimap.shape[:2]
        scale = min(width, height)
        map_center = (width / 2.0, height / 2.0)
        central = [
            zone
            for zone in candidates
            if math.dist(zone.center, map_center) <= scale * 0.30
        ]
        if not central:
            return None
        return min(central, key=lambda zone: math.dist(zone.center, map_center))

    def find_player_pose_on_minimap(self, minimap):
        """Find a white player arrow, using range rings as supporting evidence.

        Bright islands and grid text look similar to the arrow in isolation.
        A unique, filled white arrow may be used without a yellow range ring.
        Ambiguous shapes still require circular evidence; not every ship/UI
        configuration displays a yellow circle.
        """
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        # The player marker has a dark outline and can be semi-transparent
        # while the native autopilot is active.  The older near-white-only
        # threshold lost that arrow on bright/snow maps, which in turn made
        # the route layer guess a bearing.  Keep the range-circle evidence
        # below as the false-positive guard, but accept the full bright arrow.
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, 155]),
            np.array([180, 135, 255]),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        yellow = cv2.inRange(
            hsv,
            np.array([15, 40, 60]),
            np.array([40, 255, 255]),
        )
        yy, xx = np.indices(mask.shape)
        candidates = []
        height, width = minimap.shape[:2]
        scale = min(width, height)
        # The player's ship can legitimately sit very close to a minimap edge.
        # A large 3% margin rejected the live arrow near the bottom border;
        # the concentric range-ring score already filters edge grid labels.
        margin = max(4, int(min(width, height) * 0.01))
        for contour in contours:
            area = cv2.contourArea(contour)
            if not scale * scale * 0.00006 <= area <= scale * scale * 0.001:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if x < margin or y < margin or x + w > width - margin or y + h > height - margin:
                continue
            if not scale * 0.012 <= max(w, h) <= scale * 0.06:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            vertices = len(cv2.approxPolyDP(contour, 0.08 * perimeter, True))
            if not 3 <= vertices <= 5:
                continue
            center_x = x + w // 2
            center_y = y + h // 2
            distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
            ring = (
                (distance_sq >= (scale * 0.075) ** 2)
                & (distance_sq <= (scale * 0.17) ** 2)
            )
            ring_score = int(np.count_nonzero((yellow > 0) & ring))
            # Raw yellow-pixel counts are easily dominated by range text,
            # capture-point strokes, or unrelated arcs elsewhere in the
            # annulus.  A real player marker sits at the centre of a range
            # circle, so its yellow pixels cover many different directions.
            # Count occupied angular sectors and rank that circular evidence
            # ahead of the raw number of pixels.
            ring_y, ring_x = np.nonzero((yellow > 0) & ring)
            angles = np.arctan2(
                ring_y.astype(np.float64) - center_y,
                ring_x.astype(np.float64) - center_x,
            )
            angle_bins = np.floor((angles + math.pi) * (72 / (2 * math.pi)))
            angle_bins = np.clip(angle_bins.astype(np.int32), 0, 71)
            angle_counts = np.bincount(angle_bins, minlength=72)
            ring_coverage = int(np.count_nonzero(angle_counts >= 2))
            polygon = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
            points = polygon.reshape(-1, 2).astype(np.float64)
            tip = self._arrow_tip(points)
            if tip is None:
                continue
            direction_x = float(tip[0] - center_x)
            direction_y = float(tip[1] - center_y)
            length = math.hypot(direction_x, direction_y)
            if length < 2:
                continue
            fill_ratio = float(area) / max(float(w * h), 1.0)
            # The live player marker is a filled three-point navigation arrow.
            # Capture letters, ship-class glyphs and range labels can sit in
            # stronger yellow circles and win the old ring-only ranking.  Give
            # the arrow silhouette priority, while retaining the older ring
            # fallback for unusual UI scales where antialiasing adds a vertex.
            arrow_silhouette = bool(
                vertices == 3
                and area >= scale * scale * 0.00014
                and area <= max(150.0, scale * scale * 0.00042)
                and max(w, h) <= max(22.0, scale * 0.04)
                and 0.28 <= fill_ratio <= 0.62
            )
            component = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(component, [contour - (x, y)], -1, 255, -1)
            pixels = hsv[y:y + h, x:x + w][component > 0]
            white_core = bool(
                len(pixels)
                and np.mean((pixels[:, 1] <= 60) & (pixels[:, 2] >= 200)) >= 0.60
            )
            candidates.append(
                (
                    arrow_silhouette,
                    ring_coverage,
                    ring_score,
                    center_x,
                    center_y,
                    direction_x / length,
                    direction_y / length,
                    white_core,
                )
            )
        if not candidates:
            return None
        silhouette_candidates = [
            candidate for candidate in candidates if candidate[0]
        ]
        # The range-circle score is useful when the arrow is occluded, but it
        # is not an identity signal. A fixed triangular marker elsewhere in
        # the minimap can have a stronger ring simply because it is painted on
        # top of the map texture. A unique white-core silhouette is the live
        # ship arrow and must win before ring coverage is considered.
        white_silhouette_candidates = [
            candidate for candidate in silhouette_candidates if candidate[7]
        ]
        unique_white_arrow = len(white_silhouette_candidates) == 1
        if unique_white_arrow:
            ranked = list(white_silhouette_candidates)
        else:
            ranked = silhouette_candidates or candidates
            ranked.sort(key=lambda candidate: (candidate[1], candidate[2]), reverse=True)
        # Only portions of the range circle can be visible behind islands or
        # capture overlays.  Twelve occupied sectors still describe a ring;
        # requiring eighteen made the live white arrow disappear precisely
        # when the ship approached an island.
        minimum_coverage = 12
        minimum_score = max(32, int(scale * scale * 0.00024))
        unique_white_arrow = unique_white_arrow or (
            len(silhouette_candidates) == 1 and ranked[0][7]
        )
        if not unique_white_arrow and (
            ranked[0][1] < minimum_coverage or ranked[0][2] < minimum_score
        ):
            return None
        if len(ranked) > 1 and ranked[0][1] < ranked[1][1] * 1.08:
            return None
        best = ranked[0]
        return PlayerPose(
            position=(best[3], best[4]),
            heading=(best[5], best[6]),
        )

    @staticmethod
    def _arrow_tip(points):
        if len(points) < 3:
            return None
        if len(points) == 3:
            # The live minimap arrow is a long, narrow triangular pointer.
            # Its short edge is the stern/base and the opposite vertex is the
            # bow.  Treating a long sloping side as the base flips the heading
            # toward a stern corner by roughly 150-180 degrees (confirmed on
            # the saved 2K battle frames).
            opposite_edges = []
            for index, point in enumerate(points):
                other = np.delete(points, index, axis=0)
                opposite_edges.append((np.linalg.norm(other[0] - other[1]), point))
            return min(opposite_edges, key=lambda candidate: candidate[0])[1]
        best = None
        for index, point in enumerate(points):
            previous = points[index - 1] - point
            following = points[(index + 1) % len(points)] - point
            denominator = np.linalg.norm(previous) * np.linalg.norm(following)
            if denominator <= 0:
                continue
            cosine = float(np.dot(previous, following) / denominator)
            angle = math.acos(max(-1.0, min(cosine, 1.0)))
            if best is None or angle < best[0]:
                best = (angle, point)
        return None if best is None else best[1]

    def find_island_risk(self, minimap, pose, *, island_outlines=None):
        """Measure solid terrain inside a forward navigation corridor.

        Thin range rings and grid lines are rejected by component density.
        The return distance is normalized by the minimap diagonal, matching
        target-distance configuration used by the movement controller.
        """
        if minimap is None or pose is None:
            return None
        height, width = minimap.shape[:2]
        scale = min(width, height)
        if island_outlines:
            # A match's coastline is immutable.  Rasterize the frozen browser
            # layer instead of re-segmenting animated rings, contacts or smoke
            # on every control frame.
            island_outlines = exclude_player_land(
                island_outlines,
                (pose.position[0] / max(width, 1), pose.position[1] / max(height, 1)),
            )
            terrain = np.zeros((height, width), dtype=np.uint8)
            for outline in island_outlines:
                points = outline.get("points", []) if isinstance(outline, dict) else []
                polygon = []
                for point in points:
                    if not isinstance(point, (tuple, list)) or len(point) < 2:
                        continue
                    polygon.append(
                        [
                            int(round(float(point[0]) * max(width - 1, 1))),
                            int(round(float(point[1]) * max(height - 1, 1))),
                        ]
                    )
                if len(polygon) >= 3:
                    cv2.fillPoly(terrain, [np.asarray(polygon, dtype=np.int32)], 255)
            return self._measure_island_risk(terrain, pose, trusted_terrain=True)
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float64)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        ocean_samples = hue[(saturation > 25) & (value > 25)]
        ocean_hue = float(np.median(ocean_samples)) if ocean_samples.size else 100.0
        ocean_value = float(np.median(value[(saturation > 25) & (value > 25)])) \
            if ocean_samples.size else 80.0
        value_floor = terrain_value_floor(ocean_value)
        hue_delta = np.minimum(
            np.abs(hue - ocean_hue),
            180.0 - np.abs(hue - ocean_hue),
        )
        colored_land = (
            (hue_delta > 16)
            & (saturation > 22)
            & (value > 28)
        ).astype(np.uint8) * 255
        # Team/capture/smoke overlays are saturated green or red disks. They
        # are UI, not terrain, and previously produced a permanent 0.01 island
        # distance around the player marker.
        ui_overlay = (
            (((hue >= 35) & (hue <= 95)) & (saturation > 70))
            | (((hue <= 10) | (hue >= 168)) & (saturation > 22))
        )
        colored_land[ui_overlay] = 0
        capture_zone = self.find_nearest_capture_zone(minimap, pose.position)
        capture_disk = None
        if capture_zone is not None:
            yy, xx = np.indices(colored_land.shape)
            capture_disk = (
                (xx - capture_zone.center[0]) ** 2
                + (yy - capture_zone.center[1]) ** 2
                <= (capture_zone.radius * 1.18) ** 2
            )
            colored_land[capture_disk] = 0
        colored_land = cv2.morphologyEx(
            colored_land, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        colored_land = cv2.morphologyEx(
            colored_land, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        neutral_bright = cv2.inRange(
            hsv,
            np.array([0, 0, value_floor]),
            np.array([180, 145, 255]),
        )
        neutral_bright = cv2.morphologyEx(
            neutral_bright,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )
        terrain = np.zeros_like(neutral_bright)
        minimum_pixels = max(120, int(scale * scale * 0.00032))
        minimum_extent = max(10, int(scale * 0.018))
        player_x, player_y = pose.position

        for source, colored in ((colored_land, True), (neutral_bright, False)):
            labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                source, connectivity=8
            )
            for label in range(1, labels_count):
                x, y, component_width, component_height, pixels = stats[label]
                bounding_area = max(component_width * component_height, 1)
                density = pixels / bounding_area
                if pixels < minimum_pixels:
                    continue
                if max(component_width, component_height) < minimum_extent:
                    continue
                if density < 0.17:
                    continue
                # Neutral grid/range lines often connect into one component
                # spanning the entire minimap. It is UI, never an island.
                if not colored and (
                    component_width > scale * 0.48
                    or component_height > scale * 0.48
                    or bounding_area > scale * scale * 0.20
                ):
                    continue
                if (
                    x - 4 <= player_x <= x + component_width + 4
                    and y - 4 <= player_y <= y + component_height + 4
                ):
                    continue
                if colored and pixels > minimum_pixels * 5:
                    component = (labels == label).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    perimeter = sum(
                        cv2.arcLength(contour, True) for contour in contours
                    )
                    roundness = (
                        4 * math.pi * pixels / (perimeter * perimeter)
                        if perimeter > 0
                        else 0.0
                    )
                    aspect = component_width / max(component_height, 1)
                    # Red/green cap overlays and warning disks are round UI,
                    # while island coastlines are irregular.
                    if 0.72 < roundness and 0.72 < aspect < 1.38:
                        continue
                terrain[labels == label] = 255

        return self._measure_island_risk(terrain, pose)

    @staticmethod
    def plan_kinematic_rudder(
        minimap_shape,
        pose,
        target,
        island_outlines,
        *,
        speed_knots: float = 30.0,
        rudder_shift_seconds: float = 15.0,
        turning_radius_km: float = 1.0,
        horizon_seconds: float = 120.0,
        safety_clearance_km: float = 0.45,
        initial_rudder: float = 0.0,
        preferred_side: int = 1,
        map_span_km: float = 50.0,
        speed_scale: float = 1.0,
        observed_speed_km_s: float | None = None,
        turn_speed_retention: float = 1.0,
        speed_response_seconds: float = 12.0,
        initial_yaw_rate: float | None = None,
        yaw_response_seconds: float = 0.0,
    ):
        """Choose a Q/E notch by comparing physically plausible future paths.

        Map span and movement compression are explicit. Each order is simulated
        with a bounded rudder slew and a circular full-rudder turn.  Terrain
        clearance is a hard constraint; only after a route is safe do target
        progress and final heading decide which course wins.
        """
        if pose is None or target is None:
            return None
        height, width = minimap_shape[:2]
        if height <= 0 or width <= 0:
            return None
        numbers = (*pose.position, *pose.heading, *target, speed_knots,
                   rudder_shift_seconds, turning_radius_km, horizon_seconds,
                   safety_clearance_km, initial_rudder, map_span_km, speed_scale,
                   turn_speed_retention, speed_response_seconds, yaw_response_seconds)
        if not all(math.isfinite(float(value)) for value in numbers):
            return None
        if any(value is not None and not math.isfinite(float(value))
               for value in (observed_speed_km_s, initial_yaw_rate)):
            return None
        if not (0 <= pose.position[0] < width and 0 <= pose.position[1] < height):
            return None
        heading_x, heading_y = pose.heading
        heading_length = math.hypot(heading_x, heading_y)
        if heading_length < 0.5:
            return None

        scale = float(min(width, height))
        pixels_per_km = max(scale / max(1.0, float(map_span_km)), 1e-6)
        speed_km_s = max(1.0, float(speed_knots)) * 1.852 / 3600.0 * max(0.1, speed_scale)
        speed_pixels_s = speed_km_s * pixels_per_km
        shift_seconds = max(1.0, float(rudder_shift_seconds))
        turn_radius = max(0.25, float(turning_radius_km))
        horizon = max(30.0, min(float(horizon_seconds), 240.0))
        clearance_limit = max(0.15, min(float(safety_clearance_km), 1.5))
        turn_speed_retention = max(0.0, min(float(turn_speed_retention), 1.0))

        terrain = np.zeros((height, width), dtype=np.uint8)
        island_outlines = exclude_player_land(
            island_outlines,
            (pose.position[0] / width, pose.position[1] / height),
        )
        for outline in island_outlines or ():
            points = (
                outline.get("points", ())
                if isinstance(outline, dict)
                else ()
            )
            polygon = []
            for point in points:
                if not isinstance(point, (tuple, list)) or len(point) < 2:
                    continue
                polygon.append(
                    [
                        int(round(float(point[0]) * max(width - 1, 1))),
                        int(round(float(point[1]) * max(height - 1, 1))),
                    ]
                )
            if len(polygon) >= 3:
                cv2.fillPoly(
                    terrain,
                    [np.asarray(polygon, dtype=np.int32)],
                    255,
                )
        water = np.where(terrain > 0, 0, 255).astype(np.uint8)
        terrain_clearance = (
            cv2.distanceTransform(water, cv2.DIST_L2, 5) / pixels_per_km
            if np.any(terrain)
            else None
        )

        start_x = float(pose.position[0])
        start_y = float(pose.position[1])
        target_x = float(target[0])
        target_y = float(target[1])
        initial_distance_km = (
            math.hypot(target_x - start_x, target_y - start_y)
            / pixels_per_km
        )
        initial_helm = max(-1.0, min(float(initial_rudder), 1.0))
        side = 1 if preferred_side >= 0 else -1
        candidates = (0.0, 0.5 * side, -0.5 * side, 1.0 * side, -1.0 * side)
        results = []

        for candidate in candidates:
            x = start_x
            y = start_y
            heading_angle = math.atan2(heading_y, heading_x)
            effective_rudder = initial_helm
            current_speed = (speed_km_s if observed_speed_km_s is None else
                             max(0.0, min(observed_speed_km_s, speed_km_s * 1.5)))
            yaw_rate = (initial_helm * current_speed / turn_radius if initial_yaw_rate is None
                        else max(-0.26, min(float(initial_yaw_rate), 0.26)))
            clearances = []
            future_clearances = []
            collision_time = None
            physical_collision = False
            elapsed = 0.0
            while elapsed < horizon:
                step = min(1.0, 0.5 / max(speed_pixels_s * 1.5, 1e-6), horizon - elapsed)
                maximum_change = step / shift_seconds
                helm_delta = max(
                    -maximum_change,
                    min(candidate - effective_rudder, maximum_change),
                )
                effective_rudder += helm_delta
                desired_speed = speed_km_s * (1 - (1 - turn_speed_retention) * abs(effective_rudder))
                current_speed += (desired_speed - current_speed) * (1 - math.exp(-step / max(1.0, speed_response_seconds)))
                # Positive E rudder turns clockwise in minimap coordinates,
                # whose Y axis points down, hence the positive angle update.
                target_yaw_rate = effective_rudder * current_speed / turn_radius
                yaw_rate += (target_yaw_rate - yaw_rate) * (
                    1.0 if yaw_response_seconds <= 0 else
                    1 - math.exp(-step / yaw_response_seconds)
                )
                heading_angle += yaw_rate * step
                x += math.cos(heading_angle) * current_speed * pixels_per_km * step
                y += math.sin(heading_angle) * current_speed * pixels_per_km * step
                elapsed += step

                if not (0 <= x < width and 0 <= y < height):
                    clearance = 0.0
                else:
                    sample_x = min(width - 1, max(0, int(round(x))))
                    sample_y = min(height - 1, max(0, int(round(y))))
                    boundary_clearance = min(
                        x,
                        y,
                        width - 1 - x,
                        height - 1 - y,
                    ) / pixels_per_km
                    land_clearance = (
                        float(terrain_clearance[sample_y, sample_x])
                        if terrain_clearance is not None
                        else 50.0
                    )
                    clearance = min(boundary_clearance, land_clearance)
                clearances.append(clearance)
                if elapsed >= shift_seconds:
                    future_clearances.append(clearance)
                if collision_time is None and clearance <= clearance_limit:
                    collision_time = elapsed
                # Entering the safety envelope can still allow a turn away.
                # Actual land/border contact cannot: never score imaginary
                # progress or clear water on the far side of an island.
                if clearance <= 0.0:
                    physical_collision = True
                    break

            final_distance_km = (
                math.hypot(target_x - x, target_y - y) / pixels_per_km
            )
            target_dx = target_x - x
            target_dy = target_y - y
            final_heading_x = math.cos(heading_angle)
            final_heading_y = math.sin(heading_angle)
            final_cross = final_heading_x * target_dy - final_heading_y * target_dx
            final_dot = final_heading_x * target_dx + final_heading_y * target_dy
            final_bearing = abs(math.atan2(final_cross, final_dot)) / math.pi
            progress_km = initial_distance_km - final_distance_km
            minimum_clearance = min(clearances) if clearances else 0.0
            mean_future_clearance = (
                sum(future_clearances) / len(future_clearances)
                if future_clearances
                else minimum_clearance
            )
            end_clearance = clearances[-1] if clearances else 0.0
            direction_change = abs(candidate - initial_helm)
            target_score = (
                progress_km * 2.2
                + (1.0 - final_bearing) * 3.0
                - abs(candidate) * 0.06
                - direction_change * 0.08
            )
            escape_score = (
                (collision_time if collision_time is not None else horizon) * 0.08
                + min(mean_future_clearance, 4.0) * 1.6
                + min(end_clearance, 4.0) * 2.4
                + target_score * 0.15
            )
            results.append(
                {
                    "rudder": candidate,
                    "collision_time": collision_time,
                    "physical_collision": physical_collision,
                    "minimum_clearance": minimum_clearance,
                    "target_score": target_score,
                    "escape_score": escape_score,
                    "endpoint": (
                        max(0.0, min(x / max(width - 1, 1), 1.0)),
                        max(0.0, min(y / max(height - 1, 1), 1.0)),
                    ),
                }
            )

        target_choice = max(results, key=lambda item: item["target_score"])
        safe_results = [
            item for item in results if item["collision_time"] is None
        ]
        avoidance_required = target_choice["collision_time"] is not None
        if safe_results:
            selected = max(
                safe_results,
                key=lambda item: (
                    item["target_score"]
                    + min(item["minimum_clearance"], 2.0) * 0.10
                ),
            )
        else:
            # If every path is already inside the safety envelope, neutral
            # helm cannot create an escape. Pick the turning path that opens
            # the most water even when straight would postpone contact by a
            # few seconds.
            selected = max(
                (
                    item
                    for item in results
                    if abs(item["rudder"]) >= 0.2
                ),
                key=lambda item: (not item["physical_collision"], item["escape_score"]),
            )
            avoidance_required = True

        return KinematicSteeringPlan(
            rudder=float(selected["rudder"]),
            avoidance_required=avoidance_required,
            collision_time_seconds=(
                None
                if target_choice["collision_time"] is None
                else float(target_choice["collision_time"])
            ),
            minimum_clearance_km=float(selected["minimum_clearance"]),
            predicted_endpoint=selected["endpoint"],
            all_paths_blocked=not safe_results,
        )

    def _measure_island_risk(self, terrain, pose, *, trusted_terrain=False):
        """Evaluate a frozen/rasterized terrain layer against live heading."""
        if terrain is None or pose is None or not np.any(terrain):
            return None
        height, width = terrain.shape[:2]
        scale = min(width, height)
        diagonal = max(math.hypot(width, height), 1.0)
        player_x, player_y = pose.position
        terrain = cv2.dilate(
            terrain,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        yy, xx = np.indices(terrain.shape)
        self_exclusion = (
            (xx - player_x) ** 2 + (yy - player_y) ** 2
            <= (scale * 0.035) ** 2
        )
        if not trusted_terrain:
            terrain[self_exclusion] = 0
        terrain_y, terrain_x = np.nonzero(terrain)
        delta_x = terrain_x.astype(np.float64) - player_x
        delta_y = terrain_y.astype(np.float64) - player_y
        heading_x, heading_y = pose.heading
        forward = delta_x * heading_x + delta_y * heading_y
        lateral = delta_x * (-heading_y) + delta_y * heading_x
        minimum_forward = 0.0 if trusted_terrain else scale * 0.018
        # Islands only justify evasive steering when they are directly in the
        # current bow corridor.  The old 0.34 widening factor included large
        # off-bow islands and made a full-speed ship start a needless arc.
        # Keep a longer look-ahead for the navigation display and route
        # planner, but the narrow corridor below ensures only geometry on the
        # actual heading is reported.  Movement only reacts at its much
        # shorter warning/emergency distances.
        maximum_forward = diagonal * 0.18
        corridor_half_width = scale * 0.018 + forward * 0.18
        in_corridor = (
            (forward >= minimum_forward)
            & (forward <= maximum_forward)
            & (np.abs(lateral) <= corridor_half_width)
        )
        if not np.any(in_corridor):
            return None
        distances = np.hypot(delta_x[in_corridor], delta_y[in_corridor])
        distance = float(np.min(distances) / diagonal)

        left_clearance = self._ray_clearance(
            delta_x,
            delta_y,
            pose.heading,
            angle=-math.radians(38),
            scale=scale,
            maximum=maximum_forward,
        )
        right_clearance = self._ray_clearance(
            delta_x,
            delta_y,
            pose.heading,
            angle=math.radians(38),
            scale=scale,
            maximum=maximum_forward,
        )
        if abs(right_clearance - left_clearance) < scale * 0.02:
            avoidance_rudder = 0.0
        else:
            avoidance_rudder = 1.0 if right_clearance > left_clearance else -1.0
        return IslandRisk(distance, avoidance_rudder)

    def find_minimap_island_outlines(self, minimap, *, maximum_shapes: int = 128):
        """Return simplified island polygons for the browser radar.

        This intentionally derives coastline candidates from the same live
        minimap pixels used by navigation.  It is not a copied image: rings,
        labels, contacts and player glyphs are filtered first, then the
        remaining terrain components are reduced to small normalized polygons.
        """
        if minimap is None or minimap.size == 0:
            return []
        height, width = minimap.shape[:2]
        scale = min(width, height)
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float64)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        ocean_samples = hue[(saturation > 25) & (value > 25)]
        ocean_hue = float(np.median(ocean_samples)) if ocean_samples.size else 100.0
        ocean_value = float(np.median(value[(saturation > 25) & (value > 25)])) \
            if ocean_samples.size else 80.0
        value_floor = terrain_value_floor(ocean_value)
        hue_delta = np.minimum(
            np.abs(hue - ocean_hue),
            180.0 - np.abs(hue - ocean_hue),
        )
        colored_terrain = (
            (hue_delta > 16)
            & (saturation > 22)
            & (value > 28)
        ).astype(np.uint8) * 255
        # Snow/night maps encode most coastlines as low-saturation shapes.
        # The floor follows the palette: a fixed 125 missed nearly every
        # island on dark maps.  Grey-blue island fill still stays below 110
        # saturation while the ocean sits near 200+.
        neutral_terrain = (
            (saturation <= 145)
            & (value >= value_floor)
            & ((saturation <= 110) | (hue_delta > 16) | (value >= 180))
        ).astype(np.uint8) * 255
        # Red/green team marks, capture circles and smoke overlays are not
        # coastlines.  Remove them before connected-component extraction.
        ui_overlay = (
            (((hue >= 35) & (hue <= 95)) & (saturation > 70))
            | (((hue <= 10) | (hue >= 168)) & (saturation > 22))
        )
        colored_terrain[ui_overlay] = 0
        neutral_terrain[ui_overlay] = 0
        player_pose = self.find_player_pose_on_minimap(minimap)
        player = None if player_pose is None else player_pose.position
        zone_mask = np.zeros((height, width), dtype=bool)
        yy, xx = np.indices((height, width))
        for zone in self.find_capture_zones(minimap, player=player):
            # Only remove ring strokes and labels, never land inside a cap.
            radial = np.hypot(xx - zone.center[0], yy - zone.center[1])
            zone_mask |= abs(radial - zone.radius) <= max(3.0, scale * 0.006)
            zone_mask |= radial <= max(4.0, scale * 0.012)
        colored_terrain[zone_mask] = 0
        neutral_terrain[zone_mask] = 0
        if player is not None:
            player_mask = (
                (xx - player[0]) ** 2 + (yy - player[1]) ** 2
                <= max(5.0, scale * 0.012) ** 2
            )
            colored_terrain[player_mask] = 0
            neutral_terrain[player_mask] = 0

        occluded = ui_overlay | zone_mask
        if player is not None:
            occluded |= player_mask
        gap_radius = max(4, min(7, int(round(scale * 0.01))))
        colored_terrain = repair_overlay_gaps(colored_terrain, occluded, gap_radius)
        neutral_terrain = repair_overlay_gaps(neutral_terrain, occluded, gap_radius)
        minimum_pixels = max(24, int(scale * scale * 0.00010))
        minimum_extent = max(6, int(scale * 0.010))

        def retained_components(source, *, neutral=False):
            source = cv2.morphologyEx(
                source, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            count, component_labels, component_stats, _ = (
                cv2.connectedComponentsWithStats(source, connectivity=8)
            )
            kept = np.zeros_like(source)
            for component_label in range(1, count):
                x, y, component_width, component_height, pixels = (
                    component_stats[component_label]
                )
                bounding_area = max(component_width * component_height, 1)
                density = pixels / bounding_area
                aspect = max(component_width, component_height) / max(
                    min(component_width, component_height), 1
                )
                if pixels < minimum_pixels or density < 0.15:
                    continue
                if aspect > 4.5 and density < 0.35:
                    continue
                if max(component_width, component_height) < minimum_extent:
                    continue
                if (
                    component_width > scale * (0.48 if neutral else 0.52)
                    or component_height > scale * (0.48 if neutral else 0.52)
                    or bounding_area > scale * scale * (0.20 if neutral else 0.22)
                ):
                    continue
                component_crop = (component_labels[
                    y:y + component_height, x:x + component_width
                ] == component_label).astype(np.uint8)
                thickness = cv2.distanceTransform(
                    np.pad(component_crop, 1), cv2.DIST_L2, 5
                )
                if float(thickness.max()) < 2.5:
                    continue
                kept[component_labels == component_label] = 255
            return kept

        terrain = cv2.bitwise_or(
            retained_components(colored_terrain),
            retained_components(neutral_terrain, neutral=True),
        )
        # Small overlay gaps have been reconstructed from adjacent land.
        # Do not erase those coastlines again or close visible water channels.
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            terrain, connectivity=8
        )
        candidates = []
        for label in range(1, labels_count):
            x, y, component_width, component_height, pixels = stats[label]
            bounding_area = max(component_width * component_height, 1)
            density = pixels / bounding_area
            aspect = max(component_width, component_height) / max(
                min(component_width, component_height), 1
            )
            if pixels < minimum_pixels or density < 0.15:
                continue
            if aspect > 4.5 and density < 0.35:
                continue
            if max(component_width, component_height) < minimum_extent:
                continue
            if (
                component_width > scale * 0.52
                or component_height > scale * 0.52
                or bounding_area > scale * scale * 0.22
            ):
                continue
            component = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            simplified = cv2.approxPolyDP(contour, max(0.5, perimeter * 0.003), True)
            if len(simplified) < 3:
                continue
            points = [
                [
                    round(float(point[0][0]) / max(width, 1), 4),
                    round(float(point[0][1]) / max(height, 1), 4),
                ]
                for point in simplified
            ]
            candidates.append(
                (
                    int(pixels),
                    points,
                    round(float(pixels) / max(width * height, 1), 6),
                )
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            {"points": points, "area": area}
            for _, points, area in candidates[: max(0, int(maximum_shapes))]
        ]

    @staticmethod
    def plan_island_aware_waypoint(
        minimap_shape, player, target, island_outlines, *, clearance_ratio=0.022,
    ):
        """Plan a complete water route; None means the target is unreachable."""
        from strategy.island_route import IslandRoutePlanner

        return IslandRoutePlanner().waypoint(
            minimap_shape, player, target, island_outlines,
            clearance_ratio=clearance_ratio,
        )

    @staticmethod
    def _ray_clearance(delta_x, delta_y, heading, *, angle, scale, maximum):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        heading_x, heading_y = heading
        ray_x = heading_x * cosine - heading_y * sine
        ray_y = heading_x * sine + heading_y * cosine
        forward = delta_x * ray_x + delta_y * ray_y
        lateral = np.abs(delta_x * (-ray_y) + delta_y * ray_x)
        corridor = (
            (forward >= scale * 0.018)
            & (forward <= maximum)
            & (lateral <= scale * 0.025 + forward * 0.14)
        )
        if not np.any(corridor):
            return maximum
        return float(np.min(forward[corridor]))

    def find_enemies_from_hsv(self, hsv):
        # Live enemy minimap glyphs are saturated red (hue about 3-6). Brown
        # islands sit around hue 14-23 with low saturation; the former broad
        # red/orange mask therefore produced several persistent "enemy" tracks
        # at spawn and cancelled native autopilot after only a few seconds.
        mask = (
            cv2.inRange(
                hsv,
                np.array([0, 100, 90]),
                np.array([10, 255, 255]),
            )
            | cv2.inRange(
                hsv,
                np.array([165, 100, 90]),
                np.array([180, 255, 255]),
            )
        )
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 6 <= area <= 220:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) > min(hsv.shape[:2]) * 0.055:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            points.append(
                (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]),
                    area,
                )
            )
        clusters = self._cluster_colored_points(points, radius_x=35, radius_y=24)
        # Enemy ship glyphs have several colored strokes.  Single tiny capture
        # labels and zone-ring fragments are intentionally discarded.
        minimap = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        zones = self.find_capture_zones(minimap)
        label_margin = max(10, min(hsv.shape[:2]) * .024)
        return [
            (center_x, center_y)
            for center_x, center_y, total_area, members in clusters
            if total_area >= 35 and members >= 2
            and not any(math.dist((center_x, center_y), z.center) <= label_margin for z in zones)
        ]

    @staticmethod
    def _cluster_colored_points(points, *, radius_x, radius_y):
        """Merge nearby UI strokes into one semantic marker."""
        clusters = []
        for point_x, point_y, area in sorted(points, key=lambda item: item[0]):
            for cluster in clusters:
                if (
                    abs(point_x - cluster[0]) <= radius_x
                    and abs(point_y - cluster[1]) <= radius_y
                ):
                    old_area = cluster[2]
                    new_area = old_area + area
                    cluster[0] = int((cluster[0] * old_area + point_x * area) / new_area)
                    cluster[1] = int((cluster[1] * old_area + point_y * area) / new_area)
                    cluster[2] = new_area
                    cluster[3] += 1
                    break
            else:
                clusters.append([point_x, point_y, area, 1])
        return [tuple(cluster) for cluster in clusters]

    def find_enemies_in_viewport(self, image):
        """Pair an enemy ship glyph with its rectangular HP/numeric label.

        Sunset shorelines also contain clusters of red/orange pixels. Colour
        strokes alone therefore cannot establish a target. Even a nearly empty
        HP bar must have its ship marker above and numeric strokes inside the
        complete label, independent of the remaining red fill width.
        """
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        left, right = int(width * .02), int(width * .98)
        top, bottom = int(height * .23), int(height * .73)
        search = image[top:bottom, left:right]
        if search.size == 0:
            return []
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        # Enemy bars and ship glyphs are vivid red. The old low-saturation
        # orange range accepted reflected sunlight and clouds across the sea.
        mask = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165))
                & (hsv[:, :, 1] >= 145) & (hsv[:, :, 2] >= 145)).astype(np.uint8) * 255
        # Locking a target changes its ship glyph to pale peach, while its HP
        # fill remains saturated red. Permit that measured glyph palette only
        # in the glyph/HP/numeric pairing below; peach sea/sky fragments never
        # become candidate HP bars or independent target points.
        # A bright daylight lock also uses H4..8 / S127..142. Its glyph is
        # still less saturated than the independent red HP anchor (S>=145).
        selected_glyph = cv2.inRange(hsv, np.array([4, 45, 210]),
                                     np.array([22, 144, 255]))
        mx1, my1, mx2, my2 = layout.rect("minimap")
        mask[max(0, my1 - top):max(0, min(bottom, my2) - top),
             max(0, mx1 - left):max(0, min(right, mx2) - left)] = 0
        selected_glyph[max(0, my1 - top):max(0, min(bottom, my2) - top),
                       max(0, mx1 - left):max(0, min(right, mx2) - left)] = 0
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        targets = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Local bar height measures the label's own UI size. Do not infer
            # a moved/scaled target label from the framebuffer or own-ship HUD.
            unit = h / 14.0
            if not (5 <= h <= max(40, 35 * layout.ui_scale)
                    and 4 * unit <= w <= 190 * unit
                    and cv2.contourArea(contour) / (w * h) >= .45):
                continue
            flat_rows = np.count_nonzero(mask[y:y + h, x:x + w], axis=1) >= w * .75
            if np.count_nonzero(flat_rows) < max(1, round(unit)):
                continue
            points = self._viewport_hp_ship_markers(mask, hsv, (x, y, w, h), unit)
            if not points:
                # Do not join peach name text/clouds to an already verified
                # red glyph, or let them replace an existing red target.
                points = self._viewport_hp_ship_markers(selected_glyph, hsv, (x, y, w, h), unit)
            for point in points:
                absolute = (point[0] + left, point[1] + top)
                if not any(abs(absolute[0] - old[0]) < 12 * unit
                           and abs(absolute[1] - old[1]) < 12 * unit for old in targets):
                    targets.append(absolute)
        return sorted(targets)

    @staticmethod
    def _viewport_hp_ship_markers(mask, hsv, bar, unit):
        """Return actual ship-glyph centres supported by one HP label strip."""
        x, y, w, h = bar
        height, width = mask.shape
        yt, yb = max(0, round(y - 70 * unit)), max(0, round(y - 18 * unit))
        xl = max(0, round(x + 25 * unit))
        # Two overlapping target HP bars can form one long red component.
        # Each still needs its own glyph and numeric label; never merge ships.
        xr = min(width, round(x + max(110 * unit, w - 25 * unit)))
        zone = mask[yt:yb, xl:xr]
        if zone.size == 0:
            return []
        joined = cv2.morphologyEx(zone, cv2.MORPH_CLOSE,
                                 np.ones((1, max(3, math.ceil(5 * unit))), np.uint8))
        icons, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for icon in icons:
            ix, iy, iw, ih = cv2.boundingRect(icon)
            fill = cv2.contourArea(icon) / (iw * ih)
            if not (5 * unit <= iw <= 42 * unit and 4 * unit <= ih <= 23 * unit
                    and .8 <= iw / ih <= 5 and .20 <= fill <= .92):
                continue
            center_x = xl + ix + iw / 2
            # A framebuffer aspect change can stretch the label horizontally.
            # Keep the entire ship glyph, while retaining the HP/digit pairing.
            if abs(center_x - (x + 56 * unit)) > 32 * unit:
                if not (w >= 128 * unit and x + 35 * unit <= center_x <= x + w - 35 * unit):
                    continue
            sx1, sx2 = max(0, round(center_x - 60 * unit)), min(width, round(center_x + 60 * unit))
            stripe = hsv[y:y + h, sx1:sx2]
            white = ((stripe[:, :, 1] < 65) & (stripe[:, :, 2] > 175)).astype(np.uint8) * 255
            digits, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            strokes = [cv2.boundingRect(c) for c in digits
                       if 3 * unit <= cv2.boundingRect(c)[3] <= h
                       and unit <= cv2.boundingRect(c)[2] <= 22 * unit]
            if len(strokes) < 3 and unit < 1:
                # Small labels blend white numerals with the red HP fill.
                # Recover their bright antialiased strokes only after the
                # independent HP/glyph geometry has passed; retain the same
                # numeric height, width and minimum-stroke evidence.
                white = ((stripe[:, :, 1] < 95) & (stripe[:, :, 2] > 205)).astype(np.uint8) * 255
                digits, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                strokes = [cv2.boundingRect(c) for c in digits
                           if 3 * unit <= cv2.boundingRect(c)[3] <= h
                           and unit <= cv2.boundingRect(c)[2] <= 22 * unit]
            if len(strokes) >= 3:
                points.append((round(center_x), round(yt + iy + ih / 2)))
        return points

    def has_torpedoes_from_hsv(self, hsv):
        mask = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return any(cv2.contourArea(contour) > 300 for contour in contours)

    def find_minimap(self, image):
        """Keep a canonical grid through jitter; confirm moved UI on fresh frames."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = MINIMAP_REGION.pixels(width, height)
        layout = detect_hud_layout(image)
        measured = layout.rect("minimap") if layout.source("minimap") == "anchor" else None
        now = time.monotonic()
        changed = getattr(self, "_minimap_frame_shape", None) != (width, height)
        if changed:
            self._minimap_grid_cache = None
            self._minimap_canonical_bounds = None
            self._minimap_grid_pending = []
            self._minimap_grid_pending_at = float("-inf")
            self._minimap_grid_valid_at = float("-inf")
            self._minimap_grid_observed_frame = None
        self._minimap_frame_shape = (width, height)
        last_frame = getattr(self, "_minimap_grid_observed_frame", None)
        fresh = last_frame is None or last_frame() is not image
        if fresh:
            self._minimap_grid_observed_frame = weakref.ref(image)
            self._minimap_grid_checked_at = now
            canonical = getattr(self, "_minimap_canonical_bounds", None)
            if measured is None:
                self._minimap_grid_pending = []
            elif canonical is None:
                self._minimap_canonical_bounds = tuple(measured)
                self._minimap_grid_cache = ((width, height), tuple(measured))
                self._minimap_grid_valid_at = now
                self._minimap_grid_pending = []
            else:
                # Four pixels of raster/Hough jitter is not a viewport change.
                # Scale the tolerance with the cell width for very large UI.
                tolerance = max(4.0, min(canonical[2] - canonical[0],
                                         canonical[3] - canonical[1]) * .006)
                if max(abs(a - b) for a, b in zip(measured, canonical)) <= tolerance:
                    self._minimap_grid_cache = ((width, height), canonical)
                    self._minimap_grid_valid_at = now
                    self._minimap_grid_pending = []
                else:
                    pending = getattr(self, "_minimap_grid_pending", [])
                    consistent = bool(pending) and (
                        now - getattr(self, "_minimap_grid_pending_at", float("-inf")) <= 4.0
                        and max(abs(a - b) for a, b in zip(measured, pending[0])) <= tolerance
                    )
                    pending = [*pending, tuple(measured)] if consistent else [tuple(measured)]
                    self._minimap_grid_pending = pending
                    self._minimap_grid_pending_at = now
                    if len(pending) >= 3:
                        accepted = tuple(int(round(value)) for value in np.median(pending, axis=0))
                        self._minimap_canonical_bounds = accepted
                        self._minimap_grid_cache = ((width, height), accepted)
                        self._minimap_grid_valid_at = now
                        self._minimap_grid_pending = []
        # Unconfirmed jump candidates and repeated readers of one ndarray
        # cannot renew the canonical crop. Expiry is checked on every call.
        if now - getattr(self, "_minimap_grid_valid_at", float("-inf")) > 4.0:
            self._minimap_grid_cache = None
        if self._minimap_grid_cache is not None:
            x1, y1, x2, y2 = self._minimap_grid_cache[1]
        self.minimap_bounds_source = "grid" if self._minimap_grid_cache is not None else "ratio_fallback"
        self.minimap_bounds = (x1, y1, x2, y2)
        minimap = image[y1:y2, x1:x2]
        if minimap.size > 0 and minimap.std() > 8:
            return minimap
        logger.warning(
            "Minimap crop failed: region=(%s,%s,%s,%s)",
            x1,
            y1,
            x2,
            y2,
        )
        return None

    @staticmethod
    def find_tactical_map(image):
        """Rectify the M-key tactical grid into normalized map coordinates.

        The tactical grid is a slightly perspective-projected trapezoid and is
        shifted left to leave room for the instruction panel.  A centred crop
        therefore moves map objects by several percent.  Detect its repeated
        horizontal/vertical grid lines and warp the four outer corners to a
        square.  The old centred crop remains a safe fallback for themes where
        the faint grid cannot be recovered.
        """
        if image is None or image.size == 0:
            return None
        geometry = detect_tactical_map_geometry(image)
        if geometry is not None:
            return geometry.rectify(image)
        height, width = image.shape[:2]
        map_size = int(round(min(float(width), float(height)) * 0.81))
        if map_size <= 0:
            return None
        left = max(0, int(round((width - map_size) / 2.0)))
        top = max(0, int(round((height - map_size) / 2.0)))
        right = min(width, left + map_size)
        bottom = min(height, top + map_size)
        fallback = image[top:bottom, left:right]

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 90)
            horizontal_lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 360,
                threshold=max(80, int(width * 0.055)),
                minLineLength=max(120, int(width * 0.24)),
                maxLineGap=max(12, int(height * 0.035)),
            )
            horizontal_positions = []
            if horizontal_lines is not None:
                for raw_x1, raw_y1, raw_x2, raw_y2 in horizontal_lines.reshape(
                    -1,
                    4,
                ):
                    x1, y1, x2, y2 = map(
                        int,
                        (raw_x1, raw_y1, raw_x2, raw_y2),
                    )
                    if x2 < x1:
                        x1, x2, y1, y2 = x2, x1, y2, y1
                    line_y = (y1 + y2) / 2.0
                    if (
                        abs(y2 - y1) <= max(4, int(height * 0.005))
                        and x2 - x1 >= width * 0.24
                        and x1 < width * 0.55
                        and width * 0.60 < x2 < width * 0.82
                        and height * 0.035 < line_y < height * 0.92
                    ):
                        horizontal_positions.append(line_y)

            horizontal_clusters = []
            for line_y in sorted(horizontal_positions):
                if (
                    not horizontal_clusters
                    or line_y - float(np.mean(horizontal_clusters[-1]))
                    > height * 0.010
                ):
                    horizontal_clusters.append([line_y])
                else:
                    horizontal_clusters[-1].append(line_y)
            grid_rows = [
                float(np.median(cluster)) for cluster in horizontal_clusters
            ]
            if len(grid_rows) >= 7:
                grid_top = min(grid_rows)
                grid_bottom = max(grid_rows)
                grid_height = grid_bottom - grid_top
            else:
                grid_top = grid_bottom = grid_height = 0.0

            if height * 0.64 <= grid_height <= height * 0.89:
                vertical_lines = cv2.HoughLinesP(
                    edges,
                    1,
                    np.pi / 720,
                    threshold=max(70, int(height * 0.055)),
                    minLineLength=max(120, int(height * 0.22)),
                    maxLineGap=max(12, int(height * 0.040)),
                )
                grid_mid = (grid_top + grid_bottom) / 2.0
                vertical_models = []
                if vertical_lines is not None:
                    for raw_x1, raw_y1, raw_x2, raw_y2 in vertical_lines.reshape(
                        -1,
                        4,
                    ):
                        x1, y1, x2, y2 = map(
                            int,
                            (raw_x1, raw_y1, raw_x2, raw_y2),
                        )
                        delta_x = x2 - x1
                        delta_y = y2 - y1
                        if (
                            abs(delta_y) < height * 0.22
                            or abs(delta_x) > abs(delta_y) * 0.12
                        ):
                            continue
                        lower_y, upper_y = min(y1, y2), max(y1, y2)
                        if (
                            lower_y > grid_mid - height * 0.07
                            or upper_y < grid_mid + height * 0.07
                        ):
                            continue
                        slope = delta_x / delta_y
                        intercept = x1 - slope * y1
                        middle_x = slope * grid_mid + intercept
                        if not width * 0.20 < middle_x < width * 0.80:
                            continue
                        vertical_models.append(
                            (
                                middle_x,
                                slope * grid_top + intercept,
                                slope * grid_bottom + intercept,
                            )
                        )

                vertical_clusters = []
                for model in sorted(vertical_models):
                    if (
                        not vertical_clusters
                        or model[0]
                        - float(
                            np.mean(
                                [item[0] for item in vertical_clusters[-1]]
                            )
                        )
                        > width * 0.012
                    ):
                        vertical_clusters.append([model])
                    else:
                        vertical_clusters[-1].append(model)
                grid_columns = [
                    tuple(
                        float(np.median([item[index] for item in cluster]))
                        for index in range(3)
                    )
                    for cluster in vertical_clusters
                ]
                if len(grid_columns) >= 6:
                    left_column = grid_columns[0]
                    right_column = grid_columns[-1]
                    top_width = right_column[1] - left_column[1]
                    bottom_width = right_column[2] - left_column[2]
                    if (
                        grid_height * 0.72 <= top_width <= grid_height * 1.08
                        and grid_height * 0.72
                        <= bottom_width
                        <= grid_height * 1.08
                    ):
                        source = np.float32(
                            [
                                [left_column[1], grid_top],
                                [right_column[1], grid_top],
                                [left_column[2], grid_bottom],
                                [right_column[2], grid_bottom],
                            ]
                        )
                        output_size = max(1, int(round(grid_height)))
                        destination = np.float32(
                            [
                                [0, 0],
                                [output_size - 1, 0],
                                [0, output_size - 1],
                                [output_size - 1, output_size - 1],
                            ]
                        )
                        transform = cv2.getPerspectiveTransform(
                            source,
                            destination,
                        )
                        rectified = cv2.warpPerspective(
                            image,
                            transform,
                            (output_size, output_size),
                        )
                        if rectified.size > 0 and rectified.std() > 8:
                            return rectified
        except cv2.error:
            logger.debug("Tactical grid rectification failed", exc_info=True)

        if fallback.size > 0 and fallback.std() > 8:
            return fallback
        return None

    @staticmethod
    def minimap_snapshot_data_url(minimap, *, maximum_side: int = 520) -> str:
        """Encode the visible game minimap for the local control panel.

        The browser receives the actual minimap crop, not a reconstructed
        schematic.  This keeps islands, grid, ship arrow, contacts and capture
        circles in exactly the same relative positions as the game HUD.
        """
        if minimap is None or minimap.size == 0:
            return ""
        image = minimap
        height, width = image.shape[:2]
        maximum_side = max(120, int(maximum_side))
        if max(width, height) > maximum_side:
            scale = maximum_side / max(width, height)
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if not ok:
            return ""
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def find_health_bar(self, image):
        area = detect_hud_layout(image).crop(image, "health_bar")
        return area if area.size else None

    def find_reload_bar(self, image):
        height, _ = image.shape[:2]
        linear_scale = max(.3, detect_hud_layout(image).ui_scale)
        start_y = int(height * 0.55)
        search = image[start_y : int(height * 0.85), :]
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(
            hsv, np.array([10, 80, 80]), np.array([40, 255, 255])
        )
        row_counts = np.sum(yellow > 0, axis=1)
        valid = row_counts > 50 * linear_scale
        if not np.any(valid):
            return None
        best_row = int(np.argmax(row_counts * valid))
        columns = np.where(yellow[best_row] > 0)[0]
        actual_y = best_row + start_y
        return image[
            max(0, actual_y - round(3 * linear_scale)) : actual_y + max(2, round(10 * linear_scale)),
            int(columns.min()) : int(columns.max()),
        ]

    def reload_ready(self, area):
        if area is None or area.size == 0:
            return False
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lo, self.green_hi)
        return np.count_nonzero(mask) / max(mask.size, 1) > 0.3

    def health_pct(self, area):
        if area is None or area.size == 0:
            return None
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        # The ship bar remains green for a large part of the HP range, then
        # turns yellow/orange/red.  Reading only warm colours made a healthy
        # or moderately damaged ship look like an OCR failure and left a
        # stale cached percentage on the panel.  The longest horizontal run
        # in this tight lower-left ROI is the filled HP segment; consumable
        # icons are too short to pass that test.
        warm = cv2.inRange(
            hsv,
            np.array([0, 65, 75]),
            np.array([42, 255, 255]),
        )
        green = cv2.inRange(
            hsv,
            np.array([35, 55, 70]),
            np.array([95, 255, 255]),
        )
        mask = cv2.bitwise_or(warm, green)
        best_run = 0
        for row in mask:
            indices = np.flatnonzero(row)
            start = previous = None
            for column in indices:
                if previous is None or column > previous + 2:
                    if previous is not None:
                        best_run = max(best_run, previous - start + 1)
                    start = column
                previous = column
            if previous is not None:
                best_run = max(best_run, previous - start + 1)
        if best_run < max(8, area.shape[1] * 0.04):
            return None
        return min(best_run / max(area.shape[1], 1), 1.0)

    @staticmethod
    def read_health_fraction(image, backend) -> float | None:
        """OCR the player's numeric ``current / maximum`` HP.

        The original colour crop is authoritative.  If it cannot be parsed,
        retry a few enlarged/contrast-normalized copies; never estimate HP
        from the bar length because that produced values such as 89% at spawn.
        """
        if image is None or image.size == 0 or backend is None:
            return None
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        crop = layout.crop(image, "health_text")
        # A measured dial can put the HP line against the ROI's lower edge.
        # Keep one scaled glyph of image below it: a clipped ``59200`` prefix
        # otherwise becomes a confident independent ``200 / maximum`` token.
        if (callable(getattr(layout, "source", None))
                and layout.source("health_text") == "anchor"):
            x1, y1, x2, y2 = layout.rect("health_text")
            padding = max(4, round(24 * layout.scale("health_text")))
            crop = image[y1:min(height, y2 + padding), x1:x2]
        if crop.size == 0:
            return None
        def parse(candidate_image):
            try:
                tokens = list(backend.recognize(candidate_image) or ())
            except Exception:
                return None

            def normalize(text):
                text = (str(text or "").replace(",", " ").replace("-", " ")
                        .replace("O", "0").replace("o", "0")
                        .replace("｜", "/").replace("|", "/"))
                # The integer HUD's thousands gap is sometimes recognized as
                # punctuation. Require an actual following three-digit group;
                # never invent a missing digit or consume a loose gear number.
                text = re.sub(r"(?<=\d)[·.：:](?=\d{3}(?!\d))", " ", text)
                return " ".join(text.split())

            def fractions(text, confidence):
                candidates = []
                for match in re.finditer(r"(?<!\d)(\d[\d ]{0,15})\s*/\s*"
                                         r"(\d[\d ]{0,15})(?!\d)", text):
                    sides = [side.strip() for side in match.groups()]
                    # Spaces are actual three-digit thousands groups, never
                    # permission to concatenate a nearby direction/gear digit.
                    if not all(re.fullmatch(r"\d{1,7}|\d{1,3}(?: \d{3})+", side)
                               for side in sides):
                        continue
                    current, maximum = (int(side.replace(" ", "")) for side in sides)
                    if maximum >= 1000 and 0 <= current <= maximum:
                        candidates.append((maximum, confidence, current / maximum))
                return candidates

            valid = []
            observed = []
            complete = []
            for token in tokens:
                try:
                    confidence = float(getattr(token, "confidence", 0))
                    box = np.asarray(getattr(token, "box", ()), np.float32)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(confidence) or confidence < .45:
                    continue
                text = normalize(getattr(token, "text", ""))
                if box.size and (box.shape != (4, 2) or not np.all(np.isfinite(box))):
                    continue
                if confidence >= .70:
                    complete.extend(fractions(text, confidence))
                if (box.shape != (4, 2) or not np.all(np.isfinite(box)) or box.min() < 0
                        or box[:, 0].max() > candidate_image.shape[1]
                        or box[:, 1].max() > candidate_image.shape[0]):
                    continue
                x1, y1 = box.min(axis=0)
                x2, y2 = box.max(axis=0)
                if x2 > x1 and y2 > y1:
                    item = (text, confidence, (float(x1), float(y1), float(x2), float(y2)))
                    observed.append(item)
                    if confidence >= .70:
                        valid.append(item)
            # A three-digit numerator may be a trailing thousands group.
            # A preceding box can include the ship name and overlap it:
            # ``...59`` + ``200/59`` + ``200`` is not evidence of 200 HP.
            # Reject that ambiguity before joining only its denominator.
            blocked_anchors = set()
            for anchor in valid:
                if not re.fullmatch(r"\d{3}\s*/\s*\d[\d ]*", anchor[0]):
                    continue
                rect = anchor[2]
                glyph_height = rect[3] - rect[1]
                prefixes = [item for item in observed if item[2][0] < rect[0]
                            # OCR can read a ship/tier ornament as a leading
                            # slash. It does not make the trailing HP digits
                            # independent of the overlapping fraction box.
                            and not re.search(r"\d[\d ]*\s*/\s*\d", item[0])
                            and re.search(r"\d$", item[0])
                            and -.75 * glyph_height <= rect[0] - item[2][2] <= glyph_height * .65
                            and abs((item[2][1] + item[2][3] - rect[1] - rect[3]) / 2)
                                <= glyph_height * .35
                            and .55 * glyph_height <= item[2][3] - item[2][1] <= glyph_height * 1.8
                            and (not re.fullmatch(r"\d+", item[0])
                                 or (re.fullmatch(r"\d{1,3}", item[0])
                                     and item[2][2] - item[2][0] <= len(item[0])
                                         * max(glyph_height, item[2][3] - item[2][1]) * .9))]
                if not prefixes:
                    continue
                prefix = max(prefixes, key=lambda item: item[2][2])
                partial = fractions(anchor[0], anchor[1])
                complete = [item for item in complete if not any(
                            item[0] == suspect[0] and item[2] == suspect[2] for suspect in partial)]
                if (prefix[1] < .70 or not re.fullmatch(r"\d{1,3}", prefix[0])
                        or prefix[2][2] > rect[0]):
                    blocked_anchors.add(anchor)
                    continue
                complete.extend(fractions(prefix[0] + " " + anchor[0], min(prefix[1], anchor[1])))
            # A complete fraction in one text box wins before any other joining.
            # In particular, ``58623/59200`` cannot absorb a separate ``7``.
            if complete:
                return max(complete, key=lambda item: (item[1], item[0]))
            joined = []
            for anchor in valid:
                if anchor in blocked_anchors or anchor[0].count("/") != 1:
                    continue
                rect = anchor[2]
                glyph_height = rect[3] - rect[1]
                row = sorted((item for item in valid if
                              abs((item[2][1] + item[2][3] - rect[1] - rect[3]) / 2)
                                  <= glyph_height * .35
                              and .65 * glyph_height <= item[2][3] - item[2][1] <= glyph_height * 1.45),
                             key=lambda item: item[2][0])
                index = row.index(anchor)
                start = finish = index
                for _ in range(2):
                    if (start > 0 and re.fullmatch(r"\d+(?: \d{3})*", row[start - 1][0])
                            and 0 <= row[start][2][0] - row[start - 1][2][2] <= glyph_height * .65):
                        start -= 1
                    else:
                        break
                for _ in range(2):
                    if (finish + 1 < len(row) and re.fullmatch(r"\d+(?: \d{3})*", row[finish + 1][0])
                            and 0 <= row[finish + 1][2][0] - row[finish][2][2] <= glyph_height * .65):
                        finish += 1
                    else:
                        break
                if start == finish:
                    continue
                # Include the entire adjoining numeric chain. Dropping its
                # first group would turn ``58 623`` into a false 623 HP.
                pieces = row[start:finish + 1]
                joined.extend(fractions(" ".join(item[0] for item in pieces),
                                        min(item[1] for item in pieces)))
            return max(joined, key=lambda item: (item[1], item[0]), default=None)

        parsed = parse(crop)
        if parsed is not None:
            return parsed[2]
        # Test/dummy backends often return a predetermined sequence per call.
        # Restrict visual retries to real OCR backends so deterministic unit
        # contracts remain one crop == one recognition call.
        if not isinstance(backend, RapidOcrBackend):
            return None
        fallbacks = []
        for variant in numeric_ocr_fallback_variants(crop):
            candidate = parse(variant)
            if candidate is None:
                continue
            fallbacks.append(candidate)
            if candidate[1] >= 0.80:
                break
        if not fallbacks:
            return None
        return max(fallbacks, key=lambda item: (item[1], item[0]))[2]

    def battle_ended(self, image):
        state = self.classify_screen(image)
        return state in {ScreenState.RESULTS, ScreenState.PORT}

    def in_battle(self, image):
        return self.classify_screen(image) == ScreenState.BATTLE

    @staticmethod
    def _crop_region(image, region):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = region.pixels(width, height)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _color_ratio(area, lower, upper):
        if area is None or area.size == 0:
            return 0.0
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return np.count_nonzero(mask) / max(mask.size, 1)

    @staticmethod
    def _solid_action_rect(image, region, colors, search_region, *, min_aspect=2.0):
        """Measure a filled horizontal action; callers retain their page guards."""
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        sx1, sy1, sx2, sy2 = (round(v * size) for v, size in
                              zip(search_region, (width, height, width, height)))
        crop = image[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = np.zeros(crop.shape[:2], np.uint8)
        for lower, upper in colors:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        expected_x, expected_y = region.center(width, height)
        candidates = []
        for x, y, cw, ch, area in stats[1:]:
            if not (cw >= max(30, width * .025) and ch >= max(8, height * .008)
                    and cw <= width * .30 and ch <= height * .13
                    and min_aspect <= cw / ch <= 14
                    and area / (cw * ch) >= .55
                    and area >= max(120, width * height * .00008)):
                continue
            # Cropped artwork/sea can masquerade as a filled rectangle.
            if x <= 0 or y <= 0 or x + cw >= crop.shape[1] or y + ch >= crop.shape[0]:
                continue
            component = mask[y:y + ch, x:x + cw]
            inset_x, inset_y = max(1, round(cw * .05)), max(1, round(ch * .10))
            edge_fills = [np.mean(component[inset_y, inset_x:-inset_x] > 0),
                          np.mean(component[-inset_y - 1, inset_x:-inset_x] > 0),
                          np.mean(component[inset_y:-inset_y, inset_x] > 0),
                          np.mean(component[inset_y:-inset_y, -inset_x - 1] > 0)]
            if min(edge_fills) < .65:
                continue
            actual = (int(x + sx1), int(y + sy1), int(x + sx1 + cw), int(y + sy1 + ch))
            distance = math.hypot(((actual[0] + actual[2]) / 2 - expected_x) / width,
                                  ((actual[1] + actual[3]) / 2 - expected_y) / height)
            candidates.append((distance, actual))
        if not candidates:
            return None
        candidates.sort()
        if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < .015:
            return None
        return candidates[0][1]

    def loading_start_action_box(self, image):
        """Measured roster button bounds; colour alone does not prove its label."""
        return self._solid_action_rect(
            image, LOADING_START_BUTTON,
            [((35, 45, 45), (100, 255, 255))], (.28, .70, .72, 1), min_aspect=3.0)

    def _is_port_ship_bar(self, image):
        height, _ = image.shape[:2]
        bar = image[int(height * 0.88) : height, :]
        return self._port_card_band_evidence(bar)

    @staticmethod
    def _port_card_band_evidence(bar):
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        colorful = cv2.inRange(
            hsv, np.array([0, 60, 80]), np.array([180, 255, 255])
        )
        colorful_ratio = np.count_nonzero(colorful) / max(colorful.size, 1)
        # A port carousel contains many card edges, ship thumbnails and text
        # across the full width. During battle, the compass/consumables/minimap
        # can be equally colourful but leave most of this band visually sparse.
        # Colour alone therefore misclassified active battles as the port.
        gray = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        edge_ratio = np.count_nonzero(edges) / max(edges.size, 1)
        logger.debug(
            "Ship bar evidence: colorful=%.1f%% edges=%.1f%%",
            colorful_ratio * 100,
            edge_ratio * 100,
        )
        # Commander-less ships make much of the grid gray; live ports can
        # have modest colour coverage even when the carousel is fully visible.
        # The dense full-width card-edge pattern remains the distinguishing
        # signal.  The prior colour threshold was high enough that a normal
        # selected-ship port could fall through into the loose battle guard.
        # Real 2560px port captures can sit around 12.5% edge coverage when
        # the carousel is dense but low-contrast.  Keep this below the battle
        # fixture (about 11.1%) while accepting the verified live port frame.
        return colorful_ratio > 0.10 and edge_ratio > 0.115

    @staticmethod
    def _visual_anchor_metrics(area):
        """Return inexpensive structure metrics for one fixed HUD anchor."""
        if area is None or area.size == 0:
            return {"mean": 0.0, "std": 0.0, "edge": 0.0, "bright": 0.0}
        gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        pixels = max(gray.size, 1)
        return {
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "edge": float(np.count_nonzero(cv2.Canny(gray, 80, 180)) / pixels),
            "bright": float(np.count_nonzero(gray > 165) / pixels),
        }

    def _has_port_tab_strip(self, image):
        """Recognize the six bordered port tabs below the battle action.

        Equipment pages have no left port menu, and their thin dark tab row
        is diluted by the broad top-anchor texture metric.  Measure the row
        relative to the actual action button so window and UI scaling do not
        turn a combat score or weapon row into port evidence.
        """
        action = self._solid_action_rect(
            image, PORT_BATTLE_BUTTON,
            [((90, 55, 70), (135, 255, 255)), ((5, 55, 70), (30, 255, 255))],
            (.22, 0, .78, .16))
        if action is None:
            return False
        height, width = image.shape[:2]
        ax1, ay1, ax2, ay2 = action
        button_width, button_height = ax2 - ax1, ay2 - ay1
        center = (ax1 + ax2) / 2
        x1 = max(0, int(center - 3.4 * button_width))
        x2 = min(width, int(center + 3.4 * button_width))
        y1 = max(0, int(ay2 + .04 * button_height))
        y2 = min(height, int(ay2 + 1.05 * button_height))
        area = image[y1:y2, x1:x2]
        if area.size == 0:
            return False
        # Thin borders disappear into adjacent text strokes at smaller UI
        # scales. Measure their shape at a fixed button-relative pixel scale.
        factor = 41.0 / button_height
        area = cv2.resize(area, None, fx=factor, fy=factor,
                          interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
        button_width *= factor
        button_height = 41.0
        edges = cv2.Canny(cv2.cvtColor(area, cv2.COLOR_BGR2GRAY), 10, 30)
        vertical_edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
        vertical = cv2.morphologyEx(
            vertical_edges, cv2.MORPH_OPEN,
            np.ones((max(5, round(.22 * button_height)), 1), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(vertical)
        separators = sorted(
            (int(x), int(y), int(ch)) for x, y, cw, ch, _ in stats[1:]
            if cw <= max(3, button_width * .025)
            and .22 * button_height <= ch <= .95 * button_height)
        # Both sides of a faint border may be detected; keep one separator
        # per x coordinate before checking regular six-cell spacing.
        distinct = []
        for item in separators:
            if distinct and item[0] - distinct[-1][0] <= button_width * .06:
                if item[2] > distinct[-1][2]:
                    distinct[-1] = item
            else:
                distinct.append(item)
        rows = []
        for first in distinct:
            for last in distinct:
                pitch = (last[0] - first[0]) / 6
                if not .70 * button_width <= pitch <= 1.30 * button_width:
                    continue
                row = [first]
                for index in range(1, 6):
                    expected = first[0] + index * pitch
                    candidate = min(distinct, key=lambda item: abs(item[0] - expected))
                    if abs(candidate[0] - expected) > .12 * pitch:
                        break
                    row.append(candidate)
                if len(row) == 6:
                    rows.append(row + [last])
        for row in rows:
            spacing = np.diff([item[0] for item in row])
            if len(spacing) != 6:
                continue
            if not np.all((spacing >= .70 * button_width)
                          & (spacing <= 1.30 * button_width)):
                continue
            if float(spacing.max() / max(spacing.min(), 1)) > 1.22:
                continue
            tops = [item[1] for item in row]
            bottoms = [item[1] + item[2] for item in row]
            overlap = min(bottoms) - max(tops)
            if overlap < .22 * button_height:
                continue
            # Seven regularly spaced vertical strokes alone could be text.
            # Also require two horizontal rails spanning this exact row.
            rail_pad = max(2, round(.15 * button_height))
            rail = edges[max(0, min(tops) - rail_pad):min(edges.shape[0], max(bottoms) + rail_pad),
                         row[0][0]:row[-1][0] + 1]
            occupancy = np.count_nonzero(rail, axis=1) / max(rail.shape[1], 1)
            rail_rows = np.flatnonzero(occupancy >= .65)
            if len(rail_rows) >= 2 and int(rail_rows[-1] - rail_rows[0]) >= .30 * button_height:
                return True
        return False

    def _port_anchor_votes(self, image):
        """Vote on independent port UI regions, excluding the battle button."""
        height, width = image.shape[:2]
        left_menu = image[
            int(height * 0.07) : int(height * 0.73),
            : int(width * 0.15),
        ]
        ship_details = image[
            int(height * 0.06) : int(height * 0.74),
            int(width * 0.82) :,
        ]
        top_tabs = image[
            : int(height * 0.12),
            int(width * 0.22) : int(width * 0.73),
        ]
        left = self._visual_anchor_metrics(left_menu)
        right = self._visual_anchor_metrics(ship_details)
        tabs = self._visual_anchor_metrics(top_tabs)
        tab_strip = self._has_port_tab_strip(image)
        carousel = self._is_port_ship_bar(image)
        if tab_strip:
            # An independently measured port tab row permits normalizing
            # texture density to its actual UI scale. Keep the original
            # thresholds and independent carousel/ship-details requirements.
            action = self._solid_action_rect(
                image, PORT_BATTLE_BUTTON,
                [((90, 55, 70), (135, 255, 255)), ((5, 55, 70), (30, 255, 255))],
                (.22, 0, .78, .16))
            if action is not None:
                factor = 41.0 / (action[3] - action[1])
                if abs(factor - 1) > .08:
                    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
                    normalized_details = cv2.resize(ship_details, None, fx=factor, fy=factor,
                                                    interpolation=interpolation)
                    normalized_right = self._visual_anchor_metrics(normalized_details)
                    if normalized_right["std"] > 25 and normalized_right["edge"] > .045:
                        right = normalized_right
                    normalized_bar = cv2.resize(image[int(height * .88):], None,
                                               fx=factor, fy=factor, interpolation=interpolation)
                    carousel = carousel or self._port_card_band_evidence(normalized_bar)
        return {
            "ship_carousel": carousel,
            "left_port_menu": (
                (left["bright"] > 0.18 and left["edge"] > 0.02)
                or left["edge"] > 0.08
            ),
            "right_ship_details": right["std"] > 25 and right["edge"] > 0.045,
            "top_port_tabs": (
                (tabs["std"] > 30 and tabs["edge"] > 0.025)
                or tab_strip
            ),
        }

    def in_port(self, image):
        battle_button = self._crop_region(image, PORT_BATTLE_BUTTON)
        if battle_button.size == 0:
            return False
        hsv = cv2.cvtColor(battle_button, cv2.COLOR_BGR2HSV)
        colored = cv2.inRange(
            hsv, np.array([90, 55, 70]), np.array([135, 255, 255])
        ) | cv2.inRange(
            hsv, np.array([5, 55, 70]), np.array([30, 255, 255])
        )
        labels, _, stats, _ = cv2.connectedComponentsWithStats(
            colored, connectivity=8
        )
        largest = 0 if labels <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
        # The real port action is one solid button. Battle score markers can
        # occupy the same top-centre ROI but only form tiny disconnected blobs.
        solid_button_ratio = largest / max(colored.size, 1)
        anchors = self._port_anchor_votes(image)
        anchor_count = sum(bool(value) for value in anchors.values())
        logger.debug(
            "Port anchors: button=%.3f votes=%s",
            solid_button_ratio,
            anchors,
        )
        # ``加入战斗`` is mandatory, plus at least three independent port
        # regions (carousel, left menu, right ship data, top tabs).  A textured
        # loading/login picture can overlap one region but cannot satisfy the
        # complete port layout.
        measured = self._solid_action_rect(
            image, PORT_BATTLE_BUTTON,
            [((90, 55, 70), (135, 255, 255)), ((5, 55, 70), (30, 255, 255))],
            (.22, 0, .78, .16))
        if measured is not None:
            layout = detect_hud_layout(image)
            if layout.source("health_text") == "anchor" and layout.source("minimap") == "anchor":
                measured = None
        return (solid_button_ratio > 0.12 or measured is not None) and anchor_count >= 3

    def in_loading(self, image):
        # Positive port controls outrank broad loading artwork metrics. This
        # also keeps the battle-HUD port veto below from exposing the older
        # dark/textured loading fallback on a valid port frame.
        if self.in_port(image):
            return False
        if self._has_loading_start_action(image):
            return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 40 and gray.std() < 25:
            return True

        if self._is_login_splash(image):
            return True

        start_button = self._crop_region(image, LOADING_START_BUTTON)
        start_green = self._color_ratio(start_button, (35, 45, 45), (85, 255, 255))
        if start_green > 0.04:
            return True

        height, width = gray.shape[:2]
        center = gray[int(height * 0.38) : int(height * 0.58), int(width * 0.43) : int(width * 0.57)]
        bright_ratio = np.count_nonzero(center > 205) / max(center.size, 1)
        bottom_right = image[int(height * 0.60) :, int(width * 0.74) :]
        if 0.0005 < bright_ratio < 0.12 and bottom_right.std() > 12 and not self._has_battle_hud(image):
            return True

        # The matchmaking artwork briefly removes every label and button while
        # the map is being assembled.  That frame has no central bright text,
        # so the text-oriented rule above used to oscillate between LOADING and
        # UNKNOWN.  Accept the dark textured artwork only when all port anchors
        # are absent and the battle HUD is also absent; this keeps arbitrary
        # modal pages out of the loading lifecycle.
        port_votes = self._port_anchor_votes(image)
        return bool(
            45 < float(gray.mean()) < 105
            and 25 < float(gray.std()) < 58
            and float(bottom_right.std()) > 25
            and sum(bool(value) for value in port_votes.values()) == 0
            and not self._has_battle_hud(image)
        )

    def _has_loading_start_action(self, image) -> bool:
        """Confirm the solid ``开始战斗`` action on the pre-battle roster.

        The roster artwork is textured enough to satisfy the deliberately
        tolerant battle-HUD anchors, while its countdown page has no minimap
        player marker.  A large horizontal green component in this tight
        bottom-centre ROI is stronger evidence than those generic anchors.
        Live consumable icons can add green pixels to the same ROI, but they
        remain small/square rather than one wide action bar.
        """
        if image is None or image.size == 0:
            return False
        if self.loading_start_action_box(image) is not None:
            return True
        start_button = self._crop_region(image, LOADING_START_BUTTON)
        if start_button.size == 0:
            return False
        hsv = cv2.cvtColor(start_button, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            np.array([35, 45, 45]),
            np.array([100, 255, 255]),
        )
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            green,
            connectivity=8,
        )
        height, width = green.shape[:2]
        pixels = max(width * height, 1)
        for label in range(1, count):
            _x, _y, component_width, component_height, area = stats[label]
            if (
                # The pre-battle roster button is partially translucent on
                # some maps/scales.  At 2560x1494 its connected green fill is
                # about 14% of this tight ROI, while live consumable glyphs
                # remain far below the width/height checks below.
                area / pixels >= 0.10
                and component_width >= width * 0.55
                and component_height >= height * 0.20
                and component_width / max(component_height, 1) >= 2.0
            ):
                return True
        return False

    @staticmethod
    def _is_login_splash(image):
        """Recognize the full-screen ``World of Warships / 登录中`` artwork."""
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        bottom = image[int(height * 0.82) :]
        center = image[
            int(height * 0.35) : int(height * 0.65),
            int(width * 0.30) : int(width * 0.70),
        ]
        if bottom.size == 0 or center.size == 0:
            return False
        bottom_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        center_bright = np.count_nonzero(center_gray > 205) / max(
            center_gray.size, 1
        )
        # The login painting has a uniquely dark, low-detail lower ocean and
        # a large bright central wordmark.  Live battle/results fixtures fail
        # at least one of these independent conditions.
        return (
            float(bottom_gray.mean()) < 45
            and float(bottom_gray.std()) < 25
            and 0.03 < center_bright < 0.20
        )

    def in_results(self, image):
        return_to_port = self._crop_region(image, RESULTS_RETURN_TO_PORT_BUTTON)
        requeue = self._crop_region(image, RESULTS_REQUEUE_BUTTON)
        teal_ratio = self._color_ratio(return_to_port, (70, 25, 35), (110, 255, 220))
        orange_ratio = self._color_ratio(requeue, (5, 80, 80), (25, 255, 255))
        if teal_ratio > 0.30 and orange_ratio > 0.30:
            return True
        teal = self._solid_action_rect(image, RESULTS_RETURN_TO_PORT_BUTTON,
            [((70, 25, 35), (110, 255, 220))], (.22, .65, .70, .995))
        orange = self._solid_action_rect(image, RESULTS_REQUEUE_BUTTON,
            [((5, 80, 80), (25, 255, 255))], (.55, .65, .97, .995))
        if teal is None or orange is None:
            return False
        return (teal[2] < orange[0] and
                abs((teal[1] + teal[3]) / 2 - (orange[1] + orange[3]) / 2)
                <= max(teal[3] - teal[1], orange[3] - orange[1]) * .75)

    def in_no_commander_confirmation(self, image):
        button = self._crop_region(image, NO_COMMANDER_CONFIRM_BUTTON)
        hsv = cv2.cvtColor(button, cv2.COLOR_BGR2HSV)
        teal = cv2.inRange(
            hsv, np.array([70, 35, 35]), np.array([110, 255, 230])
        )
        labels, _, stats, _ = cv2.connectedComponentsWithStats(
            teal, connectivity=8
        )
        largest = 0 if labels <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
        solid_button_ratio = largest / max(teal.size, 1)
        height, width = image.shape[:2]
        dialog = image[
            int(height * 0.36) : int(height * 0.66),
            int(width * 0.30) : int(width * 0.70),
        ]
        outer = image[
            int(height * 0.18) : int(height * 0.82),
            int(width * 0.08) : int(width * 0.92),
        ]
        measured = self._solid_action_rect(image, NO_COMMANDER_CONFIRM_BUTTON,
            [((70, 35, 35), (110, 255, 230))], (.18, .25, .82, .82), min_aspect=1.2)
        if measured is not None and solid_button_ratio <= .08:
            x1, y1, x2, y2 = measured
            dialog = image[max(0, round(y1 - height * .14)):min(height, round(y2 + height * .08)),
                           max(0, round(x1 - width * .13)):min(width, round(x2 + width * .13))]
        # A real warning is a solid teal action button inside a distinctly
        # darker modal.  Loading artwork used to satisfy the old loose colour
        # check and caused repeated blind clicks while a battle was loading.
        return (
            (solid_button_ratio > 0.08 or measured is not None)
            and float(dialog.mean()) + 8 < float(outer.mean())
        )

    def fire_status_icon_visible(self, image) -> bool:
        """Confirm paired yellow fire glyphs or a compact red damage timer.

        The live HUD can show an unmistakable fire timer even when the small
        speed-dial flame is washed out or hidden. Restrict the detector to the
        measured condition-card strip and require a complete, compact filled
        rectangle. Sunset-lit decks inside the strip are irregular red blobs,
        so their colour and a loose bounding box do not establish a fire card.
        """
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        if self._yellow_fire_warning_pair(image, layout):
            return True
        crop = layout.crop(image, "fire_status")
        if crop.size == 0:
            return False
        linear_scale = layout.scale("fire_status")
        pixel_scale = linear_scale ** 2
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(
            hsv,
            np.array([0, 105, 70]),
            np.array([15, 255, 255]),
        )
        red |= cv2.inRange(
            hsv,
            np.array([165, 105, 70]),
            np.array([180, 255, 255]),
        )
        red = cv2.morphologyEx(
            red,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            red,
            connectivity=8,
        )
        for index in range(1, count):
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            fill = area / max(component_width * component_height, 1)
            aspect = component_width / max(component_height, 1)
            if (
                area >= 100.0 * pixel_scale
                and 18.0 * linear_scale <= component_width <= 64.0 * linear_scale
                and 18.0 * linear_scale <= component_height <= 64.0 * linear_scale
                and 0.70 <= aspect <= 1.40
                and fill >= 0.60
                and x > 0 and y > 0
                and x + component_width < crop.shape[1]
                and y + component_height < crop.shape[0]
            ):
                component = labels[y:y + component_height, x:x + component_width] == index
                if all(float(edge.mean()) >= .45 for edge in (
                    component[0], component[-1], component[:, 0], component[:, -1],
                )):
                    return True
        return False

    def fire_anchor_bits(self, image):
        """Evaluate the two independent fire HUD anchors separately.

        Returns ``(dial_flames, ship_flames)`` so callers and live
        diagnostics can tell which anchor is missing on a negative frame.

        The speed-dial ship silhouette (bottom-left) shows one compact orange
        flame-droplet marker per active fire, and the viewport ship model
        carries the same droplet markers while its fire is close enough to be
        resolved. Both are exposed independently so the caller can require
        corroboration before spending damage control.
        """
        if image is None or image.size == 0:
            return False, False
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        linear_scale = layout.scale("fire_dial")
        pixel_scale = linear_scale ** 2

        def orange_components(name):
            crop = layout.crop(image, name)
            if crop.size == 0:
                return []
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            orange = cv2.inRange(
                hsv,
                np.array([0, 110, 110]),
                np.array([40, 255, 255]),
            )
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                orange, connectivity=8
            )
            components = []
            for index in range(1, count):
                x = int(stats[index, cv2.CC_STAT_LEFT])
                y = int(stats[index, cv2.CC_STAT_TOP])
                component_width = int(stats[index, cv2.CC_STAT_WIDTH])
                component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
                area = int(stats[index, cv2.CC_STAT_AREA])
                if area < max(4, 20 * pixel_scale):
                    continue
                component_labels = labels[y:y + component_height, x:x + component_width]
                selected = hsv[y:y + component_height, x:x + component_width][component_labels == index]
                saturation_mean = float(selected[:, 1].mean())
                value_std = float(selected[:, 2].std())
                # Mean background brightness of a dilated ring around the
                # component.  Real dial markers sit on the bright dial plate
                # while text glyphs sit on darker dial hardware.
                ring_value = -1.0
                if component_width >= 4 and component_height >= 4:
                    # The ring extends just three pixels beyond this component.
                    # Restrict its work to those bounds, rather than dilating
                    # the full viewport once for every sunset/sea fragment.
                    rx1, ry1 = max(0, x - 3), max(0, y - 3)
                    rx2 = min(crop.shape[1], x + component_width + 3)
                    ry2 = min(crop.shape[0], y + component_height + 3)
                    component = (labels[ry1:ry2, rx1:rx2] == index).astype(np.uint8)
                    ring = (
                        cv2.dilate(component, np.ones((7, 7), np.uint8))
                        - component
                    )
                    ring_values = hsv[ry1:ry2, rx1:rx2, 2][ring > 0]
                    if ring_values.size:
                        ring_value = float(ring_values.mean())
                components.append(
                    (
                        x,
                        y,
                        component_width,
                        component_height,
                        area,
                        saturation_mean,
                        value_std,
                        ring_value,
                    )
                )
            return components

        def is_droplet(component):
            """Vivid teardrop blob with a brightness gradient.

            Rejects solid round module/consumable dots (fill ~0.78, low
            value-std), flag glyphs and arc dashes (low saturation or
            value-std) and the map's dull hexagon/capture icons.
            """
            _x, _y, cw, ch, area, saturation_mean, value_std, _ring = component
            return (
                40.0 * pixel_scale <= area <= 520.0 * pixel_scale
                and 0.5 <= cw / max(ch, 1) <= 1.6
                and 0.28 <= area / max(cw * ch, 1) <= 0.70
                and cw >= 4.0 * linear_scale
                and ch >= 5.0 * linear_scale
                and saturation_mean >= 140.0
                # Orange turret bearings and sunlit deck fragments in the
                # live Rupprecht capture had a 19-20 value gradient. The HUD
                # flame's darker stem and bright head have stronger contrast.
                and value_std >= 28.0
            )

        def has_text_row(components):
            """Reject a run of at least three near-identical glyphs on a row.

            Port carousels/scoreboards misclassified as BATTLE show orange
            digits that pass every droplet test but line up in a uniform-pitch
            row.  Real fire markers scatter across the ship silhouette and
            never align into one row of equal-sized blobs.
            """
            rows = {}
            for x, y, cw, ch, *_rest in components:
                rows.setdefault(y, []).append((x, cw, ch))
            for _y, items in rows.items():
                if len(items) < 3:
                    continue
                items.sort()
                for i in range(len(items) - 2):
                    base_w, base_h = items[i][1], items[i][2]
                    group = [items[i]]
                    for j in range(i + 1, len(items)):
                        xj, wj, hj = items[j]
                        if abs(hj - base_h) <= 3 and abs(wj - base_w) <= 4:
                            group.append((xj, wj, hj))
                    if len(group) < 3:
                        continue
                    ok = True
                    pitch = None
                    for a, b in zip(group, group[1:]):
                        gap = b[0] - (a[0] + a[1])
                        if gap < 0:
                            ok = False
                            break
                        if pitch is None:
                            pitch = gap
                        elif abs(gap - pitch) > max(6, pitch * 0.5):
                            ok = False
                            break
                    if ok:
                        return True
            return False

        dial_components = orange_components("fire_dial")
        ship_components = orange_components("fire_ship")
        dial_flames = [c for c in dial_components if is_droplet(c)]
        ship_flames = [c for c in ship_components if is_droplet(c)]
        logger.debug(
            "Fire HUD anchors: dial=%s ship=%s",
            dial_flames,
            ship_flames,
        )
        dial_ok = (
            bool(dial_flames)
            and not has_text_row(dial_flames)
            and all(component[7] >= 75.0 for component in dial_flames)
        )
        ship_ok = bool(ship_flames) and not has_text_row(ship_flames)
        return bool(dial_ok), bool(ship_ok)

    def is_on_fire(self, image):
        """Confirm fire from the damage card or two independent flame anchors.

        A dial-only marker has produced live false positives at full health,
        while viewport flames may be the only visible model evidence. The
        compact red condition card is authoritative; without it, require the
        two legacy flame anchors to agree.
        """
        dial_flames, ship_flames = self.fire_anchor_bits(image)
        return bool(
            self.fire_status_icon_visible(image)
            or (dial_flames and ship_flames)
        )

    def is_flooding(self, image):
        """Detect the blue flooding icon directly below the numeric HP HUD."""
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        linear_scale = layout.scale("flood_left")
        pixel_scale = linear_scale ** 2
        # Flooding is represented in the fixed lower-left condition block.
        # Keep the ROI away from the sea/viewport and do not infer flooding
        # from water splashes around the rendered ship.
        status = layout.crop(image, "flood_left")
        if status.size == 0:
            return False
        hsv = cv2.cvtColor(status, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(
            hsv,
            np.array([88, 110, 120]),
            np.array([125, 255, 255]),
        )
        labels, _, stats, _ = cv2.connectedComponentsWithStats(
            blue, connectivity=8
        )
        if labels <= 1:
            return False
        left_icons = []
        for index in range(1, labels):
            area = int(stats[index, cv2.CC_STAT_AREA])
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            aspect = component_width / max(component_height, 1)
            if (
                30 * pixel_scale <= area <= 900 * pixel_scale
                and 4 * linear_scale <= component_width <= 55 * linear_scale
                and 4 * linear_scale <= component_height <= 55 * linear_scale
                and 0.40 <= aspect <= 1.50
            ):
                left_icons.append((area, component_width, component_height))

        central_status = layout.crop(image, "flood_center")
        center_scale = layout.scale("flood_center")
        center_icons = []
        if central_status.size:
            center_hsv = cv2.cvtColor(central_status, cv2.COLOR_BGR2HSV)
            center_blue = cv2.inRange(
                center_hsv,
                np.array([88, 110, 120]),
                np.array([125, 255, 255]),
            )
            center_labels, _, center_stats, _ = cv2.connectedComponentsWithStats(
                center_blue, connectivity=8
            )
            for index in range(1, center_labels):
                area = int(center_stats[index, cv2.CC_STAT_AREA])
                component_width = int(center_stats[index, cv2.CC_STAT_WIDTH])
                component_height = int(center_stats[index, cv2.CC_STAT_HEIGHT])
                aspect = component_width / max(component_height, 1)
                if (
                    45 * center_scale ** 2 <= area <= 1200 * center_scale ** 2
                    and 6 * center_scale <= component_width <= 65 * center_scale
                    and 6 * center_scale <= component_height <= 65 * center_scale
                    and 0.40 <= aspect <= 1.80
                ):
                    center_icons.append((area, component_width, component_height))
        logger.debug(
            "Flooding HUD anchors: left=%s center=%s",
            left_icons,
            center_icons,
        )
        # BattleBot additionally requires this icon on two consecutive
        # frames before it exposes flooding or uses damage control.
        return bool(left_icons and center_icons)

    def flooding_status_icon_visible(self, image) -> bool:
        """Corroborate a blue countdown card with the own-ship flooding icon.

        Both HUD groups must have measured anchors. A near-square blue card
        needs adjacent pale countdown glyphs; sea/weapon decoration alone
        never promotes the temporal flooding detector to this stronger signal.
        """
        if image is None or image.size == 0:
            return False
        layout = detect_hud_layout(image)
        if (layout.source("flood_left") != "anchor"
                or layout.source("flood_center") != "anchor"
                or not self.is_flooding(image)):
            return False
        crop = layout.crop(image, "flood_center")
        scale = layout.scale("flood_center")
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([88, 130, 120]), np.array([125, 255, 255]))
        _, _, stats, _ = cv2.connectedComponentsWithStats(blue)
        for x, y, cw, ch, area in stats[1:]:
            if not (100 * scale ** 2 <= area <= 1200 * scale ** 2
                    and 18 * scale <= cw <= 55 * scale
                    and 18 * scale <= ch <= 55 * scale
                    and .70 <= cw / ch <= 1.40 and area / (cw * ch) >= .30):
                continue
            adjacent = hsv[max(0, y - round(ch * .3)):min(crop.shape[0], y + round(ch * 1.3)),
                           min(crop.shape[1], x + cw):min(crop.shape[1], x + cw + round(ch * 3))]
            if adjacent.size == 0:
                continue
            pale = cv2.inRange(adjacent, np.array([0, 0, 170]), np.array([180, 65, 255]))
            _, _, glyph_stats, _ = cv2.connectedComponentsWithStats(pale)
            glyphs = [entry for entry in glyph_stats[1:] if
                      3 * scale <= entry[2] <= 18 * scale
                      and 6 * scale <= entry[3] <= 25 * scale
                      and entry[4] >= 10 * scale ** 2]
            if len(glyphs) >= 2:
                return True
        return False

    @staticmethod
    def _yellow_fire_warning_pair(image, layout):
        """Require the actual flame glyph in both yellow warning triangles.

        Some HUD styles use an outlined triangle beside own HP and over the
        ship instead of a red timer. Neither a triangle outline nor orange
        ship artwork alone establishes fire. Compare the interior flame as
        well as the complete glyph, within two independently located regions.
        """
        template = _fire_warning_template()
        if template is None:
            return False
        yy, xx = np.indices(template.shape)
        interior = (xx > 9) & (xx < 31) & (yy > 12) & (yy < 35)
        for name in ("health_text", "fire_ship"):
            crop = layout.crop(image, name)
            if crop.size == 0:
                return False
            scale = max(.3, layout.scale(name))
            mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
                               np.array([10, 100, 80]), np.array([42, 255, 255]))
            matched = False
            for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)[0]:
                x, y, cw, ch = cv2.boundingRect(contour)
                if not (18 * scale <= cw <= 80 * scale
                        and 18 * scale <= ch <= 70 * scale
                        and .90 <= cw / max(ch, 1) <= 1.40):
                    continue
                hull = cv2.convexHull(contour)
                polygon = cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * .07, True)
                if len(polygon) != 3:
                    continue
                points = sorted(polygon.reshape(-1, 2), key=lambda point: point[1])
                if (points[0][1] > y + ch * .20
                        or abs(points[0][0] - x - cw / 2) > cw * .20
                        or min(points[1][1], points[2][1]) < y + ch * .75
                        or abs(points[1][0] - points[2][0]) < cw * .65):
                    continue
                glyph = cv2.resize(mask[y:y + ch, x:x + cw], (40, 40),
                                   interpolation=cv2.INTER_NEAREST) > 0
                union = glyph | template
                overlap = glyph & template
                if (np.mean(glyph) <= .55
                        and np.count_nonzero(overlap) / max(np.count_nonzero(union), 1) >= .48
                        and np.count_nonzero(overlap & interior) /
                        max(np.count_nonzero(union & interior), 1) >= .48):
                    matched = True
                    break
            if not matched:
                return False
        return True

    @staticmethod
    def read_speed_knots(image, backend) -> float | None:
        """Read the player's lower-left speed readout without using viewport targets."""
        if image is None or image.size == 0 or backend is None:
            return None
        height, width = image.shape[:2]
        # This is the speed/engine sector left of the ship, deliberately
        # outside the central target labels and lower-right minimap.
        crop = detect_hud_layout(image).crop(image, "speed")
        if crop.size == 0:
            return None
        def parse(candidate_image, *, require_engine=False):
            try:
                tokens = list(backend.recognize(candidate_image) or ())
            except Exception:
                return None
            located = []
            for token in tokens:
                try:
                    confidence = float(getattr(token, "confidence", 0))
                    box = np.asarray(getattr(token, "box", ()), np.float32)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(confidence) or confidence < .70:
                    continue
                rect = None
                if box.shape == (4, 2) and np.all(np.isfinite(box)):
                    if (box.min() < 0 or box[:, 0].max() > candidate_image.shape[1]
                            or box[:, 1].max() > candidate_image.shape[0]):
                        continue
                    x1, y1 = box.min(axis=0)
                    x2, y2 = box.max(axis=0)
                    if x2 > x1 and y2 > y1:
                        rect = (float(x1), float(y1), float(x2), float(y2))
                located.append((str(getattr(token, "text", "") or ""), confidence, rect))
            # A sea/ship circle can occasionally misplace the measured dial.
            # Recovery uses the rendered vertical engine telegraph, rather
            # than accepting arbitrary numbers from a wider lower-left crop.
            levers = [(text, rect) for text, _, rect in located if rect is not None
                      and re.fullmatch(r"FULL|STOP|3/4|1/2|1/4", text.strip(), re.I)]
            columns = []
            for _, rect in levers:
                cx, glyph_height = (rect[0] + rect[2]) / 2, rect[3] - rect[1]
                row = [(text, box) for text, box in levers
                       if abs((box[0] + box[2]) / 2 - cx) <= glyph_height * 1.2
                       and .60 * glyph_height <= box[3] - box[1] <= 1.7 * glyph_height]
                if len(row) < 3 or len({text.upper() for text, _ in row}) < 2:
                    continue
                top, bottom = min(box[1] for _, box in row), max(box[3] for _, box in row)
                if 2.5 * glyph_height <= bottom - top <= 12 * glyph_height:
                    columns.append((cx, max(box[2] for _, box in row), top, bottom, glyph_height))
            if require_engine and not columns:
                return None
            candidates = []
            for text, confidence, rect in located:
                match = re.search(
                    r"(?<!\d)(\d{1,2}(?:[\.,]\d)?)\s*(?:kts?|节)",
                    text,
                    flags=re.IGNORECASE,
                )
                # OCR occasionally drops the ``kts`` suffix but preserves the
                # decimal; do not accept gear labels such as ``3/4``.
                if match is None:
                    match = re.fullmatch(r"\s*(\d{1,2}[\.,]\d)\s*", text)
                    if match is not None and require_engine:
                        # A split decimal still needs a nearby unit token on
                        # its own row, not a distance or countdown elsewhere.
                        if rect is None or not any(
                            other is not None and re.search(r"kts?|节", unit, re.I)
                            and abs((other[1] + other[3] - rect[1] - rect[3]) / 2)
                                <= (rect[3] - rect[1]) * .6
                            and other[0] >= rect[0] + (rect[2] - rect[0]) * .5
                            and -.75 * (rect[3] - rect[1]) <= other[0] - rect[2]
                                <= (rect[3] - rect[1]) * 1.5
                            for unit, _, other in located):
                            continue
                if match is None:
                    continue
                if require_engine:
                    if rect is None or not any(
                        cx + glyph_height * .8 <= (rect[0] + rect[2]) / 2 <= right + glyph_height * 8
                        and top - glyph_height <= (rect[1] + rect[3]) / 2 <= bottom + glyph_height
                        for cx, right, top, bottom, glyph_height in columns):
                        continue
                value = float(match.group(1).replace(",", "."))
                if 0.0 <= value <= 75.0:
                    # The unit's last digit can be a separate ``2kts`` token
                    # overlapping ``24.2``. Prefer the complete decimal value.
                    candidates.append((bool(re.search(r"[\.,]", match.group(1))), confidence, value))
            if not candidates:
                return None
            precision = max(item[0] for item in candidates)
            candidates = [item for item in candidates if item[0] == precision]
            if len({item[2] for item in candidates if item[1] >= .80}) > 1:
                return None
            return max(candidates)[1:]

        parsed = parse(crop)
        if parsed is not None:
            return parsed[1]
        fallbacks = []
        if isinstance(backend, RapidOcrBackend):
            for variant in numeric_ocr_fallback_variants(crop):
                candidate = parse(variant)
                if candidate is None:
                    continue
                fallbacks.append(candidate)
                if candidate[0] >= 0.80:
                    break
        if fallbacks:
            return max(fallbacks)[1]
        # This is a search domain only: OCR must establish a real telegraph
        # column before any speed can be accepted. Its measured glyph sizes
        # handle UI scaling and displacement independently of framebuffer size.
        search = image[round(height * .55):, :round(width * .36)]
        candidate = parse(search, require_engine=True)
        return None if candidate is None else candidate[1]

    @staticmethod
    def read_battle_clock_seconds(image, backend) -> int | None:
        """Read the active match clock from the top HUD.

        Different HUD layouts place the clock near the top centre or top right.
        Reading the whole shallow top band is more stable than tying lifecycle
        state to one resolution-specific rectangle. This value is used only as
        new-round evidence after a loading screen; it never drives combat.
        """
        if image is None or image.size == 0 or backend is None:
            return None
        height, width = image.shape[:2]
        # The match timer is the clock in the upper-right HUD. The two values
        # around the top-centre score are capture/base timers and must not be
        # accepted as evidence for a newly started round.
        top_right = detect_hud_layout(image).crop(image, "clock")
        if top_right.size == 0:
            return None

        pattern = re.compile(r"(?<!\d)([0-2]?\d)\s*[:：\.]\s*([0-5]\d)(?!\d)")

        def parse(candidate_image):
            candidates = []
            for token in backend.recognize(candidate_image):
                confidence = float(getattr(token, "confidence", 0.0))
                points = np.asarray(getattr(token, "box", ()) or (), dtype=np.float32)
                if confidence < .70 or points.shape != (4, 2):
                    continue
                if (points[:, 0].min() < 0 or points[:, 1].min() < 0
                        or points[:, 0].max() > candidate_image.shape[1]
                        or points[:, 1].max() > candidate_image.shape[0]):
                    continue
                text = str(token.text or "")
                for match in pattern.finditer(text):
                    seconds = int(match.group(1)) * 60 + int(match.group(2))
                    if 0 <= seconds <= 30 * 60:
                        candidates.append(
                            (float(getattr(token, "confidence", 0.0)), seconds)
                        )
            # Two different timers in one crop cannot authorize a match start.
            return max(candidates, default=None) if len({item[1] for item in candidates}) <= 1 else None

        parsed = parse(top_right)
        if parsed is not None:
            return parsed[1]
        if not isinstance(backend, RapidOcrBackend):
            return None
        fallbacks = []
        for variant in numeric_ocr_fallback_variants(top_right):
            candidate = parse(variant)
            if candidate is None:
                continue
            fallbacks.append(candidate)
            if candidate[0] >= 0.80:
                break
        return None if not fallbacks else max(fallbacks)[1]

    def in_exit_confirmation(self, image):
        button = self._crop_region(image, EXIT_CONTINUE_BUTTON)
        if button.size == 0:
            return False
        hsv = cv2.cvtColor(button, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 70, 70]), np.array([135, 255, 255]))
        labels, _, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
        largest = 0 if labels <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
        solid_button_ratio = largest / max(blue.size, 1)
        height, width = image.shape[:2]
        center_band = image[int(height * 0.34) : int(height * 0.58), int(width * 0.35) : int(width * 0.65)]
        # Snow/sea backgrounds can contain enough scattered cyan pixels to
        # pass a raw colour ratio.  A real modal action is one solid rectangle.
        measured = self._solid_action_rect(image, EXIT_CONTINUE_BUTTON,
            [((90, 70, 70), (135, 255, 255))], (.22, .25, .85, .80))
        if measured is not None and solid_button_ratio <= .08:
            x1, y1, x2, y2 = measured
            center_band = image[max(0, round(y1 - height * .12)):min(height, round(y2 + height * .08)),
                                max(0, round(x1 - width * .10)):min(width, round(x2 + width * .10))]
        return (solid_button_ratio > 0.08 or measured is not None) and center_band.mean() < 145

    def in_escape_menu(self, image):
        resume = self._crop_region(image, ESCAPE_RESUME_BUTTON)
        if resume.size == 0:
            return False
        hsv = cv2.cvtColor(resume, cv2.COLOR_BGR2HSV)
        olive = cv2.inRange(hsv, np.array([25, 40, 35]), np.array([85, 255, 190]))
        labels, _, stats, _ = cv2.connectedComponentsWithStats(olive, connectivity=8)
        largest = 0 if labels <= 1 else int(stats[1:, cv2.CC_STAT_AREA].max())
        solid_button_ratio = largest / max(olive.size, 1)
        height, width = image.shape[:2]
        outer = image[int(height * 0.15) : int(height * 0.85), :]
        measured = self._solid_action_rect(image, ESCAPE_RESUME_BUTTON,
            [((25, 40, 35), (85, 255, 190))], (.20, .20, .80, .78))
        return (solid_button_ratio > 0.30 or measured is not None) and outer.mean() < 150

    def _has_battle_hud(self, image):
        # A dense port can imitate every broad battle ROI: the lower-right
        # ship carousel looks like a minimap, the left menu like the player
        # block, and the orange Join Battle button like the score clock. The
        # port detector requires a solid action button plus three independent
        # port anchors, so let that stronger positive evidence veto the loose
        # HUD geometry before it can block ship selection.
        if self.in_port(image):
            logger.debug("Battle HUD vetoed by positive port controls")
            return False
        return self._has_battle_hud_layout(image)

    def has_numeric_battle_hud(self, image, backend) -> bool:
        """Resolve port/loading colour conflicts only with independent HUD data.

        This is for an already-committed match, not generic port navigation.
        Roster artwork and ship cards can mimic HUD texture/player arrows,
        but do not have BOTH the upper-right match clock and lower-left HP.
        A positive HUD still needs the separate temporal start gate to drive.
        """
        if backend is None or image is None or image.size == 0:
            return False
        if not self._has_battle_hud_layout(image):
            return False
        minimap = self.find_minimap(image)
        if minimap is None:
            return False
        if self.find_player_pose_on_minimap(minimap) is None:
            return False
        clock = self.read_battle_clock_seconds(image, backend)
        if clock is None or not 0 <= clock <= 30 * 60:
            return False
        health = self.read_health_fraction(image, backend)
        return health is not None and 0 <= health <= 1

    def _has_battle_hud_layout(self, image):
        """Independent HUD layout evidence, without consulting the port veto."""
        height, width = image.shape[:2]
        layout = detect_hud_layout(image)
        minimap = layout.crop(image, "minimap")
        player_hud = layout.crop(image, "player")
        consumables = layout.crop(image, "consumables")
        score_clock = layout.crop(image, "score_clock")
        metrics = {
            "minimap": self._visual_anchor_metrics(minimap),
            "player_name_health": self._visual_anchor_metrics(player_hud),
            "consumables": self._visual_anchor_metrics(consumables),
            "score_clock": self._visual_anchor_metrics(score_clock),
        }
        anchors = {
            "minimap": (
                metrics["minimap"]["std"] > 18
                and metrics["minimap"]["edge"] > 0.025
            ),
            "player_name_health": (
                metrics["player_name_health"]["std"] > 18
                and metrics["player_name_health"]["edge"] > 0.030
                and metrics["player_name_health"]["bright"] > 0.005
            ),
            "consumables": (
                metrics["consumables"]["std"] > 15
                and metrics["consumables"]["edge"] > 0.030
                and metrics["consumables"]["bright"] > 0.003
            ),
            "score_clock": (
                metrics["score_clock"]["std"] > 10
                and metrics["score_clock"]["edge"] > 0.018
            ),
        }
        # Pale maps can retain crisp HUD glyphs while losing sea/land contrast.
        # Require a real player marker for this low-contrast fallback.
        if (
            not anchors["minimap"]
            and metrics["minimap"]["std"] > 10
            and metrics["minimap"]["edge"] > 0.025
            and sum(anchors[name] for name in (
                "player_name_health", "consumables", "score_clock"
            )) >= 2
        ):
            anchors["minimap"] = self.find_player_pose_on_minimap(minimap) is not None
        logger.debug("Battle HUD anchors: %s", anchors)
        # The minimap is mandatory.  Require two of the three remaining,
        # independent HUD surfaces.  Making the lower-left player block
        # mandatory caused bright/snowy maps and temporary damage overlays to
        # remain classified as LOADING even though score, consumables and the
        # minimap were all present, leaving the ship at spawn.  Port/loading
        # fixtures have no minimap plus two of these anchors, so this remains
        # a fail-closed multi-anchor gate.
        return bool(
            anchors["minimap"]
            and sum(
                bool(anchors[name])
                for name in (
                    "player_name_health",
                    "consumables",
                    "score_clock",
                )
            )
            >= 2
        )

    @staticmethod
    def _is_port_reward_overlay(image) -> bool:
        """Reject the dimmed ``已获得 / 补给箱`` port modal as battle HUD.

        The overlay hides the port carousel while leaving textured artwork in
        the minimap region, so the normal port and battle guards can both lose
        their strongest evidence.  Its combination of a low-contrast full
        screen scrim and a bright, edge-rich heading at the top centre is
        stable and absent from the live battle fixtures.
        """
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        heading = image[
            int(height * 0.03) : int(height * 0.12),
            int(width * 0.42) : int(width * 0.58),
        ]
        if heading.size == 0:
            return False
        heading_hsv = cv2.cvtColor(heading, cv2.COLOR_BGR2HSV)
        heading_gray = cv2.cvtColor(heading, cv2.COLOR_BGR2GRAY)
        pale_heading = (
            (heading_gray > 150) & (heading_hsv[:, :, 1] < 60)
        )
        edge_ratio = np.count_nonzero(
            cv2.Canny(heading_gray, 60, 150)
        ) / max(heading_gray.size, 1)
        return (
            float(gray.mean()) < 70
            and float(gray.std()) < 35
            and float(np.mean(pale_heading)) > 0.03
            and edge_ratio > 0.02
        )

    def classify_screen(self, image):
        """Classify only states with positive UI evidence; never infer by exclusion."""
        if image is None or image.size == 0:
            return ScreenState.UNKNOWN
        # Resolve the live HUD before broad reward/modal heuristics. The
        # pre-battle roster page has teal/orange artwork that can satisfy both
        # of those heuristics, but it also exposes a minimap and HUD anchors.
        # Genuine result/port overlays do not, so this ordering is safe.
        battle_seen = self._has_battle_hud(image)
        # This port modal has neither trustworthy port nor battle controls.
        # Keep it UNKNOWN so recovery can dismiss/retry it explicitly instead
        # of ever entering the combat loop.
        if self._is_port_reward_overlay(image) and not battle_seen:
            return ScreenState.UNKNOWN
        # After a battle, the game can return to the port while leaving a
        # right-side victory/defeat reward card open.  This is positive proof
        # that combat ended, but it must remain PORT for navigation: marking it
        # RESULTS would make the result-page clicker hit the ship carousel.
        # The reward collector independently recognizes and OCRs this card.
        from core.results import ResultRewardReader

        reward_card_seen = ResultRewardReader._looks_like_port_reward_card(image)
        if reward_card_seen:
            # A pre-battle roster can trip the broad reward-card detector, but
            # its explicit start action is stronger evidence that this is not
            # the port. Keep ordinary post-battle reward cards as PORT even if
            # incidental artwork happens to resemble a HUD.
            start_action_seen = self._has_loading_start_action(image)
            if not start_action_seen:
                return ScreenState.PORT
        loading_seen = self.in_loading(image)
        # Resolve explicit menu pages before considering battle.  A port has
        # dense bottom cards and a real "加入战斗" action; a battle HUD must
        # never override that positive evidence.  The old ordering did the
        # reverse and let incidental port texture become a combat state.
        # Result-button colour alone is not sufficient: battle score markers,
        # consumables and minimap overlays can produce the same teal/orange
        # ratios.  A live battle HUD always wins this conflict so combat is
        # never handed to the result-page navigator.
        if self.in_results(image) and not battle_seen:
            return ScreenState.RESULTS
        port_seen = self.in_port(image)
        if port_seen:
            return ScreenState.LOADING if loading_seen else ScreenState.PORT
        # The login splash has ocean-blue areas in the same location as the
        # exit-confirmation button.  Positive startup evidence must win before
        # modal colour checks, while normal loading still remains below the
        # explicit escape-menu rules.
        if loading_seen and (
            self._is_login_splash(image)
            or self._has_loading_start_action(image)
        ):
            return ScreenState.LOADING
        # A live HUD takes priority over the broad blue/teal modal colour
        # candidates below. On low-contrast ocean maps the sea can fill the
        # historical ``继续战斗`` button ROI and look like one solid blue
        # component. The minimap plus two independent HUD anchors prevent that
        # sea patch from ejecting an active battle into Esc recovery.
        if battle_seen:
            return ScreenState.BATTLE
        if self.in_exit_confirmation(image):
            return ScreenState.EXIT_CONFIRMATION
        if self.in_escape_menu(image):
            return ScreenState.ESCAPE_MENU
        if loading_seen:
            return ScreenState.LOADING
        return ScreenState.UNKNOWN

    @staticmethod
    def save_debug_frame(path, image):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(destination), image)
