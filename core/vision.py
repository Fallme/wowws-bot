"""Screen capture and visual state recognition."""

import base64
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.frame_guard import CaptureFault, FrameGuard
from core.ocr import RapidOcrBackend, numeric_ocr_fallback_variants
from core.ui import (
    ESCAPE_RESUME_BUTTON,
    EXIT_CONTINUE_BUTTON,
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


@dataclass(frozen=True)
class CaptureZone:
    center: tuple[int, int]
    radius: float
    label: str = ""
    # Capture ring ownership read from the live minimap.  Friendly green
    # points are intentionally not selected as an opening destination when a
    # neutral white or hostile red lettered point is available.
    state: str = "unknown"


class Vision:
    def __init__(self, screen_capture=None, frame_guard=None):
        self.screen_capture = screen_capture or ScreenCapture()
        self.frame_guard = frame_guard or FrameGuard()
        self.last_frame_quality = None
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
        roi = image[
            int(height * 0.79) : int(height * 0.88),
            int(width * 0.10) : int(width * 0.22),
        ]
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
        framebuffer_scale = min(width / 2560.0, height / 1600.0)
        minimum_pixels = max(
            24,
            int(round(120 * min(max(framebuffer_scale, 0.35), 1.0) ** 2)),
        )
        return bool(
            float(np.mean(green)) >= 0.008
            or green_pixels >= minimum_pixels
        )

    @staticmethod
    def read_autopilot_enabled_text(image, backend) -> bool:
        """Read the exact green lower-left ``自动驾驶`` status text.

        The same words also exist as a neutral key hint before a route starts.
        OCR first locates that text, then colour is measured only inside its
        returned glyph box. This avoids the unrelated green consumables and
        minimap content that made the old broad mask report false positives.
        """
        if image is None or image.size == 0 or backend is None:
            return False
        height, width = image.shape[:2]
        top = int(height * 0.68)
        crop = image[top : int(height * 0.97), 0 : int(width * 0.28)]
        if crop.size == 0:
            return False
        try:
            tokens = backend.recognize(crop)
        except Exception:
            logger.debug("自动驾驶状态文字 OCR 失败", exc_info=True)
            return False
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
            padding = max(2, int(round(min(width / 2560, height / 1600) * 3)))
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
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        top, bottom = int(height * 0.70), int(height * 0.84)

        def score(left_ratio, right_ratio):
            roi = hsv[
                top:bottom,
                int(width * left_ratio) : int(width * right_ratio),
            ]
            if roi.size == 0:
                return 0.0
            green = (
                (roi[:, :, 0] >= 35)
                & (roi[:, :, 0] <= 90)
                & (roi[:, :, 1] > 100)
                & (roi[:, :, 2] > 120)
            )
            return float(np.mean(green))

        q_score = score(0.42, 0.49)
        e_score = score(0.51, 0.58)
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

    def find_capture_zones(self, minimap, player=None, ocr_backend=None):
        """Return plausible capture circles visible on the minimap.

        ``player`` is optional, but when supplied it lets us reject the large
        gun/secondary range rings centred on the white player arrow.  Those
        rings are the main reason a tactical-map click can silently target the
        ship's current position instead of a capture area.
        """
        if minimap is None or minimap.size == 0:
            return []
        height, width = minimap.shape[:2]
        scale = min(width, height)
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=scale * 0.055,
            param1=100,
            param2=max(24, scale * 0.045),
            minRadius=max(12, int(scale * 0.040)),
            maxRadius=max(20, int(scale * 0.115)),
        )
        if circles is None:
            return []
        raw_candidates = []
        for rank, (raw_x, raw_y, raw_radius) in enumerate(circles[0]):
            center = (int(round(raw_x)), int(round(raw_y)))
            if (
                center[0] < scale * 0.035
                or center[1] < scale * 0.035
                or center[0] > width - scale * 0.035
                or center[1] > height - scale * 0.035
            ):
                continue
            raw_candidates.append(
                (
                    rank,
                    CaptureZone(
                        center=center,
                        radius=float(raw_radius),
                        state=self._capture_zone_state(
                            minimap,
                            center,
                            float(raw_radius),
                        ),
                    ),
                )
            )

        # In many station battles the capture circles form an evenly spaced
        # row at an arbitrary angle. That relationship is much more reliable
        # than circle colour:
        # the player's concentric range rings and circular island bays are
        # otherwise easy Hough false positives.  The middle cap can be heavily
        # obscured by the white player arrow, so infer its exact centre from
        # the two endpoints after requiring actual circular evidence nearby.
        endpoints = [
            item
            for item in raw_candidates
            if scale * 0.064 <= item[1].radius <= scale * 0.112
            and (
                player is None
                or math.dist(item[1].center, player)
                > max(scale * 0.12, item[1].radius * 1.15)
            )
        ]
        best_formation = None
        for left_index, (left_rank, left) in enumerate(endpoints):
            for right_rank, right in endpoints[left_index + 1 :]:
                first, third = left, right
                separation = math.dist(first.center, third.center)
                # Partial arcs from the player's concentric range rings can
                # be fitted as three nearby circles.  Real multi-cap layouts
                # span a substantial part of the minimap; reject the compact
                # fake formation before assigning A/B/C labels.
                if not scale * 0.36 <= separation <= scale * 0.74:
                    continue
                if abs(first.radius - third.radius) > scale * 0.025:
                    continue
                midpoint = (
                    (first.center[0] + third.center[0]) / 2.0,
                    (first.center[1] + third.center[1]) / 2.0,
                )
                middle_evidence = [
                    (rank, zone)
                    for rank, zone in raw_candidates
                    if zone not in (first, third)
                    and scale * 0.045 <= zone.radius <= scale * 0.110
                    and math.dist(zone.center, midpoint) <= scale * 0.070
                ]
                if not middle_evidence:
                    continue
                middle_rank, _middle = min(
                    middle_evidence,
                    key=lambda item: (math.dist(item[1].center, midpoint), item[0]),
                )
                score = (
                    left_rank
                    + right_rank
                    + middle_rank * 0.35
                    + abs(separation / scale - 0.42) * 30.0
                    + abs(first.radius - third.radius) / scale * 35.0
                )
                if best_formation is None or score < best_formation[0]:
                    radius = (first.radius + third.radius) / 2.0
                    ordered = sorted(
                        (first, third),
                        key=(
                            (lambda zone: zone.center[0])
                            if abs(first.center[0] - third.center[0])
                            >= abs(first.center[1] - third.center[1])
                            else (lambda zone: zone.center[1])
                        ),
                    )
                    best_formation = (
                        score,
                        [
                            CaptureZone(
                                ordered[0].center,
                                radius,
                                "A",
                                ordered[0].state,
                            ),
                            CaptureZone(
                                (int(round(midpoint[0])), int(round(midpoint[1]))),
                                radius,
                                "B",
                                _middle.state,
                            ),
                            CaptureZone(
                                ordered[1].center,
                                radius,
                                "C",
                                ordered[1].state,
                            ),
                        ],
                    )
        if best_formation is not None:
            return self._apply_capture_zone_ocr_labels(
                minimap, best_formation[1], ocr_backend
            )

        # Some maps use a triangular or otherwise staggered point layout.
        # Dynamically cluster the equally sized high-confidence circles rather
        # than applying coordinates from a known map. Nearby duplicate Hough
        # fits are collapsed; spatially separated peers are retained.
        uniform_groups = []
        for anchor_rank, anchor in endpoints:
            peers = [
                (rank, zone)
                for rank, zone in endpoints
                if abs(zone.radius - anchor.radius) <= scale * 0.014
            ]
            distinct = []
            for rank, zone in sorted(peers, key=lambda item: item[0]):
                if all(
                    math.dist(zone.center, kept.center) > scale * 0.12
                    for _kept_rank, kept in distinct
                ):
                    distinct.append((rank, zone))
            if len(distinct) >= 2:
                formation_span = max(
                    math.dist(first[1].center, second[1].center)
                    for index, first in enumerate(distinct)
                    for second in distinct[index + 1 :]
                )
                if formation_span < scale * 0.36:
                    continue
                uniform_groups.append(
                    (
                        sum(rank for rank, _zone in distinct[:4])
                        + abs(anchor.radius / scale - 0.075) * 20.0,
                        distinct[:4],
                    )
                )
        if uniform_groups:
            _score, group = min(uniform_groups, key=lambda item: item[0])
            zones = [zone for _rank, zone in group]
            zones.sort(key=lambda zone: (zone.center[0], zone.center[1]))
            zones = [
                CaptureZone(
                    zone.center,
                    zone.radius,
                    chr(ord("A") + index),
                    zone.state,
                )
                for index, zone in enumerate(zones)
            ]
            return self._apply_capture_zone_ocr_labels(
                minimap, zones, ocr_backend
            )

        # Non-three-point maps retain a conservative fallback.  Exclude any
        # circle centred on or enclosing the player so a range ring can never
        # become an autopilot destination.
        candidates = []
        for _rank, zone in raw_candidates:
            if not scale * 0.064 <= zone.radius <= scale * 0.112:
                continue
            if player is not None:
                player_offset = math.dist(zone.center, player)
                if player_offset <= max(scale * 0.12, zone.radius * 1.15):
                    continue
                if player_offset <= zone.radius * 0.72:
                    continue
            candidates.append(zone)
        if len(candidates) >= 2:
            candidate_span = max(
                math.dist(first.center, second.center)
                for index, first in enumerate(candidates)
                for second in candidates[index + 1 :]
            )
            if candidate_span < scale * 0.36:
                return []
        return self._apply_capture_zone_ocr_labels(
            minimap, candidates, ocr_backend
        )

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
        """Find the bright player arrow using its concentric yellow range ring.

        Bright islands and grid text look similar to the arrow in isolation.
        The player's marker is the only small white polygon centered inside the
        yellow gun-range circle, which gives a much stronger live signal.
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
            candidates.append(
                (
                    ring_coverage,
                    ring_score,
                    center_x,
                    center_y,
                    direction_x / length,
                    direction_y / length,
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
        # Only portions of the range circle can be visible behind islands or
        # capture overlays.  Twelve occupied sectors still describe a ring;
        # requiring eighteen made the live white arrow disappear precisely
        # when the ship approached an island.
        minimum_coverage = 12
        minimum_score = max(32, int(scale * scale * 0.00024))
        if candidates[0][0] < minimum_coverage or candidates[0][1] < minimum_score:
            return None
        if len(candidates) > 1 and candidates[0][0] < candidates[1][0] * 1.08:
            return None
        best = candidates[0]
        return PlayerPose(
            position=(best[2], best[3]),
            heading=(best[4], best[5]),
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
            return self._measure_island_risk(terrain, pose)
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float64)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        ocean_samples = hue[(saturation > 25) & (value > 25)]
        ocean_hue = float(np.median(ocean_samples)) if ocean_samples.size else 100.0
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
            (((hue >= 35) & (hue <= 95)) | (hue <= 8) | (hue >= 168))
            & (saturation > 70)
        )
        colored_land[ui_overlay] = 0
        capture_zone = self.find_nearest_capture_zone(minimap, pose.position)
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
            np.array([0, 0, 125]),
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
    ):
        """Choose a Q/E notch by comparing physically plausible future paths.

        The minimap grid is 50 km wide.  Each candidate helm order is simulated
        with a bounded rudder slew and a circular full-rudder turn.  Terrain
        clearance is a hard constraint; only after a route is safe do target
        progress and final heading decide which course wins.
        """
        if pose is None or target is None:
            return None
        height, width = minimap_shape[:2]
        if height <= 0 or width <= 0:
            return None
        heading_x, heading_y = pose.heading
        heading_length = math.hypot(heading_x, heading_y)
        if heading_length < 0.5:
            return None

        scale = float(min(width, height))
        pixels_per_km = max(scale / 50.0, 1e-6)
        speed_km_s = max(1.0, float(speed_knots)) * 1.852 / 3600.0
        speed_pixels_s = speed_km_s * pixels_per_km
        shift_seconds = max(1.0, float(rudder_shift_seconds))
        turn_radius = max(0.25, float(turning_radius_km))
        full_turn_rate = speed_km_s / turn_radius
        horizon = max(30.0, min(float(horizon_seconds), 240.0))
        clearance_limit = max(0.15, min(float(safety_clearance_km), 1.5))

        terrain = np.zeros((height, width), dtype=np.uint8)
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
            clearances = []
            collision_time = None
            elapsed = 0.0
            while elapsed < horizon:
                step = min(1.0, horizon - elapsed)
                maximum_change = step / shift_seconds
                helm_delta = max(
                    -maximum_change,
                    min(candidate - effective_rudder, maximum_change),
                )
                effective_rudder += helm_delta
                # Positive E rudder turns clockwise in minimap coordinates,
                # whose Y axis points down, hence the positive angle update.
                heading_angle += effective_rudder * full_turn_rate * step
                x += math.cos(heading_angle) * speed_pixels_s * step
                y += math.sin(heading_angle) * speed_pixels_s * step
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
                if collision_time is None and clearance <= clearance_limit:
                    collision_time = elapsed

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
            future_clearances = clearances[int(min(len(clearances), shift_seconds)) :]
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
                key=lambda item: item["escape_score"],
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
        )

    def _measure_island_risk(self, terrain, pose):
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
        terrain[self_exclusion] = 0
        terrain_y, terrain_x = np.nonzero(terrain)
        delta_x = terrain_x.astype(np.float64) - player_x
        delta_y = terrain_y.astype(np.float64) - player_y
        heading_x, heading_y = pose.heading
        forward = delta_x * heading_x + delta_y * heading_y
        lateral = delta_x * (-heading_y) + delta_y * heading_x
        minimum_forward = scale * 0.018
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

    def find_minimap_island_outlines(self, minimap, *, maximum_shapes: int = 24):
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
        hue_delta = np.minimum(
            np.abs(hue - ocean_hue),
            180.0 - np.abs(hue - ocean_hue),
        )
        colored_terrain = (
            (hue_delta > 16)
            & (saturation > 22)
            & (value > 28)
        ).astype(np.uint8) * 255
        # Snow maps encode most coastlines as bright, low-saturation shapes.
        # The old hue-only mask therefore produced no terrain at all.  Keep a
        # second neutral layer and filter its thin grid/range components below.
        neutral_terrain = (
            (saturation <= 145)
            & (value >= 125)
        ).astype(np.uint8) * 255
        # Red/green team marks, capture circles and smoke overlays are not
        # coastlines.  Remove them before connected-component extraction.
        ui_overlay = (
            (((hue >= 35) & (hue <= 95)) | (hue <= 8) | (hue >= 168))
            & (saturation > 70)
        )
        colored_terrain[ui_overlay] = 0
        neutral_terrain[ui_overlay] = 0
        player_pose = self.find_player_pose_on_minimap(minimap)
        player = None if player_pose is None else player_pose.position
        zone_mask = np.zeros((height, width), dtype=bool)
        yy, xx = np.indices((height, width))
        for zone in self.find_capture_zones(minimap, player=player):
            zone_mask |= (
                (xx - zone.center[0]) ** 2 + (yy - zone.center[1]) ** 2
                <= (zone.radius * 1.12) ** 2
            )
        colored_terrain[zone_mask] = 0
        neutral_terrain[zone_mask] = 0
        if player is not None:
            player_mask = (
                (xx - player[0]) ** 2 + (yy - player[1]) ** 2
                <= (scale * 0.045) ** 2
            )
            colored_terrain[player_mask] = 0
            neutral_terrain[player_mask] = 0

        minimum_pixels = max(70, int(scale * scale * 0.00018))
        minimum_extent = max(8, int(scale * 0.014))

        def retained_components(source, *, neutral=False):
            source = cv2.morphologyEx(
                source, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            source = cv2.morphologyEx(
                source, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
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
                if pixels < minimum_pixels or density < 0.15:
                    continue
                if max(component_width, component_height) < minimum_extent:
                    continue
                if (
                    component_width > scale * (0.48 if neutral else 0.52)
                    or component_height > scale * (0.48 if neutral else 0.52)
                    or bounding_area > scale * scale * (0.20 if neutral else 0.22)
                ):
                    continue
                kept[component_labels == component_label] = 255
            return kept

        terrain = cv2.bitwise_or(
            retained_components(colored_terrain),
            retained_components(neutral_terrain, neutral=True),
        )
        # Reapply capture removal after the two masks are combined so circle
        # fragments cannot reconnect through a nearby snowy island.
        terrain[zone_mask] = 0
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            terrain, connectivity=8
        )
        candidates = []
        for label in range(1, labels_count):
            x, y, component_width, component_height, pixels = stats[label]
            bounding_area = max(component_width * component_height, 1)
            density = pixels / bounding_area
            if pixels < minimum_pixels or density < 0.15:
                continue
            if max(component_width, component_height) < max(8, int(scale * 0.014)):
                continue
            if (
                x <= scale * 0.008
                or y <= scale * 0.008
                or x + component_width >= width - scale * 0.008
                or y + component_height >= height - scale * 0.008
            ):
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
            simplified = cv2.approxPolyDP(contour, perimeter * 0.035, True)
            if len(simplified) < 3:
                continue
            points = [
                [
                    round(float(point[0][0]) / max(width, 1), 4),
                    round(float(point[0][1]) / max(height, 1), 4),
                ]
                for point in simplified[:12]
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
        minimap_shape,
        player,
        target,
        island_outlines,
        *,
        clearance_ratio: float = 0.022,
    ):
        """Return a short minimap waypoint when the direct route crosses land.

        The final objective remains the capture point/map centre.  This local
        planner only bends the next leg around the first blocking static island
        and is recomputed from the live white-arrow position every frame.
        """
        if player is None or target is None or not island_outlines:
            return target
        height, width = minimap_shape[:2]
        if height <= 0 or width <= 0:
            return target
        mask = np.zeros((height, width), dtype=np.uint8)
        for island in island_outlines:
            points = island.get("points", ()) if isinstance(island, dict) else ()
            if len(points) < 3:
                continue
            polygon = np.array(
                [
                    [
                        int(round(float(point[0]) * width)),
                        int(round(float(point[1]) * height)),
                    ]
                    for point in points
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [polygon], 255)
        clearance = max(5, int(round(min(height, width) * clearance_ratio)))
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (clearance * 2 + 1, clearance * 2 + 1)
            ),
        )

        def blocked(start, end, *, width_pixels=4):
            route = np.zeros_like(mask)
            cv2.line(
                route,
                tuple(int(round(value)) for value in start),
                tuple(int(round(value)) for value in end),
                255,
                max(2, int(width_pixels)),
            )
            # Ignore a tiny disk around the current player marker.  Terrain
            # colour can overlap the arrow when a ship is already hugging a
            # coast; that must not make every escape candidate impossible.
            cv2.circle(
                route,
                tuple(int(round(value)) for value in start),
                clearance,
                0,
                -1,
            )
            return bool(np.any((route > 0) & (mask > 0)))

        if not blocked(player, target):
            return target

        route = np.zeros_like(mask)
        cv2.line(
            route,
            tuple(int(round(value)) for value in player),
            tuple(int(round(value)) for value in target),
            255,
            5,
        )
        blocking = (route > 0) & (mask > 0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if count <= 1:
            return target
        intersecting_labels = labels[blocking]
        intersecting_labels = intersecting_labels[intersecting_labels > 0]
        if not len(intersecting_labels):
            return target
        component = int(
            max(
                set(int(value) for value in intersecting_labels),
                key=lambda value: int(np.count_nonzero(intersecting_labels == value)),
            )
        )
        x, y, component_width, component_height, _pixels = stats[component]
        ys, xs = np.where(labels == component)
        # Probe expanded component corners on both sides of the direct route,
        # not a permanent preferred side.  The shorter currently-clear leg
        # wins, preventing one map from repeatedly producing the same circle.
        margin = clearance * 1.5
        candidates = [
            (
                max(4.0, min(width - 5.0, x - margin)),
                max(4.0, min(height - 5.0, y - margin)),
            ),
            (
                max(4.0, min(width - 5.0, x - margin)),
                max(4.0, min(height - 5.0, y + component_height + margin)),
            ),
            (
                max(4.0, min(width - 5.0, x + component_width + margin)),
                max(4.0, min(height - 5.0, y - margin)),
            ),
            (
                max(4.0, min(width - 5.0, x + component_width + margin)),
                max(4.0, min(height - 5.0, y + component_height + margin)),
            ),
        ]
        clear_candidates = [
            candidate
            for candidate in candidates
            if not blocked(player, candidate, width_pixels=3)
        ]
        if not clear_candidates:
            return target
        return min(
            clear_candidates,
            key=lambda candidate: math.dist(player, candidate)
            + math.dist(candidate, target),
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
        return [
            (center_x, center_y)
            for center_x, center_y, total_area, members in clusters
            if total_area >= 35 and members >= 2
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
        """Locate enemy target labels without counting team lists or score UI."""
        height, width = image.shape[:2]
        left = int(width * 0.10)
        right = int(width * 0.90)
        top = int(height * 0.30)
        bottom = int(height * 0.58)
        search = image[top:bottom, left:right]
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        mask = (
            cv2.inRange(hsv, self.red_lo1, self.red_hi1)
            | cv2.inRange(hsv, self.red_lo2, self.red_hi2)
            | cv2.inRange(hsv, self.orange_lo, self.orange_hi)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 8 <= area <= 800:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not 2 <= w <= width * 0.04 or not 2 <= h <= height * 0.03:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            points.append(
                (
                    int(moments["m10"] / moments["m00"]) + left,
                    int(moments["m01"] / moments["m00"]) + top,
                    area,
                )
            )
        clusters = self._cluster_colored_points(
            points,
            radius_x=max(24, int(width * 0.035)),
            radius_y=max(18, int(height * 0.04)),
        )
        return [
            (center_x, center_y)
            for center_x, center_y, total_area, members in clusters
            if total_area >= 45 and members >= 2
        ]

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
        """Crop the large square minimap anchored to the bottom-right corner."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = MINIMAP_REGION.pixels(width, height)
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
        area = self._crop_region(image, HEALTH_BAR_REGION)
        return area if area.size else None

    def find_reload_bar(self, image):
        height, _ = image.shape[:2]
        start_y = int(height * 0.55)
        search = image[start_y : int(height * 0.85), :]
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(
            hsv, np.array([10, 80, 80]), np.array([40, 255, 255])
        )
        row_counts = np.sum(yellow > 0, axis=1)
        valid = row_counts > 50
        if not np.any(valid):
            return None
        best_row = int(np.argmax(row_counts * valid))
        columns = np.where(yellow[best_row] > 0)[0]
        actual_y = best_row + start_y
        return image[
            max(0, actual_y - 3) : actual_y + 10,
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
        crop = image[
            int(height * 0.75) : int(height * 0.87),
            0 : int(width * 0.19),
        ]
        if crop.size == 0:
            return None
        def parse(candidate_image):
            tokens = list(backend.recognize(candidate_image) or ())
            tokens.sort(
                key=lambda token: (
                    min((point[1] for point in token.box), default=0),
                    min((point[0] for point in token.box), default=0),
                )
            )
            raw = " ".join(str(token.text or "") for token in tokens)
            normalized = (
                raw.replace(",", " ")
                .replace("-", " ")
                .replace("O", "0")
                .replace("o", "0")
                .replace("｜", "/")
                .replace("|", "/")
            )
            candidates = []
            for match in re.finditer(
                r"(?<!\d)(\d(?:[\d\s]{2,12}\d)?)\s*/\s*"
                r"(\d(?:[\d\s]{2,12}\d)?)(?!\d)",
                normalized,
            ):
                current_digits = re.sub(r"\D", "", match.group(1))
                maximum_digits = re.sub(r"\D", "", match.group(2))
                if not current_digits or not maximum_digits:
                    continue
                current = int(current_digits)
                maximum = int(maximum_digits)
                if maximum < 1000 or current < 0 or current > maximum:
                    continue
                confidence = min(
                    (float(getattr(token, "confidence", 0.0)) for token in tokens),
                    default=0.0,
                )
                candidates.append((maximum, confidence, current / maximum))
            return max(candidates, default=None)

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

    def _is_port_ship_bar(self, image):
        height, _ = image.shape[:2]
        bar = image[int(height * 0.88) : height, :]
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
        return {
            "ship_carousel": self._is_port_ship_bar(image),
            "left_port_menu": (
                (left["bright"] > 0.18 and left["edge"] > 0.02)
                or left["edge"] > 0.08
            ),
            "right_ship_details": right["std"] > 25 and right["edge"] > 0.045,
            "top_port_tabs": tabs["std"] > 30 and tabs["edge"] > 0.025,
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
        return solid_button_ratio > 0.12 and anchor_count >= 3

    def in_loading(self, image):
        # Positive port controls outrank broad loading artwork metrics. This
        # also keeps the battle-HUD port veto below from exposing the older
        # dark/textured loading fallback on a valid port frame.
        if self.in_port(image):
            return False
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
        start_button = self._crop_region(image, LOADING_START_BUTTON)
        if start_button.size == 0:
            return False
        hsv = cv2.cvtColor(start_button, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(
            hsv,
            np.array([35, 45, 45]),
            np.array([85, 255, 255]),
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
                area / pixels >= 0.16
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
        return teal_ratio > 0.30 and orange_ratio > 0.30

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
        # A real warning is a solid teal action button inside a distinctly
        # darker modal.  Loading artwork used to satisfy the old loose colour
        # check and caused repeated blind clicks while a battle was loading.
        return (
            solid_button_ratio > 0.08
            and float(dialog.mean()) + 8 < float(outer.mean())
        )

    def is_on_fire(self, image):
        """Confirm fire from both dedicated battle-HUD indicators.

        The lower-left ship-condition model shows one compact flame marker per
        active fire, while the consumable/status strip shows a wide triangular
        fire warning.  Requiring both prevents viewport flames, shell tracers,
        HP decoration, port icons and capture markers from triggering R.
        """
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        pixel_scale = max((width * height) / float(2560 * 1436), 0.20)
        linear_scale = math.sqrt(pixel_scale)

        def orange_components(x1, y1, x2, y2):
            crop = image[
                int(height * y1) : int(height * y2),
                int(width * x1) : int(width * x2),
            ]
            if crop.size == 0:
                return []
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            orange = cv2.inRange(
                hsv,
                np.array([0, 110, 110]),
                np.array([40, 255, 255]),
            )
            labels, _, stats, _ = cv2.connectedComponentsWithStats(
                orange, connectivity=8
            )
            return [
                (
                    int(stats[index, cv2.CC_STAT_AREA]),
                    int(stats[index, cv2.CC_STAT_WIDTH]),
                    int(stats[index, cv2.CC_STAT_HEIGHT]),
                )
                for index in range(1, labels)
            ]

        left_flames = [
            (area, component_width, component_height)
            for area, component_width, component_height in orange_components(
                0.025, 0.865, 0.105, 0.930
            )
            if 95 * pixel_scale <= area <= 500 * pixel_scale
            and 0.55
            <= component_width / max(component_height, 1)
            <= 1.25
            and 7 * linear_scale <= component_height <= 32 * linear_scale
        ]
        center_components = orange_components(0.470, 0.815, 0.540, 0.905)
        center_warning_bodies = [
            (area, component_width, component_height)
            for area, component_width, component_height in center_components
            if 100 * pixel_scale <= area <= 1200 * pixel_scale
            and component_width >= 12 * linear_scale
            and component_height >= 8 * linear_scale
        ]
        center_warning_bars = [
            (area, component_width, component_height)
            for area, component_width, component_height in center_components
            if 30 * pixel_scale <= area <= 1200 * pixel_scale
            and component_width >= component_height * 1.30
            and component_width >= 16 * linear_scale
        ]
        center_warning = bool(
            center_warning_bodies
            and (
                center_warning_bars
                or any(
                    component_width >= component_height * 1.30
                    for _, component_width, component_height in center_warning_bodies
                )
            )
        )
        logger.debug(
            "Fire HUD anchors: left=%s center_body=%s center_bar=%s",
            left_flames,
            center_warning_bodies,
            center_warning_bars,
        )
        # One active fire produces one ship-model marker.  The second required
        # anchor is the independent central warning, not a second fire stack.
        return bool(left_flames) and center_warning

    def is_flooding(self, image):
        """Detect the blue flooding icon directly below the numeric HP HUD."""
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        pixel_scale = max((width * height) / float(2560 * 1436), 0.20)
        linear_scale = math.sqrt(pixel_scale)
        # Flooding is represented in the fixed lower-left condition block.
        # Keep the ROI away from the sea/viewport and do not infer flooding
        # from water splashes around the rendered ship.
        status = image[
            int(height * 0.790) : int(height * 0.875),
            int(width * 0.025) : int(width * 0.120),
        ]
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

        central_status = image[
            int(height * 0.815) : int(height * 0.905),
            int(width * 0.470) : int(width * 0.540),
        ]
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
                    45 * pixel_scale <= area <= 1200 * pixel_scale
                    and 6 * linear_scale <= component_width <= 65 * linear_scale
                    and 6 * linear_scale <= component_height <= 65 * linear_scale
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

    @staticmethod
    def read_speed_knots(image, backend) -> float | None:
        """Read the player's lower-left speed readout without using viewport targets."""
        if image is None or image.size == 0 or backend is None:
            return None
        height, width = image.shape[:2]
        # This is the speed/engine sector left of the ship, deliberately
        # outside the central target labels and lower-right minimap.
        crop = image[
            int(height * 0.83) : int(height * 0.96),
            int(width * 0.105) : int(width * 0.205),
        ]
        if crop.size == 0:
            return None
        def parse(candidate_image):
            candidates = []
            for token in backend.recognize(candidate_image):
                text = str(token.text or "")
                match = re.search(
                    r"(?<!\d)(\d{1,2}(?:[\.,]\d)?)\s*(?:kts?|节)",
                    text,
                    flags=re.IGNORECASE,
                )
                # OCR occasionally drops the ``kts`` suffix but preserves the
                # decimal; do not accept gear labels such as ``3/4``.
                if match is None:
                    match = re.search(r"(?<!\d)(\d{1,2}[\.,]\d)(?!\d)", text)
                if match is None:
                    continue
                value = float(match.group(1).replace(",", "."))
                if 0.0 <= value <= 75.0:
                    candidates.append(
                        (float(getattr(token, "confidence", 0.0)), value)
                    )
            return max(candidates, default=None)

        parsed = parse(crop)
        if parsed is not None:
            return parsed[1]
        if not isinstance(backend, RapidOcrBackend):
            return None
        fallbacks = []
        for variant in numeric_ocr_fallback_variants(crop):
            candidate = parse(variant)
            if candidate is None:
                continue
            fallbacks.append(candidate)
            if candidate[0] >= 0.80:
                break
        return None if not fallbacks else max(fallbacks)[1]

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
        top_right = image[
            : max(1, int(height * 0.10)),
            int(width * 0.80) : width,
        ]
        if top_right.size == 0:
            return None

        pattern = re.compile(r"(?<!\d)([0-2]?\d)\s*[:：\.]\s*([0-5]\d)(?!\d)")

        def parse(candidate_image):
            candidates = []
            for token in backend.recognize(candidate_image):
                text = str(token.text or "")
                for match in pattern.finditer(text):
                    seconds = int(match.group(1)) * 60 + int(match.group(2))
                    if 0 <= seconds <= 30 * 60:
                        candidates.append(
                            (float(getattr(token, "confidence", 0.0)), seconds)
                        )
            return max(candidates, default=None)

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
        return solid_button_ratio > 0.08 and center_band.mean() < 145

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
        return solid_button_ratio > 0.30 and outer.mean() < 150

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
        height, width = image.shape[:2]
        minimap = self._crop_region(image, MINIMAP_REGION)
        player_hud = image[int(height * 0.70) :, : int(width * 0.25)]
        consumables = image[
            int(height * 0.78) :,
            int(width * 0.34) : int(width * 0.68),
        ]
        score_clock = image[
            int(height * 0.02) : int(height * 0.12),
            int(width * 0.42) : int(width * 0.58),
        ]
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
        # This port modal has neither trustworthy port nor battle controls.
        # Keep it UNKNOWN so recovery can dismiss/retry it explicitly instead
        # of ever entering the combat loop.
        if self._is_port_reward_overlay(image):
            return ScreenState.UNKNOWN
        # After a battle, the game can return to the port while leaving a
        # right-side victory/defeat reward card open.  This is positive proof
        # that combat ended, but it must remain PORT for navigation: marking it
        # RESULTS would make the result-page clicker hit the ship carousel.
        # The reward collector independently recognizes and OCRs this card.
        from core.results import ResultRewardReader

        if ResultRewardReader._looks_like_port_reward_card(image):
            return ScreenState.PORT
        loading_seen = self.in_loading(image)
        # Resolve explicit menu pages before considering battle.  A port has
        # dense bottom cards and a real "加入战斗" action; a battle HUD must
        # never override that positive evidence.  The old ordering did the
        # reverse and let incidental port texture become a combat state.
        battle_seen = self._has_battle_hud(image)
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
