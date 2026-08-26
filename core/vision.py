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
        return float(np.mean(green)) >= 0.008

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

    def find_capture_zones(self, minimap, player=None):
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
            if scale * 0.055 <= item[1].radius <= scale * 0.112
            and (
                player is None
                or math.dist(item[1].center, player) > scale * 0.050
            )
        ]
        best_formation = None
        for left_index, (left_rank, left) in enumerate(endpoints):
            for right_rank, right in endpoints[left_index + 1 :]:
                first, third = left, right
                separation = math.dist(first.center, third.center)
                if not scale * 0.28 <= separation <= scale * 0.70:
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
            return best_formation[1]

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
            return [
                CaptureZone(
                    zone.center,
                    zone.radius,
                    chr(ord("A") + index),
                    zone.state,
                )
                for index, zone in enumerate(zones)
            ]

        # Non-three-point maps retain a conservative fallback.  Exclude any
        # circle centred on or enclosing the player so a range ring can never
        # become an autopilot destination.
        candidates = []
        for _rank, zone in raw_candidates:
            if not scale * 0.055 <= zone.radius <= scale * 0.112:
                continue
            if player is not None:
                player_offset = math.dist(zone.center, player)
                if player_offset <= scale * 0.055:
                    continue
                if player_offset <= zone.radius * 0.72:
                    continue
            candidates.append(zone)
        return candidates

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
            # The live minimap arrow is a broad triangular pointer.  Its base
            # is the longest edge, so the opposite vertex is the bow heading.
            opposite_edges = []
            for index, point in enumerate(points):
                other = np.delete(points, index, axis=0)
                opposite_edges.append((np.linalg.norm(other[0] - other[1]), point))
            return max(opposite_edges, key=lambda candidate: candidate[0])[1]
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

    def find_island_risk(self, minimap, pose):
        """Measure solid terrain inside a forward navigation corridor.

        Thin range rings and grid lines are rejected by component density.
        The return distance is normalized by the minimap diagonal, matching
        target-distance configuration used by the movement controller.
        """
        if minimap is None or pose is None:
            return None
        height, width = minimap.shape[:2]
        scale = min(width, height)
        diagonal = max(math.hypot(width, height), 1.0)
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

        if not np.any(terrain):
            return None
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
        terrain = (
            (hue_delta > 16)
            & (saturation > 22)
            & (value > 28)
        ).astype(np.uint8) * 255
        # Red/green team marks, capture circles and smoke overlays are not
        # coastlines.  Remove them before connected-component extraction.
        ui_overlay = (
            (((hue >= 35) & (hue <= 95)) | (hue <= 8) | (hue >= 168))
            & (saturation > 70)
        )
        terrain[ui_overlay] = 0
        for zone in self.find_capture_zones(minimap):
            yy, xx = np.indices(terrain.shape)
            terrain[
                (xx - zone.center[0]) ** 2 + (yy - zone.center[1]) ** 2
                <= (zone.radius * 1.12) ** 2
            ] = 0
        terrain = cv2.morphologyEx(
            terrain, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        terrain = cv2.morphologyEx(
            terrain, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            terrain, connectivity=8
        )
        minimum_pixels = max(70, int(scale * scale * 0.00018))
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
            candidates.append((int(pixels), points))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            {"points": points}
            for _, points in candidates[: max(0, int(maximum_shapes))]
        ]

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
        return self._is_port_ship_bar(image) and solid_button_ratio > 0.12

    def in_loading(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 40 and gray.std() < 25:
            return True

        start_button = self._crop_region(image, LOADING_START_BUTTON)
        start_green = self._color_ratio(start_button, (35, 45, 45), (85, 255, 255))
        if start_green > 0.04:
            return True

        height, width = gray.shape[:2]
        center = gray[int(height * 0.38) : int(height * 0.58), int(width * 0.43) : int(width * 0.57)]
        bright_ratio = np.count_nonzero(center > 205) / max(center.size, 1)
        bottom_right = image[int(height * 0.60) :, int(width * 0.74) :]
        return 0.0005 < bright_ratio < 0.12 and bottom_right.std() > 12 and not self._has_battle_hud(image)

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
        """Detect the orange fire-duration indicator near the screen centre."""
        height, width = image.shape[:2]
        status = image[
            int(height * 0.47) : int(height * 0.55),
            int(width * 0.485) : int(width * 0.555),
        ]
        if status.size == 0:
            return False
        hsv = cv2.cvtColor(status, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(
            hsv, np.array([0, 150, 130]), np.array([25, 255, 255])
        )
        labels, _, stats, _ = cv2.connectedComponentsWithStats(
            orange, connectivity=8
        )
        if labels <= 1:
            return False
        areas = sorted(
            (int(value) for value in stats[1:, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        # Icon and timer glyphs are separate components. Requiring both avoids
        # treating a single shell tracer or aiming marker as a fire condition.
        return len(areas) >= 2 and areas[0] >= 35 and sum(areas[:4]) >= 75

    def is_flooding(self, image):
        """Conservatively detect the blue flooding status icon near the reticle."""
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        status = image[
            int(height * 0.47) : int(height * 0.55),
            int(width * 0.485) : int(width * 0.555),
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
        areas = [int(value) for value in stats[1:, cv2.CC_STAT_AREA]]
        # A blue ship/team marker is not sufficient: require a compact status
        # glyph of meaningful size inside the dedicated HUD area.
        return sum(area for area in areas if 18 <= area <= 520) >= 55

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
        tokens = backend.recognize(crop)
        candidates = []
        for token in tokens:
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
                candidates.append((float(getattr(token, "confidence", 0.0)), value))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def in_exit_confirmation(self, image):
        button = self._crop_region(image, EXIT_CONTINUE_BUTTON)
        blue_ratio = self._color_ratio(button, (90, 70, 70), (135, 255, 255))
        height, width = image.shape[:2]
        center_band = image[int(height * 0.34) : int(height * 0.58), int(width * 0.35) : int(width * 0.65)]
        return blue_ratio > 0.08 and center_band.mean() < 145

    def in_escape_menu(self, image):
        resume = self._crop_region(image, ESCAPE_RESUME_BUTTON)
        olive_ratio = self._color_ratio(resume, (25, 40, 35), (85, 255, 190))
        height, width = image.shape[:2]
        outer = image[int(height * 0.15) : int(height * 0.85), :]
        return olive_ratio > 0.08 and outer.mean() < 150

    def _has_battle_hud(self, image):
        height, width = image.shape[:2]
        minimap = self._crop_region(image, MINIMAP_REGION)
        if minimap.size == 0 or minimap.std() < 18:
            return False
        lower_left = image[int(height * 0.73) :, : int(width * 0.22)]
        gray = cv2.cvtColor(lower_left, cv2.COLOR_BGR2GRAY)
        bright_ratio = np.count_nonzero(gray > 165) / max(gray.size, 1)
        return bright_ratio > 0.004

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
        loading_seen = self.in_loading(image)
        # Resolve explicit menu pages before considering battle.  A port has
        # dense bottom cards and a real "加入战斗" action; a battle HUD must
        # never override that positive evidence.  The old ordering did the
        # reverse and let incidental port texture become a combat state.
        if self.in_results(image):
            return ScreenState.RESULTS
        port_seen = self.in_port(image)
        if port_seen:
            return ScreenState.LOADING if loading_seen else ScreenState.PORT
        if self.in_exit_confirmation(image):
            return ScreenState.EXIT_CONFIRMATION
        if self.in_escape_menu(image):
            return ScreenState.ESCAPE_MENU
        # Battle is deliberately the final actionable state.  ``_has_battle_hud``
        # is a visual candidate only; lifecycle callers separately require
        # consecutive battle frames before they issue combat controls.
        if self._has_battle_hud(image):
            return ScreenState.BATTLE
        if loading_seen:
            return ScreenState.LOADING
        return ScreenState.UNKNOWN

    @staticmethod
    def save_debug_frame(path, image):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(destination), image)
