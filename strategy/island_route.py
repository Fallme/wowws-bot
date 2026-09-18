"""Persistent, clearance-aware routes through the confirmed island layer."""

import math

import cv2
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from core.terrain import rasterize


class IslandRoutePlanner:
    SIZE = 128
    # A geometric objective can fall on an island even though the nearby
    # navigable water is perfectly reachable.  The map centre is only a
    # coarse objective, so allow a bounded snap to that nearby water instead
    # of turning a valid route into a permanent navigation block.
    GOAL_SNAP_RATIO = 0.20

    def __init__(self):
        self.reset()

    def reset(self):
        self.path = []
        self.key = None
        self.goal = None

    @staticmethod
    def _clear(mask, start, end):
        line = np.zeros_like(mask)
        cv2.line(line, tuple(start), tuple(end), 1, 1)
        return not np.any(mask & line)

    def waypoint(
        self,
        shape,
        player,
        target,
        islands,
        *,
        clearance_ratio=.022,
        goal_snap_ratio=GOAL_SNAP_RATIO,
    ):
        if player is None or target is None:
            return None
        height, width = shape[:2]
        if min(height, width) <= 0:
            return None
        coords = np.array([player, target], dtype=float) / [width, height]
        if not np.all(np.isfinite(coords)) or np.any(coords < 0) or np.any(coords >= 1):
            return None
        if not islands:
            self.reset()
            return target
        n = self.SIZE
        land = rasterize(islands, size=n * 4)
        # Area pooling keeps narrow coastlines; nearest-neighbour downsampling
        # could erase a thin island and create a fictitious passage.
        land = (cv2.resize(land.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        radius = max(1, math.ceil(n * clearance_ratio))
        mask = cv2.dilate(land, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2))
        mask[[0, -1], :] = 1
        mask[:, [0, -1]] = 1
        start, goal = [tuple(np.clip(np.rint(p * (n - 1)), 0, n - 1).astype(int)) for p in coords]
        original_goal = goal
        if mask[goal[1], goal[0]]:
            # A cap's centre can be on land; stop at nearby navigable water,
            # never move an arbitrary unreachable target to the other map side.
            ys, xs = np.where(mask == 0)
            if not len(xs):
                return None
            distances = (xs - goal[0]) ** 2 + (ys - goal[1]) ** 2
            index = int(np.argmin(distances))
            snap_limit = max(radius + 2, int(round(n * float(goal_snap_ratio))))
            if distances[index] > snap_limit**2:
                return None
            goal = (int(xs[index]), int(ys[index]))
        key = mask.tobytes()
        # Only the safety margin around a waterborne starting position may be
        # relaxed. Actual land remains impassable, including next to the ship.
        local = np.zeros_like(mask)
        cv2.circle(local, start, radius, 1, -1)
        # The live white arrow can overlap a coastline by a few pixels. Treat
        # that tiny neighbourhood as the ship's known water, but keep every
        # other land cell blocked.
        land[local > 0] = 0
        mask[local > 0] = 0
        if self.goal is not None and math.dist(goal, self.goal) <= 2 and key == self.key:
            while len(self.path) > 1 and math.dist(start, self.path[0]) <= 2:
                self.path.pop(0)
            if self.path and self._clear(mask, start, self.path[0]):
                return (self.path[0][0] / (n - 1) * width, self.path[0][1] / (n - 1) * height)
        self.reset()
        if self._clear(mask, start, goal):
            return target if goal == original_goal else (goal[0] / (n - 1) * width, goal[1] / (n - 1) * height)
        clearance = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        weights = np.where(mask, 0, 1 + 2 / np.maximum(clearance, 1))
        grid = Grid(matrix=weights)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        path, _ = finder.find_path(grid.node(*start), grid.node(*goal), grid)
        if not path:
            return None
        points = [(node.x, node.y) for node in path]
        # Retain line-of-sight corners so a tiny position change does not make
        # successive frames choose opposite sides of the same island.
        anchor = start
        index = 1
        while index < len(points):
            farthest = index
            while farthest + 1 < len(points) and self._clear(mask, anchor, points[farthest + 1]):
                farthest += 1
            anchor = points[farthest]
            self.path.append(anchor)
            index = farthest + 1
        self.goal, self.key = goal, key
        if not self.path:
            return None
        point = self.path[0]
        return (point[0] / (n - 1) * width, point[1] / (n - 1) * height)
