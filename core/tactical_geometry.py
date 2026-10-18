"""Verified ten-cell tactical grids and their perspective coordinate mapping."""

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import heapq
import weakref

import cv2
import numpy as np


_GEOMETRY_CACHE = OrderedDict()
MAX_AXIS_MODELS = 64
MAX_AXIS_EVALUATIONS = 32768


@dataclass(frozen=True)
class TacticalMapGeometry:
    frame_shape: tuple[int, int]
    corners: tuple[tuple[float, float], ...]  # TL, TR, BR, BL
    size: int
    matched_rows: int = 11
    matched_columns: int = 11

    @property
    def bounds(self):
        points = np.asarray(self.corners)
        return (int(np.floor(points[:, 0].min())), int(np.floor(points[:, 1].min())),
                int(np.ceil(points[:, 0].max())), int(np.ceil(points[:, 1].max())))

    def normalized_to_pixel(self, point):
        source = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        transform = cv2.getPerspectiveTransform(source, np.float32(self.corners))
        coordinate = np.float32([[[float(point[0]), float(point[1])]]])
        x, y = cv2.perspectiveTransform(coordinate, transform)[0, 0]
        return int(round(float(x))), int(round(float(y)))

    def rectify(self, image):
        if tuple(image.shape[:2]) != self.frame_shape:
            raise ValueError("Tactical grid geometry belongs to a different frame size")
        edge = self.size - 1
        destination = np.float32([[0, 0], [edge, 0], [edge, edge], [0, edge]])
        transform = cv2.getPerspectiveTransform(np.float32(self.corners), destination)
        return cv2.warpPerspective(image, transform, (self.size, self.size))


def _cluster(models, index, tolerance=3.0):
    groups = []
    for model in sorted(models, key=lambda value: value[index]):
        if not groups or model[index] - np.median([value[index] for value in groups[-1]]) > tolerance:
            groups.append([model])
        else:
            groups[-1].append(model)
    clustered = []
    for group in groups:
        model = list(np.median(group, axis=0))
        if len(model) >= 5:
            model[3] = min(value[3] for value in group)
            model[4] = max(value[4] for value in group)
        clustered.append(tuple(model))
    return clustered


def _axes(models, minimum_span, maximum_span, limit=12, perspective=False):
    """Fit eleven positions with hard input, evaluation and storage budgets."""
    if len(models) < 8 or len(models) > MAX_AXIS_MODELS:
        return []
    positions = np.array([model[2] for model in models])
    rates = np.linspace(-.18, .18, 37) if perspective else np.array([0.0])
    pairs = [(left, right - left) for index, left in enumerate(positions)
             for right in positions[index + 1:]
             if minimum_span <= right - left <= maximum_span]
    if len(pairs) * len(rates) > MAX_AXIS_EVALUATIONS:
        return []
    fraction = np.linspace(0, 1, 11)[None, :]
    projected_fraction = fraction / (1 + rates[:, None] * (1 - fraction))
    candidates, sequence = [], 0
    keep = max(1, min(12, int(limit)))
    for left, span in pairs:
        expected = left + span * projected_fraction
        distance = abs(positions[:, None, None] - expected[None, :, :])
        nearest = distance.argmin(axis=0)
        errors = np.take_along_axis(distance, nearest[None, :, :], axis=0)[0]
        matched = errors <= max(2.1, span * .005)
        for rate_index in np.flatnonzero(matched.sum(axis=1) >= 8):
            accepted = matched[rate_index]
            selected = nearest[rate_index]
            if len(set(selected[accepted])) < 8:
                continue
            chosen = [models[index] for index in selected]
            score = (int(accepted.sum()), -float(errors[rate_index, accepted].mean()), span)
            entry = (score, -sequence, chosen, accepted.copy())
            sequence += 1
            if len(candidates) < keep:
                heapq.heappush(candidates, entry)
            elif entry[:2] > candidates[0][:2]:
                heapq.heapreplace(candidates, entry)
    return [(score, chosen, matched) for score, _, chosen, matched in
            sorted(candidates, key=lambda item: item[:2], reverse=True)]


def _intersection(horizontal, vertical):
    # y = horizontal[0] * x + horizontal[1]; x = vertical[0] * y + vertical[1]
    denominator = 1.0 - horizontal[0] * vertical[0]
    if abs(denominator) < .5:
        return None
    x = (vertical[0] * horizontal[1] + vertical[1]) / denominator
    return x, horizontal[0] * x + horizontal[1]


def _supported_row_groups(horizontal, vertical, height, scale):
    """Use a measured column family to discard unrelated short row edges.

    Snow textures and HUD text can exceed the row-search budget even when
    all eleven real columns are visible. A bounded column fit supplies their
    shared horizontal extent; it does not authorize a grid or a click. Every
    resulting row group still needs the normal two-axis/border verification.
    """
    columns = _cluster([
        (slope, intercept, slope * height / 2 + intercept, lower, upper)
        for slope, intercept, lower, upper in vertical
    ], 2)
    columns = [model for model in columns if sum(
        row[3] - 3 <= model[2] <= row[4] + 3 for row in horizontal
    ) >= 8]
    if not 8 <= len(columns) <= MAX_AXIS_MODELS:
        return []
    if columns[-1][2] - columns[0][2] < scale * .30:
        return []
    groups, seen = [], set()
    for _, selected, matches in _axes(columns, scale * .30, scale * 1.10):
        expected = np.linspace(selected[0][0], selected[-1][0], 11)
        coherent = matches & (abs(np.asarray([item[0] for item in selected]) - expected) <= .017)
        if coherent.sum() < 8:
            continue
        left, right = selected[0][2], selected[-1][2]
        rows = [model for model in horizontal
                if min(model[4], right) - max(model[3], left) >= (right - left) * .45]
        signature = tuple(rows)
        if 8 <= len(rows) <= MAX_AXIS_MODELS and signature not in seen:
            groups.append(rows)
            seen.add(signature)
    return groups


def _measure_tactical_map_geometry(image, *, minimum_line_ratio=.15):
    """Return a measured large 10x10 grid, never a guessed central crop.

    Both axes need at least eight repeated lines including the outer borders.
    A map may move and resize independently of the framebuffer. Its two sets
    of lines may form the mild trapezoid used by the game's M-map projection.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        return None
    height, width = image.shape[:2]
    if min(height, width) < 120:
        return None
    ratio = min(1.0, 1400.0 / max(height, width))
    sample = cv2.resize(image, None, fx=ratio, fy=ratio) if ratio < 1 else image
    small_height, small_width = sample.shape[:2]
    scale = min(small_height, small_width)
    try:
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 18, 54)
        raw = cv2.HoughLinesP(edges, 1, np.pi / 720,
                              threshold=max(25, round(scale * .035)),
                              minLineLength=max(30, round(scale * minimum_line_ratio)),
                              maxLineGap=max(4, round(scale * .025)))
        if raw is None:
            return None
        if len(raw) > 8192:
            return None
        horizontal, vertical = [], []
        for x1, y1, x2, y2 in raw.reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dx) >= scale * .15 and abs(dy) <= abs(dx) * .015:
                slope = dy / dx
                intercept = y1 - slope * x1
                horizontal.append((slope, intercept, slope * small_width / 2 + intercept,
                                   min(x1, x2), max(x1, x2)))
            if abs(dy) >= scale * .15 and abs(dx) <= abs(dy) * .20:
                slope = dx / dy
                intercept = x1 - slope * y1
                vertical.append((slope, intercept, min(y1, y2), max(y1, y2)))
        horizontal = _cluster(horizontal, 2)
        if len(horizontal) < 8 or len(vertical) < 8:
            return None
        row_groups = ([horizontal] if len(horizontal) <= MAX_AXIS_MODELS else
                      _supported_row_groups(horizontal, vertical, small_height, scale))
        row_axes = [candidate for group in row_groups
                    for candidate in _axes(group, scale * .30, scale * 1.10, perspective=True)]
        row_axes.sort(key=lambda item: item[0], reverse=True)
        best = None
        for row_score, rows, row_matches in row_axes[:12]:
            top, bottom = rows[0][2], rows[-1][2]
            grid_height = bottom - top
            middle = (top + bottom) / 2
            columns = _cluster([(slope, intercept, slope * middle + intercept, lower, upper)
                                for slope, intercept, lower, upper in vertical], 2)
            columns = [model for model in columns
                       if model[3] <= middle + grid_height * .20
                       and model[4] >= middle - grid_height * .20
                       and min(model[4], bottom) - max(model[3], top) >= grid_height * .15]
            if not 8 <= len(columns) <= MAX_AXIS_MODELS:
                continue
            for column_score, selected, column_matches in _axes(
                columns, grid_height * .70, grid_height * 1.25,
            ):
                corners = [_intersection(rows[0], selected[0]),
                           _intersection(rows[0], selected[-1]),
                           _intersection(rows[-1], selected[-1]),
                           _intersection(rows[-1], selected[0])]
                if any(point is None for point in corners):
                    continue
                quad = np.asarray(corners, np.float32)
                if (np.any(quad < -2) or np.any(quad[:, 0] > small_width + 2)
                        or np.any(quad[:, 1] > small_height + 2)
                        or not cv2.isContourConvex(quad)):
                    continue
                # A HUD line just beyond the real grid can continue its row
                # spacing, making a one-cell-shifted eleven-line fit larger
                # than the true map. Verify an actual outer vertical border
                # covers the top and ends at the chosen bottom border, rather
                # than merely extrapolating its slope into the HUD. A coast
                # can continue the same line above the map, so its earliest
                # segment is not necessarily the top border. The minimap
                # can cover the other side, so one complete outer side is
                # sufficient; internal columns cannot supply this evidence.
                border_tolerance = max(3.0, grid_height * .025)
                outer_sides = ((selected[0], quad[0][1], quad[3][1]),
                               (selected[-1], quad[1][1], quad[2][1]))
                if not any(model[3] <= edge_top + border_tolerance
                           and abs(model[4] - edge_bottom) <= border_tolerance
                           for model, edge_top, edge_bottom in outer_sides):
                    continue
                # A resident minimap can have the same eleven grid lines. It
                # occupies only the lower-right HUD; opening M reveals a grid
                # extending into the central viewport even with a smaller UI.
                if quad[:, 0].min() > small_width * .60 and quad[:, 1].min() > small_height * .48:
                    continue
                top_width, bottom_width = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
                map_height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
                if not (.70 <= top_width / map_height <= 1.25
                        and .70 <= bottom_width / map_height <= 1.25
                        and .70 <= top_width / bottom_width <= 1.30):
                    continue
                # The repeated lines must occupy this same quadrilateral, not
                # merely align with unrelated coast or HUD edges elsewhere.
                left, right = selected[0][2], selected[-1][2]
                supported_rows = [index for index, model in enumerate(rows)
                                  if row_matches[index]
                                  and min(model[4], right) - max(model[3], left)
                                  >= (right - left) * .45]
                if len(supported_rows) < 8:
                    continue
                expected_slopes = np.linspace(selected[0][0], selected[-1][0], 11)
                coherent_columns = column_matches & (abs(
                    np.asarray([model[0] for model in selected]) - expected_slopes
                ) <= .017)
                if coherent_columns.sum() < 8:
                    continue
                area = float(cv2.contourArea(quad))
                # Large tactical grids win over the simultaneously visible minimap.
                score = (min(len(supported_rows), int(coherent_columns.sum())), area,
                         row_score[1] + column_score[1])
                if best is None or score > best[0]:
                    original = quad / ratio
                    geometry = TacticalMapGeometry(
                        (height, width), tuple(tuple(float(value) for value in point) for point in original),
                        max(2, int(round(map_height / ratio))),
                        len(supported_rows), int(coherent_columns.sum()),
                    )
                    best = score, geometry
        if best is not None:
            return best[1]
        if (minimum_line_ratio == .15 and len(horizontal) > MAX_AXIS_MODELS
                and row_groups):
            # Only a measured repeated column family can justify this one
            # extra pass. Longer segments remove snow/HUD fragments; every
            # geometric and complete-border gate above remains mandatory.
            return _measure_tactical_map_geometry(image, minimum_line_ratio=.25)
        return None
    except (cv2.error, ValueError, ZeroDivisionError):
        return None


def detect_tactical_map_geometry(image):
    """Measure a frame once for scene recognition, rectification and input.

    Weak references retain no screenshots. A content digest also prevents a
    caller modifying an ndarray in place from reusing its former grid.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        return None
    identifier = id(image)
    pixels = memoryview(image) if image.flags.c_contiguous else image.tobytes()
    signature = (image.shape, image.dtype.str,
                 hashlib.blake2b(pixels, digest_size=8).digest())
    cached = _GEOMETRY_CACHE.get(identifier)
    if cached is not None and cached[0]() is image and cached[1] == signature:
        _GEOMETRY_CACHE.move_to_end(identifier)
        return cached[2]
    geometry = _measure_tactical_map_geometry(image)

    def discarded(reference):
        cached = _GEOMETRY_CACHE.get(identifier)
        if cached is not None and cached[0] is reference:
            _GEOMETRY_CACHE.pop(identifier, None)

    _GEOMETRY_CACHE[identifier] = (weakref.ref(image, discarded), signature, geometry)
    _GEOMETRY_CACHE.move_to_end(identifier)
    while len(_GEOMETRY_CACHE) > 32:
        _GEOMETRY_CACHE.popitem(last=False)
    return geometry
