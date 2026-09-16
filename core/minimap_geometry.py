"""Measure the displayed map grid instead of assuming a screen aspect ratio."""

import math
from dataclasses import replace

import cv2
import numpy as np


def _grid_axis(positions, extent, required=7):
    clusters = []
    for value in sorted(positions):
        if not clusters or value - np.mean(clusters[-1]) > 2.5:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    lines = np.array([np.mean(group) for group in clusters])
    best = None
    for end in lines[lines >= extent * .93]:
        for other in lines:
            for cells in range(1, 11):
                step = (end - other) / cells
                if not extent * .035 <= step <= extent * .098 or end - 10 * step < 0:
                    continue
                expected = end - np.arange(11) * step
                errors = np.min(abs(lines[:, None] - expected[None, :]), axis=0)
                matched = errors < max(1.7, step * .035)
                if matched.sum() < required or not np.any(matched[8:]):
                    continue
                score = (int(matched.sum()), -float(errors[matched].mean()))
                if best is None or score > best[0]:
                    best = (score, end - 10 * step, end)
    return None if best is None else best[1:]


def minimap_grid_bounds(image):
    """Return verified ten-cell map bounds, or None when evidence is weak."""
    height, width = image.shape[:2]
    side = min(height, round(width * .32))
    left, top = width - side, height - side
    roi = image[top:, left:]
    ratio = min(1.0, 900 / side)
    small = cv2.resize(roi, None, fx=ratio, fy=ratio)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 12, 36)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=100,
                            minLineLength=small.shape[0] * .52, maxLineGap=15)
    if lines is None:
        return None
    vertical, horizontal = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(x2 - x1) <= 1:
            vertical.append((x1 + x2) / 2)
        if abs(y2 - y1) <= 1:
            horizontal.append((y1 + y2) / 2)
    x = _grid_axis(vertical, small.shape[1])
    y = _grid_axis(horizontal, small.shape[0], required=6)
    if x is None or y is None or abs((x[1] - x[0]) / (y[1] - y[0]) - 1) > .025:
        return None
    return (round(left + x[0] / ratio), round(top + y[0] / ratio),
            round(left + x[1] / ratio), round(top + y[1] / ratio))


def confirm_capture_zones(samples, zones, shape, required=3):
    """Confirm one complete, consecutive point layer with small pixel jitter."""
    height, width = shape[:2]
    scale = min(height, width)
    normalized = [replace(z, center=(z.center[0] / width, z.center[1] / height),
                          radius=z.radius / scale) for z in zones]
    samples.append(normalized)
    required = max(2, int(required))
    recent = list(samples)[-required:]
    if len(recent) < required or not normalized or any(
        len(frame) != len(normalized) for frame in recent
    ):
        return None
    confirmed = []
    for candidate in normalized:
        matches = []
        for frame in recent:
            eligible = [z for z in frame
                        if math.dist(z.center, candidate.center) <= .015
                        and abs(z.radius - candidate.radius) <= .012
                        and (not z.label or not candidate.label or z.label == candidate.label)]
            if eligible:
                matches.append(min(eligible, key=lambda z: math.dist(z.center, candidate.center)))
        if len(matches) != required:
            continue
        center = np.median([z.center for z in matches], axis=0)
        radius = float(np.median([z.radius for z in matches]))
        labels = [z.label for z in matches if z.label]
        label = max(set(labels), key=labels.count) if labels else ""
        if label and labels.count(label) < required:
            label = ""
        confirmed.append(replace(candidate, center=(round(center[0] * width), round(center[1] * height)),
                                 radius=radius * scale, label=label))
    return confirmed or None
