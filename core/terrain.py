"""Pixel consensus for normalized, north-up minimap terrain templates.

Templates are local to a battle. Never load another map's coastlines without
an independently verified map identity and viewport transform.
"""

import cv2
import numpy as np


SIZE = 512


def rasterize(outlines, size=SIZE):
    mask = np.zeros((size, size), np.uint8)
    for item in outlines or ():
        points = np.asarray(item.get("points", ()), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
            continue
        polygon = np.clip(np.rint(points * size), 0, size - 1).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def outlines_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    outlines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12:
            continue
        # Fixed pixel tolerance preserves small headlands and narrow channels.
        polygon = cv2.approxPolyDP(contour, 0.5, True)
        if len(polygon) >= 3:
            outlines.append({"points": [[float(x) / width, float(y) / height]
                                         for x, y in polygon[:, 0]],
                             "area": area / (width * height)})
    return sorted(outlines, key=lambda item: item["area"], reverse=True)


def confirm_terrain(samples, items, required=3):
    """Vote on occupied pixels, not the centroid/area of whole islands."""
    samples.append(list(items or ()))
    required = max(2, int(required))
    if len(samples) < required:
        return None
    votes = np.sum([rasterize(frame) for frame in samples], axis=0)
    return outlines_from_mask((votes >= required).astype(np.uint8)) or None


def extend_template(existing, confirmed):
    """Add independently confirmed land, rejecting an incompatible viewport.

    This checks alignment; it does not guess a transform. Large zoom/pan changes
    must be recaptured instead of silently warping navigation geometry.
    """
    if not existing:
        return confirmed
    old, new = rasterize(existing), rasterize(confirmed)
    smaller_area = min(int(old.sum()), int(new.sum()))
    if not smaller_area or np.count_nonzero(old & new) / smaller_area < 0.65:
        return existing
    return outlines_from_mask(old | new)


def repair_overlay_gaps(terrain, occluded, radius):
    """Close only short masked gaps; never close visible water channels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    closed = cv2.morphologyEx(terrain, cv2.MORPH_CLOSE, kernel)
    repaired = terrain.copy()
    repaired[occluded] = closed[occluded]
    return repaired
