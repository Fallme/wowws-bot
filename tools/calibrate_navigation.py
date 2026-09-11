"""Offline calibration from steady straight/turn samples; never sends keys.

CSV columns: time_seconds,x,y,heading_degrees,speed_knots
x/y: normalized full-map coordinates, heading: clockwise from screen right.
Use a single steady phase per file, with samples at least 3 seconds apart.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def calibrate(rows, map_span_km):
    scales, radii = [], []
    for a, b in zip(rows, rows[1:]):
        dt = b['time_seconds'] - a['time_seconds']
        if dt < 3:
            continue
        chord = math.hypot(b['x'] - a['x'], b['y'] - a['y']) * map_span_km
        angle = abs(math.radians((b['heading_degrees'] - a['heading_degrees'] + 180) % 360 - 180))
        knots = (a['speed_knots'] + b['speed_knots']) / 2
        if knots < 5 or chord / dt > 0.20 or angle > math.pi / 2:
            continue
        arc = chord if angle < 1e-5 else chord * angle / (2 * math.sin(angle / 2))
        scales.append(arc / dt / (knots * 1.852 / 3600))
        if angle >= math.radians(8):
            radii.append(chord / (2 * math.sin(angle / 2)))
    if len(scales) < 3:
        raise ValueError('至少需要4个可靠样本，采样间隔至少3秒，且航速大于5节')
    return {'map_span_km': map_span_km, 'qe_speed_scale': round(median(scales), 4),
            'steady_turn_radius_km': round(median(radii), 4) if len(radii) >= 3 else None,
            'accepted_speed_segments': len(scales), 'accepted_turn_segments': len(radii)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--map_span_km', type=float, required=True)
    args = parser.parse_args()
    if not 10 <= args.map_span_km <= 100:
        parser.error('地图边长须在10至100km之间')
    with args.csv_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not all(math.isfinite(value) for row in rows for value in row.values()):
        parser.error('样本包含无效数值')
    print(json.dumps(calibrate(rows, args.map_span_km), ensure_ascii=False, indent=2))
