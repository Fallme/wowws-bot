"""Summarize a recorded battle event stream as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.replay import analyze_event_stream


def main() -> int:
    parser = argparse.ArgumentParser(description="离线复盘战斗事件并检查安全约束")
    parser.add_argument("events", type=Path, help="events.jsonl 路径")
    parser.add_argument("--too-close-km", type=float, default=4.5)
    parser.add_argument("--island-emergency-distance", type=float, default=0.055)
    args = parser.parse_args()
    report = analyze_event_stream(
        args.events,
        too_close_km=args.too_close_km,
        island_emergency_distance=args.island_emergency_distance,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if report.safety_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
