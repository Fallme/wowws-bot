"""Run target-distance OCR against saved battle screenshots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ocr import TargetDistanceReader, ViewportTargetTracker
from core.vision import Vision

DEFAULT_REPORT_ROOT = Path(r"E:\aimemo\docs\screenshots\wowws_bot")


def evaluate_image(path: Path, vision: Vision, reader: TargetDistanceReader):
    started = time.perf_counter()
    image = cv2.imread(str(path))
    if image is None:
        return {"path": str(path), "error": "image_read_failed"}
    points = vision.find_enemies_in_viewport(image)
    tracker = ViewportTargetTracker()
    candidates = tracker.ordered_candidates(points, preferred_x=image.shape[1] / 2)
    attempts = []
    accepted = None
    for index, point in enumerate(candidates[:3]):
        observation = reader.read(image, point, f"candidate-{index}")
        record = {
            "point": list(point),
            "value_km": None if observation is None else observation.value_km,
            "confidence": 0.0 if observation is None else observation.confidence,
            "accepted": bool(observation is not None and observation.accepted),
            "raw_text": "" if observation is None else observation.raw_text,
            "reject_reason": None
            if observation is None
            else observation.reject_reason,
        }
        attempts.append(record)
        if record["accepted"]:
            accepted = record
            break
    return {
        "path": str(path),
        "enemy_candidates": len(points),
        "attempts": attempts,
        "accepted": accepted,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="离线评估敌舰标签距离 OCR")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_REPORT_ROOT / time.strftime("ocr_eval_%Y%m%d_%H%M%S.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    vision = Vision()
    reader = TargetDistanceReader()
    results = [evaluate_image(path, vision, reader) for path in args.images]
    payload = {
        "created_at": time.time(),
        "execution_provider": reader.execution_provider,
        "images": results,
        "accepted_images": sum(bool(item.get("accepted")) for item in results),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)
    return 0 if payload["accepted_images"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
