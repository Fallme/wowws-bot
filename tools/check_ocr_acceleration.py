"""Verify the OCR model's real ONNX Runtime execution provider."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ocr import RapidOcrBackend


def main() -> int:
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "live_battle.png"
    image = cv2.imread(str(fixture))
    if image is None:
        raise FileNotFoundError(fixture)
    # Use the same target-distance crop dimensions as production so provider
    # selection is verified by real inference rather than provider discovery.
    crop = image[620:701, 925:1181]
    backend = RapidOcrBackend(prefer_gpu=True)
    started = time.perf_counter()
    tokens = backend.recognize(crop)
    cold_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    warm_tokens = backend.recognize(crop)
    payload = {
        "execution_provider": backend.execution_provider,
        "fallback_reason": backend.fallback_reason,
        "cold_start_seconds": round(cold_seconds, 3),
        "warm_inference_seconds": round(time.perf_counter() - warm_started, 3),
        "token_count": len(warm_tokens),
        "tokens": [token.text for token in warm_tokens or tokens],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if backend.execution_provider == "CUDAExecutionProvider" else 2


if __name__ == "__main__":
    raise SystemExit(main())
