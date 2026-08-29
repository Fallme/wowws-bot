"""Target-label OCR and temporally stable distance observations."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import importlib.util
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Protocol
import unicodedata

import cv2
import numpy as np


logger = logging.getLogger("ocr")


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class DistanceObservation:
    value_km: float
    raw_text: str
    confidence: float
    roi: Rect
    target_track_id: str
    captured_at: float
    accepted: bool
    reject_reason: str | None = None
    source: str = "target_label_ocr"


@dataclass(frozen=True)
class StableDistance:
    value_km: float
    confidence: float
    target_track_id: str
    observed_at: float
    sample_count: int


class OcrBackend(Protocol):
    def recognize(self, image: np.ndarray) -> list[OcrToken]: ...


def numeric_ocr_fallback_variants(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build conservative high-resolution variants for small HUD numbers.

    The game renders HP, speed and result values with glow/outline effects.
    RapidOCR normally reads the original colour crop best, so callers always
    try that first.  These variants are only used after parsing the original
    crop failed; they enlarge the glyphs and suppress the background without
    changing the production OCR model or its CUDA-first execution policy.
    """
    if image is None or image.size == 0:
        return ()
    height, width = image.shape[:2]
    scale = 2.4 if min(height, width) < 180 else 1.8
    enlarged = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return (
        enlarged,
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
    )


class RapidOcrBackend:
    """Lazy offline RapidOCR backend with CUDA-first, CPU fallback execution."""

    def __init__(self, *, prefer_gpu: bool = True, device_id: int = 0):
        self._engine = None
        self.prefer_gpu = bool(prefer_gpu)
        self.device_id = max(0, int(device_id))
        self.execution_provider = "uninitialized"
        self.fallback_reason = ""
        self._dll_directory_handles = []

    @staticmethod
    def choose_provider(available_providers, *, prefer_gpu: bool = True) -> str:
        if prefer_gpu and "CUDAExecutionProvider" in available_providers:
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"

    @staticmethod
    def _engine_providers(engine) -> dict[str, list[str]]:
        providers = {}
        for name in ("text_det", "text_cls", "text_rec"):
            component = getattr(engine, name, None)
            wrapper = getattr(component, "session", None)
            session = getattr(wrapper, "session", None)
            if session is not None and hasattr(session, "get_providers"):
                providers[name] = list(session.get_providers())
        return providers

    def _params(self, model_root: Path, *, use_cuda: bool) -> dict:
        return {
            "Global.log_level": "error",
            "Global.use_cls": False,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
            "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": self.device_id,
            "Det.model_path": str(model_root / "ch_PP-OCRv4_det_infer.onnx"),
            "Cls.model_path": str(
                model_root / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
            ),
            "Rec.model_path": str(model_root / "ch_PP-OCRv4_rec_infer.onnx"),
            "Rec.rec_keys_path": str(model_root / "ppocr_keys_v1.txt"),
        }

    @staticmethod
    def _cuda_dll_directories() -> list[Path]:
        if os.name != "nt":
            return []
        candidates = []
        configured = os.environ.get("CUDA_PATH", "").strip()
        if configured:
            root = Path(configured)
            candidates.extend((root / "bin" / "x64", root / "bin"))
        toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if toolkit_root.exists():
            versions = sorted(toolkit_root.glob("v*"), reverse=True)
            for root in versions:
                candidates.extend((root / "bin" / "x64", root / "bin"))
        try:
            cudnn_spec = importlib.util.find_spec("nvidia.cudnn")
        except ModuleNotFoundError:
            cudnn_spec = None
        if cudnn_spec is not None and cudnn_spec.submodule_search_locations:
            candidates.extend(
                Path(location) / "bin"
                for location in cudnn_spec.submodule_search_locations
            )
        unique = []
        for candidate in candidates:
            if candidate.is_dir() and candidate not in unique:
                unique.append(candidate)
        return unique

    def _prepare_cuda_runtime(self, ort):
        directories = self._cuda_dll_directories()
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in directories:
                try:
                    self._dll_directory_handles.append(
                        os.add_dll_directory(str(directory))
                    )
                except OSError as error:
                    logger.debug("无法注册 CUDA DLL 目录 %s: %s", directory, error)
        if not hasattr(ort, "preload_dlls"):
            return
        cuda_directories = [
            directory
            for directory in directories
            if "nvidia gpu computing toolkit" in str(directory).lower()
        ]
        cuda_loaded = False
        for directory in cuda_directories:
            try:
                ort.preload_dlls(cuda=True, cudnn=False, directory=str(directory))
                cuda_loaded = True
                break
            except Exception as error:
                logger.debug("从 %s 预加载 CUDA 失败: %s", directory, error)
        # cuDNN is installed inside this project virtual environment. On a
        # machine without a system CUDA toolkit, load all NVIDIA wheels.
        ort.preload_dlls(cuda=not cuda_loaded, cudnn=True, directory="")

    def _build_engine(self, RapidOCR, model_root: Path, *, use_cuda: bool):
        engine = RapidOCR(params=self._params(model_root, use_cuda=use_cuda))
        component_providers = self._engine_providers(engine)
        expected = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        if component_providers and any(
            not providers or providers[0] != expected
            for providers in component_providers.values()
        ):
            raise RuntimeError(
                f"OCR session did not select {expected}: {component_providers}"
            )
        self.execution_provider = expected
        logger.info("OCR 推理后端: %s", expected)
        return engine

    def _load(self):
        if self._engine is not None:
            return self._engine
        import onnxruntime as ort
        import rapidocr
        from rapidocr import RapidOCR

        if hasattr(ort, "set_default_logger_severity"):
            # ORT 1.29 may emit a harmless legacy/plugin CUDA discovery
            # warning even when the resulting sessions use CUDA correctly.
            # Provider verification below is the authoritative check.
            ort.set_default_logger_severity(3)
        if self.prefer_gpu:
            try:
                self._prepare_cuda_runtime(ort)
            except Exception as error:
                logger.warning("CUDA 运行库预加载失败，将检查 CPU 回退: %s", error)
        model_root = Path(rapidocr.__file__).resolve().parent / "models"
        available = list(ort.get_available_providers())
        requested = self.choose_provider(available, prefer_gpu=self.prefer_gpu)
        if requested == "CUDAExecutionProvider":
            try:
                self._engine = self._build_engine(
                    RapidOCR,
                    model_root,
                    use_cuda=True,
                )
                return self._engine
            except Exception as error:
                self.fallback_reason = f"cuda_initialization_failed: {error}"
                logger.warning(
                    "OCR CUDA 初始化失败，回退 CPU: %s",
                    error,
                )
        else:
            self.fallback_reason = (
                "cuda_provider_unavailable: " + ",".join(available)
                if self.prefer_gpu
                else "gpu_disabled"
            )
            if self.prefer_gpu:
                logger.warning(
                    "OCR 未发现 CUDAExecutionProvider，回退 CPU；可用提供器: %s",
                    available,
                )
        self._engine = self._build_engine(RapidOCR, model_root, use_cuda=False)
        return self._engine

    def _fallback_to_cpu(self, error: Exception):
        import rapidocr
        from rapidocr import RapidOCR

        self.fallback_reason = f"cuda_inference_failed: {error}"
        logger.warning("OCR CUDA 推理失败，后续改用 CPU: %s", error)
        model_root = Path(rapidocr.__file__).resolve().parent / "models"
        self._engine = self._build_engine(RapidOCR, model_root, use_cuda=False)

    def recognize(self, image: np.ndarray) -> list[OcrToken]:
        if image is None or image.size == 0:
            return []
        engine = self._load()
        try:
            result = engine(image, use_cls=False, text_score=0.60)
        except Exception as error:
            if self.execution_provider != "CUDAExecutionProvider":
                raise
            self._fallback_to_cpu(error)
            result = self._engine(image, use_cls=False, text_score=0.60)
        raw_texts = getattr(result, "txts", None)
        raw_scores = getattr(result, "scores", None)
        raw_boxes = getattr(result, "boxes", None)
        texts = tuple(raw_texts) if raw_texts is not None else ()
        scores = tuple(raw_scores) if raw_scores is not None else ()
        boxes = tuple(raw_boxes) if raw_boxes is not None else ()
        tokens = []
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) else 0.0
            raw_box = boxes[index] if index < len(boxes) else ()
            box = tuple((float(point[0]), float(point[1])) for point in raw_box)
            tokens.append(OcrToken(str(text), score, box))
        return tokens


class TargetDistanceReader:
    """Read the decimal kilometre value below a tracked enemy label."""

    DISTANCE_PATTERN = re.compile(r"(?<!\d)(\d{1,2})[\.,](\d)(?!\d)")
    HEALTH_PATTERN = re.compile(r"\d[\d\s]{0,6}\s*/\s*\d[\d\s]{1,7}")

    def __init__(
        self,
        backend: OcrBackend | None = None,
        *,
        minimum_confidence: float = 0.78,
        scale: float = 3.0,
    ):
        self.backend = backend or RapidOcrBackend()
        self.minimum_confidence = max(0.0, min(float(minimum_confidence), 1.0))
        self.scale = max(1.0, float(scale))

    @property
    def execution_provider(self) -> str:
        return str(
            getattr(self.backend, "execution_provider", "custom") or "custom"
        )

    @staticmethod
    def target_roi(image, anchor: tuple[int, int]) -> Rect:
        height, width = image.shape[:2]
        anchor_x, anchor_y = anchor
        half_width = max(110, round(width * 0.050))
        top = anchor_y - max(4, round(height * 0.004))
        bottom = anchor_y + max(65, round(height * 0.047))
        x1 = max(0, anchor_x - half_width)
        x2 = min(width, anchor_x + half_width)
        y1 = max(0, top)
        y2 = min(height, bottom)
        return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    @staticmethod
    def _normalize(text: str) -> str:
        value = unicodedata.normalize("NFKC", text).strip().lower()
        value = value.replace("，", ",").replace("。", ".")
        value = re.sub(r"(?<=\d)[oO](?=\d|\s|$)", "0", value)
        return value

    def _observation_from_tokens(
        self,
        tokens: list[OcrToken],
        *,
        roi: Rect,
        target_track_id: str,
        captured_at: float,
    ) -> DistanceObservation | None:
        candidates = []
        normalized = [(token, self._normalize(token.text)) for token in tokens]
        health_seen = any(self.HEALTH_PATTERN.search(text) for _, text in normalized)
        for token, text in normalized:
            match = self.DISTANCE_PATTERN.search(text)
            if not match:
                continue
            value = float(f"{match.group(1)}.{match.group(2)}")
            if not 0.1 <= value <= 40.0:
                continue
            unit_seen = "km" in text or "公里" in text
            confidence = token.confidence if unit_seen else token.confidence * 0.88
            accepted = confidence >= self.minimum_confidence and health_seen
            if not health_seen:
                reject_reason = "target_health_anchor_missing"
            elif confidence < self.minimum_confidence:
                reject_reason = "ocr_confidence_below_threshold"
            else:
                reject_reason = None
            candidates.append(
                DistanceObservation(
                    value_km=value,
                    raw_text=token.text,
                    confidence=confidence,
                    roi=roi,
                    target_track_id=target_track_id,
                    captured_at=captured_at,
                    accepted=accepted,
                    reject_reason=reject_reason,
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    @staticmethod
    def _clahe(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def read(
        self,
        image,
        anchor: tuple[int, int],
        target_track_id: str,
        *,
        captured_at: float | None = None,
    ) -> DistanceObservation | None:
        captured_at = time.monotonic() if captured_at is None else float(captured_at)
        roi = self.target_roi(image, anchor)
        if roi.width <= 0 or roi.height <= 0:
            return None
        crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        resized = cv2.resize(
            crop,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC,
        )
        variants = (resized, self._clahe(resized))
        best = None
        for variant in variants:
            observation = self._observation_from_tokens(
                self.backend.recognize(variant),
                roi=roi,
                target_track_id=target_track_id,
                captured_at=captured_at,
            )
            if observation is not None and (
                best is None or observation.confidence > best.confidence
            ):
                best = observation
            if best is not None and best.accepted and best.confidence >= 0.90:
                break
            if (
                best is not None
                and best.reject_reason == "target_health_anchor_missing"
                and best.confidence >= 0.90
            ):
                break
        return best


class DistanceTrackFilter:
    """Require consistent OCR readings and expire stale target distances."""

    def __init__(
        self,
        *,
        stable_samples: int = 2,
        maximum_spread_km: float = 0.45,
        stale_seconds: float = 1.2,
        maximum_rate_km_s: float = 0.8,
    ):
        self.stable_samples = max(2, int(stable_samples))
        self.maximum_spread_km = max(0.05, float(maximum_spread_km))
        self.stale_seconds = max(0.2, float(stale_seconds))
        self.maximum_rate_km_s = max(0.1, float(maximum_rate_km_s))
        self.samples = deque(maxlen=max(self.stable_samples + 2, 5))
        self.target_track_id = None
        self.stable = None

    def reset(self):
        self.samples.clear()
        self.target_track_id = None
        self.stable = None

    def update(
        self,
        now: float,
        target_track_id: str | None,
        observation: DistanceObservation | None,
    ) -> StableDistance | None:
        now = float(now)
        if target_track_id != self.target_track_id:
            self.samples.clear()
            self.stable = None
            self.target_track_id = target_track_id
        if target_track_id is None:
            return None
        if observation is not None and observation.accepted:
            if observation.target_track_id != target_track_id:
                return self.current(now)
            if self.samples:
                previous = self.samples[-1]
                elapsed = max(observation.captured_at - previous.captured_at, 0.05)
                allowed_change = max(
                    self.maximum_spread_km,
                    self.maximum_rate_km_s * elapsed,
                )
                if abs(observation.value_km - previous.value_km) > allowed_change:
                    self.samples.clear()
            self.samples.append(observation)
            if len(self.samples) >= self.stable_samples:
                recent = list(self.samples)[-self.stable_samples :]
                values = sorted(item.value_km for item in recent)
                if values[-1] - values[0] <= self.maximum_spread_km:
                    self.stable = StableDistance(
                        value_km=float(np.median(values)),
                        confidence=min(item.confidence for item in recent),
                        target_track_id=target_track_id,
                        # OCR is asynchronous.  Freshness starts when the
                        # observation reaches the world model; captured_at is
                        # still retained for physical jump validation.
                        observed_at=now,
                        sample_count=len(recent),
                    )
        return self.current(now)

    def current(self, now: float) -> StableDistance | None:
        if self.stable is None:
            return None
        if float(now) - self.stable.observed_at > self.stale_seconds:
            self.stable = None
            return None
        return self.stable


@dataclass(frozen=True)
class TargetTrack:
    track_id: str
    point: tuple[int, int]


class ViewportTargetTracker:
    """Assign a stable identity to the selected viewport enemy label."""

    def __init__(self, match_radius: float = 140.0):
        self.match_radius = max(20.0, float(match_radius))
        self.active = None
        self.counter = 0

    def reset(self):
        self.active = None
        self.counter = 0

    def update(
        self,
        points: list[tuple[int, int]],
        *,
        preferred_x: float,
    ) -> TargetTrack | None:
        if not points:
            self.active = None
            return None
        if self.active is not None:
            nearest = min(points, key=lambda point: math.dist(point, self.active.point))
            if math.dist(nearest, self.active.point) <= self.match_radius:
                self.active = TargetTrack(self.active.track_id, nearest)
                return self.active
        selected = min(points, key=lambda point: abs(point[0] - preferred_x))
        self.counter += 1
        self.active = TargetTrack(f"viewport-{self.counter}", selected)
        return self.active

    def ordered_candidates(
        self,
        points: list[tuple[int, int]],
        *,
        preferred_x: float,
    ) -> list[tuple[int, int]]:
        remaining = list(points)
        ordered = []
        if self.active is not None and remaining:
            nearest = min(remaining, key=lambda point: math.dist(point, self.active.point))
            if math.dist(nearest, self.active.point) <= self.match_radius:
                ordered.append(nearest)
                remaining.remove(nearest)
        ordered.extend(sorted(remaining, key=lambda point: abs(point[0] - preferred_x)))
        return ordered

    def adopt(self, point: tuple[int, int]) -> TargetTrack:
        if self.active is not None and math.dist(point, self.active.point) <= self.match_radius:
            self.active = TargetTrack(self.active.track_id, point)
            return self.active
        self.counter += 1
        self.active = TargetTrack(f"viewport-{self.counter}", point)
        return self.active

    def match_active(self, points: list[tuple[int, int]]) -> TargetTrack | None:
        if self.active is None or not points:
            return None
        nearest = min(points, key=lambda point: math.dist(point, self.active.point))
        if math.dist(nearest, self.active.point) > self.match_radius:
            return None
        self.active = TargetTrack(self.active.track_id, nearest)
        return self.active


@dataclass(frozen=True)
class OcrBatchResult:
    point: tuple[int, int] | None
    observation: DistanceObservation | None
    evidence: DistanceObservation | None
    error: str | None = None


class DistanceOcrService:
    """Run expensive OCR on one latest frame without blocking battle control."""

    def __init__(self, reader: TargetDistanceReader):
        self.reader = reader
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="distance-ocr",
        )
        self.future: Future | None = None

    @property
    def pending(self) -> bool:
        return self.future is not None and not self.future.done()

    def submit(
        self,
        image,
        candidates: list[tuple[tuple[int, int], str]],
        *,
        captured_at: float,
    ) -> bool:
        if self.future is not None:
            if not self.future.done():
                return False
            # Results must be consumed explicitly so evidence is never lost.
            return False
        frame = image.copy()
        self.future = self.executor.submit(
            self._run,
            frame,
            list(candidates[:3]),
            float(captured_at),
        )
        return True

    def _run(self, image, candidates, captured_at):
        evidence = None
        try:
            for point, provisional_id in candidates:
                observation = self.reader.read(
                    image,
                    point,
                    provisional_id,
                    captured_at=captured_at,
                )
                if observation is not None and (
                    evidence is None or observation.confidence > evidence.confidence
                ):
                    evidence = observation
                if observation is not None and observation.accepted:
                    return OcrBatchResult(point, observation, evidence)
            return OcrBatchResult(None, None, evidence)
        except Exception as error:
            return OcrBatchResult(None, None, evidence, str(error))

    def poll(self) -> OcrBatchResult | None:
        if self.future is None or not self.future.done():
            return None
        future = self.future
        self.future = None
        return future.result()

    def close(self):
        # Close only happens between battles or during shutdown. Waiting here
        # prevents an old OCR worker from sharing its engine with a new run.
        self.executor.shutdown(wait=True, cancel_futures=True)
