import numpy as np
import pytest

from core.ocr import (
    DistanceObservation,
    DistanceTrackFilter,
    OcrToken,
    RapidOcrBackend,
    Rect,
    TargetDistanceReader,
    ViewportTargetTracker,
)


class FakeBackend:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return self.tokens


def observation(value, when, target="target-1", confidence=0.95):
    return DistanceObservation(
        value_km=value,
        raw_text=f"{value:.1f} km",
        confidence=confidence,
        roi=Rect(0, 0, 100, 50),
        target_track_id=target,
        captured_at=when,
        accepted=True,
    )


def test_rapidocr_prefers_cuda_and_falls_back_only_when_unavailable():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert RapidOcrBackend.choose_provider(providers) == "CUDAExecutionProvider"
    assert (
        RapidOcrBackend.choose_provider(["CPUExecutionProvider"])
        == "CPUExecutionProvider"
    )
    assert (
        RapidOcrBackend.choose_provider(providers, prefer_gpu=False)
        == "CPUExecutionProvider"
    )


def test_dynamic_target_roi_scales_with_frame_and_stays_in_bounds():
    image = np.zeros((1600, 2560, 3), dtype=np.uint8)
    roi = TargetDistanceReader.target_roi(image, (1053, 626))

    assert roi.x == 925
    assert roi.y <= 626
    assert roi.width == 256
    assert roi.height >= 75


def test_decimal_distance_is_parsed_and_health_fraction_is_ignored():
    backend = FakeBackend(
        [
            OcrToken("598 / 60950", 0.99),
            OcrToken("23.7 公里", 0.95),
        ]
    )
    reader = TargetDistanceReader(backend=backend)
    image = np.zeros((1600, 2560, 3), dtype=np.uint8)

    result = reader.read(image, (1053, 626), "target-1", captured_at=1.0)

    assert result is not None
    assert result.accepted
    assert result.value_km == 23.7
    assert backend.calls == 1


def test_low_confidence_distance_is_returned_as_rejected_evidence():
    reader = TargetDistanceReader(
        backend=FakeBackend(
            [OcrToken("45 000 / 60 000", 0.99), OcrToken("9.4 km", 0.4)]
        ),
        minimum_confidence=0.8,
    )
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)

    result = reader.read(image, (800, 400), "target-1", captured_at=1.0)

    assert result is not None
    assert not result.accepted
    assert result.reject_reason == "ocr_confidence_below_threshold"


def test_aim_point_distance_without_enemy_health_anchor_is_rejected():
    reader = TargetDistanceReader(
        backend=FakeBackend([OcrToken("13.2 km", 0.99)]),
    )
    image = np.zeros((1000, 1600, 3), dtype=np.uint8)

    result = reader.read(image, (800, 500), "candidate", captured_at=1.0)

    assert result is not None
    assert not result.accepted
    assert result.reject_reason == "target_health_anchor_missing"


def test_distance_requires_two_consistent_samples_and_expires():
    track = DistanceTrackFilter(stable_samples=2, stale_seconds=1.2)

    assert track.update(1.0, "target-1", observation(9.2, 1.0)) is None
    stable = track.update(1.2, "target-1", observation(9.3, 1.2))

    assert stable is not None
    assert stable.value_km == pytest.approx(9.25)
    assert track.current(2.5) is None


def test_implausible_jump_and_target_switch_reset_history():
    track = DistanceTrackFilter(stable_samples=2)
    track.update(1.0, "target-1", observation(9.2, 1.0))
    assert track.update(1.2, "target-1", observation(18.0, 1.2)) is None
    assert track.update(1.4, "target-2", observation(7.0, 1.4, "target-2")) is None


def test_async_observation_freshness_starts_when_result_is_adopted():
    track = DistanceTrackFilter(stable_samples=2, stale_seconds=5.0)
    track.update(10.0, "target-1", observation(9.2, 1.0))
    stable = track.update(10.2, "target-1", observation(9.3, 1.2))

    assert stable is not None
    assert stable.observed_at == 10.2
    assert track.current(15.0) is not None


def test_viewport_tracker_preserves_id_for_nearby_label_and_replaces_far_one():
    tracker = ViewportTargetTracker(match_radius=80)
    first = tracker.update([(900, 500)], preferred_x=1000)
    second = tracker.update([(930, 510)], preferred_x=1000)
    switched = tracker.update([(1300, 700)], preferred_x=1000)

    assert first.track_id == second.track_id
    assert switched.track_id != second.track_id
