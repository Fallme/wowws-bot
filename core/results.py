"""Battle-result reward OCR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

import cv2
import numpy as np

from core.ocr import OcrBackend, OcrToken, RapidOcrBackend
from core.ui import RelativeRegion


RESULT_REWARD_REGIONS = {
    "credits": RelativeRegion(0.17, 0.35, 0.275, 0.41),
    # Four-digit values such as ``1 143`` extend through the old 0.343 edge.
    # Slight overlap is intentional; grouped parsing stops at the one-digit
    # star/icon artifact and therefore does not absorb the next reward value.
    "ship_xp": RelativeRegion(0.275, 0.35, 0.370, 0.41),
    "free_xp": RelativeRegion(0.360, 0.35, 0.450, 0.41),
}

# The result panel anchors its rewards farther left and lower on ultrawide
# windows. Keep a separate calibrated profile instead of stretching the
# 16:10 coordinates, which would crop away the leading digits.
WIDE_RESULT_REWARD_REGIONS = {
    "credits": RelativeRegion(0.075, 0.375, 0.225, 0.47),
    "ship_xp": RelativeRegion(0.225, 0.375, 0.365, 0.47),
    "free_xp": RelativeRegion(0.355, 0.375, 0.475, 0.47),
}


@dataclass(frozen=True)
class BattleRewards:
    credits: int = 0
    ship_xp: int = 0
    free_xp: int = 0
    recognized: bool = False
    provider: str = "unknown"
    confidence: dict[str, float] = field(default_factory=dict)
    raw_text: dict[str, str] = field(default_factory=dict)
    outcome: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    def resource_values(self) -> dict[str, int]:
        return {
            "credits": self.credits,
            "ship_xp": self.ship_xp,
            "free_xp": self.free_xp,
        }


class ResultRewardReader:
    """Read the three numeric rewards shown on the personal result page."""

    LIMITS = {
        "credits": 100_000_000,
        "ship_xp": 10_000_000,
        "free_xp": 10_000_000,
    }
    # A one-digit/partial OCR fragment such as ``8`` is never a valid battle
    # silver reward. Reject it rather than permanently adding a wrong total.
    MINIMUM_CREDITS = 1_000

    @staticmethod
    def read_outcome(image) -> str:
        """Classify the large result headline without trusting reward OCR.

        Victory is rendered gold and defeat red in the upper-left result
        headline.  Looking only there avoids mixing it with team colours,
        ship badges or the three reward figures.  An uncertain frame remains
        ``unknown`` rather than writing a wrong outcome to history.
        """
        if image is None or image.size == 0:
            return "unknown"
        height, width = image.shape[:2]
        crop = image[
            int(height * 0.055) : int(height * 0.29),
            int(width * 0.035) : int(width * 0.31),
        ]
        if crop.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]
        vivid = (saturation >= 105) & (brightness >= 125)
        gold = vivid & (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 42)
        red = vivid & ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 172))
        gold_pixels = int(np.count_nonzero(gold))
        red_pixels = int(np.count_nonzero(red))
        minimum = max(35, int(crop.shape[0] * crop.shape[1] * 0.0012))
        if gold_pixels >= minimum and gold_pixels > red_pixels * 1.35:
            return "victory"
        if red_pixels >= minimum and red_pixels > gold_pixels * 1.35:
            return "defeat"
        return "unknown"

    def __init__(self, backend: OcrBackend | None = None, *, minimum_confidence=0.65):
        self.backend = backend or RapidOcrBackend()
        self.minimum_confidence = max(0.0, min(float(minimum_confidence), 1.0))

    @staticmethod
    def _token_x(token: OcrToken) -> float:
        if not token.box:
            return 0.0
        return min(point[0] for point in token.box)

    @staticmethod
    def _token_bounds(token: OcrToken):
        if not token.box or len(token.box) < 2:
            return None
        xs = [float(point[0]) for point in token.box]
        ys = [float(point[1]) for point in token.box]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    @classmethod
    def _overlap_ratio(cls, first: OcrToken, second: OcrToken) -> float:
        first_box = cls._token_bounds(first)
        second_box = cls._token_bounds(second)
        if first_box is None or second_box is None:
            return 0.0
        left = max(first_box[0], second_box[0])
        top = max(first_box[1], second_box[1])
        right = min(first_box[2], second_box[2])
        bottom = min(first_box[3], second_box[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
        second_area = (second_box[2] - second_box[0]) * (
            second_box[3] - second_box[1]
        )
        return intersection / max(min(first_area, second_area), 1.0)

    @classmethod
    def _deduplicate_numeric_tokens(cls, numeric_tokens):
        """Drop nested OCR hypotheses such as overlapping ``19``/``198``."""
        kept = []
        for token, piece in numeric_tokens:
            duplicate_index = next(
                (
                    index
                    for index, (other, _other_piece) in enumerate(kept)
                    if cls._overlap_ratio(token, other) >= 0.55
                ),
                None,
            )
            if duplicate_index is None:
                kept.append((token, piece))
                continue
            other, other_piece = kept[duplicate_index]
            candidate_score = (len(piece), token.confidence)
            other_score = (len(other_piece), other.confidence)
            if candidate_score > other_score:
                kept[duplicate_index] = (token, piece)
        return sorted(kept, key=lambda item: cls._token_x(item[0]))

    def _read_number(
        self,
        image,
        region: RelativeRegion,
        maximum: int,
        *,
        grouped_thousands: bool = False,
    ):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = region.pixels(width, height)
        crop = image[y1:y2, x1:x2]
        tokens = sorted(self.backend.recognize(crop), key=self._token_x)
        numeric_tokens = [
            (token, re.sub(r"\D", "", token.text))
            for token in tokens
        ]
        numeric_tokens = [item for item in numeric_tokens if item[1]]
        numeric_tokens = self._deduplicate_numeric_tokens(numeric_tokens)
        used_tokens = []
        pieces = []
        if grouped_thousands and numeric_tokens:
            # Result values are rendered with spaces between thousands groups
            # (for example ``1 602``). The resource icon beside the number can
            # be misread as an extra one-digit token, so only complete
            # three-digit groups after the leading group are joined.
            first_token, first_piece = numeric_tokens[0]
            if first_token.confidence >= self.minimum_confidence:
                pieces.append(first_piece)
                used_tokens.append(first_token)
            trailing_confidence = max(0.45, self.minimum_confidence - 0.22)
            for token, piece in numeric_tokens[1:]:
                if not pieces:
                    break
                if len(piece) < 3:
                    break
                if token.confidence < trailing_confidence:
                    break
                pieces.append(piece[:3])
                used_tokens.append(token)
            digits = "".join(pieces)
        else:
            # XP fields are single values.  Do not concatenate a stray digit
            # from the neighbouring reward column.
            accepted = [
                (token, piece)
                for token, piece in numeric_tokens
                if token.confidence >= self.minimum_confidence
            ]
            digits = accepted[0][1] if accepted else ""
            used_tokens = [accepted[0][0]] if accepted else []
        value = int(digits) if digits else 0
        if value < 0 or value > maximum:
            value = 0
        confidence = (
            min(token.confidence for token in used_tokens)
            if digits and used_tokens
            else 0.0
        )
        return value, confidence, " | ".join(token.text for token in tokens)

    def read(self, image) -> BattleRewards:
        values = {}
        confidence = {}
        raw_text = {}
        height, width = image.shape[:2]
        # Layout follows aspect ratio, not absolute resolution.  A 2560x1600
        # client is still 16:10 and uses the normal three-column placement;
        # treating all high-resolution frames as ultrawide crops credit/XP
        # groups into neighbouring columns (for example 258 / 088 / 897).
        expanded_layout = width / max(height, 1) >= 1.82
        regions = (
            WIDE_RESULT_REWARD_REGIONS
            if expanded_layout
            else RESULT_REWARD_REGIONS
        )
        for name, region in regions.items():
            value, score, raw = self._read_number(
                image,
                region,
                self.LIMITS[name],
                grouped_thousands=True,
            )
            values[name] = value
            confidence[name] = round(score, 4)
            raw_text[name] = raw
        recognized = values["credits"] >= self.MINIMUM_CREDITS and (
            values["ship_xp"] > 0 or values["free_xp"] > 0
        )
        return BattleRewards(
            **values,
            recognized=recognized,
            provider=str(
                getattr(self.backend, "execution_provider", "custom") or "custom"
            ),
            confidence=confidence,
            raw_text=raw_text,
            outcome=self.read_outcome(image),
        )
