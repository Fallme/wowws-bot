"""Battle-result reward OCR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from core.ocr import OcrBackend, OcrToken, RapidOcrBackend
from core.ui import RelativeRegion


RESULT_REWARD_REGIONS = {
    "credits": RelativeRegion(0.17, 0.35, 0.275, 0.41),
    "ship_xp": RelativeRegion(0.275, 0.35, 0.343, 0.41),
    "free_xp": RelativeRegion(0.345, 0.35, 0.415, 0.41),
}

# The result panel anchors its rewards farther left and lower on ultrawide
# windows. Keep a separate calibrated profile instead of stretching the
# 16:10 coordinates, which would crop away the leading digits.
WIDE_RESULT_REWARD_REGIONS = {
    "credits": RelativeRegion(0.075, 0.375, 0.225, 0.47),
    "ship_xp": RelativeRegion(0.225, 0.375, 0.345, 0.47),
    "free_xp": RelativeRegion(0.345, 0.375, 0.455, 0.47),
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

    def __init__(self, backend: OcrBackend | None = None, *, minimum_confidence=0.65):
        self.backend = backend or RapidOcrBackend()
        self.minimum_confidence = max(0.0, min(float(minimum_confidence), 1.0))

    @staticmethod
    def _token_x(token: OcrToken) -> float:
        if not token.box:
            return 0.0
        return min(point[0] for point in token.box)

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
        accepted = [
            token for token in tokens if token.confidence >= self.minimum_confidence
        ]
        pieces = [re.sub(r"\D", "", token.text) for token in accepted]
        pieces = [piece for piece in pieces if piece]
        if grouped_thousands and pieces:
            # Result values are rendered with spaces between thousands groups
            # (for example ``1 602``). The resource icon beside the number can
            # be misread as an extra one-digit token, so only complete
            # three-digit groups after the leading group are joined.
            groups = []
            for piece in pieces[1:]:
                if len(piece) < 3:
                    break
                groups.append(piece[:3])
            digits = pieces[0] + "".join(groups)
        else:
            # XP fields are single values.  Do not concatenate a stray digit
            # from the neighbouring reward column.
            digits = pieces[0] if pieces else ""
        value = int(digits) if digits else 0
        if value < 0 or value > maximum:
            value = 0
        confidence = (
            min(token.confidence for token in accepted) if digits and accepted else 0.0
        )
        return value, confidence, " | ".join(token.text for token in tokens)

    def read(self, image) -> BattleRewards:
        values = {}
        confidence = {}
        raw_text = {}
        height, width = image.shape[:2]
        regions = (
            WIDE_RESULT_REWARD_REGIONS
            if width / max(height, 1) >= 1.9
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
        recognized = values["credits"] > 0 and (
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
        )
