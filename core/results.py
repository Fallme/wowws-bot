"""Battle-result reward OCR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

import cv2
import numpy as np

from core.ocr import (
    OcrBackend,
    OcrToken,
    RapidOcrBackend,
    numeric_ocr_fallback_variants,
)
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

# The maximized/borderless client captured by MSS is commonly 2560x1494
# (roughly 16:9 after the client-frame chrome is removed).  Its result values
# stay in the same columns as the 16:10 layout but the whole result block is
# lower.  Treating this frame as either the 16:10 or ultrawide profile crops
# through the numbers, which is why real runs read ``104`` instead of
# ``104 115`` and missed the green free-XP value completely.
BORDERLESS_RESULT_REWARD_REGIONS = {
    "credits": RelativeRegion(0.17, 0.36, 0.275, 0.44),
    "ship_xp": RelativeRegion(0.275, 0.36, 0.370, 0.44),
    # Start a little earlier than the old column boundary.  The star between
    # ship XP and free XP is wide, while two-digit free XP can otherwise be
    # clipped at its leading edge.
    "free_xp": RelativeRegion(0.350, 0.36, 0.450, 0.44),
}

# When the result page has already closed, the port keeps the last battle's
# rewards in a right-side card.  This is a separate, compact layout and is a
# useful fallback when the result transition was faster than one OCR cycle.
PORT_REWARD_REGIONS = {
    "credits": RelativeRegion(0.91, 0.525, 0.99, 0.552),
    "ship_xp": RelativeRegion(0.91, 0.548, 0.99, 0.575),
    "free_xp": RelativeRegion(0.91, 0.585, 0.99, 0.615),
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

    @staticmethod
    def _read_port_outcome(image) -> str:
        """Read the green/red headline on the compact port reward card."""
        if image is None or image.size == 0:
            return "unknown"
        height, width = image.shape[:2]
        crop = image[
            int(height * 0.39) : int(height * 0.44),
            int(width * 0.83) : int(width * 0.94),
        ]
        if crop.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        vivid = (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 80)
        green = vivid & (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95)
        red = vivid & ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
        green_ratio = float(np.mean(green))
        red_ratio = float(np.mean(red))
        if green_ratio >= 0.025 and green_ratio > red_ratio * 1.5:
            return "victory"
        if red_ratio >= 0.018 and red_ratio > green_ratio * 1.5:
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

    @staticmethod
    def _leading_digit_group(text: str) -> str:
        """Return one reward value without crossing a resource icon.

        The result row can be returned by OCR as one token such as
        ``217☆44☆``.  Spaces inside the leading value are thousands
        separators (``1 143``), while the star/icon is a hard field boundary.
        Removing every non-digit character used to turn that example into
        ``21744`` and incorrectly add free XP to ship XP.
        """
        match = re.search(r"\d", str(text or ""))
        if match is None:
            return ""
        leading = re.match(r"[\d\s]+", str(text)[match.start() :])
        if leading is None:
            return ""
        return re.sub(r"\D", "", leading.group(0))

    def _read_number_once(
        self,
        crop,
        maximum: int,
        *,
        grouped_thousands: bool = False,
    ):
        tokens = sorted(self.backend.recognize(crop), key=self._token_x)
        numeric_tokens = [
            (token, self._leading_digit_group(token.text))
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

    def _read_number(
        self,
        image,
        region: RelativeRegion,
        maximum: int,
        *,
        grouped_thousands: bool = False,
        minimum_expected: int = 1,
    ):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = region.pixels(width, height)
        crop = image[y1:y2, x1:x2]
        first = self._read_number_once(
            crop,
            maximum,
            grouped_thousands=grouped_thousands,
        )
        if first[0] >= minimum_expected or not isinstance(
            self.backend, RapidOcrBackend
        ):
            return first

        # Keep the fast and usually most accurate original-colour pass above.
        # Only a missing/clipped numeric field pays for enhanced retries.
        candidates = [first]
        for variant in numeric_ocr_fallback_variants(crop):
            candidate = self._read_number_once(
                variant,
                maximum,
                grouped_thousands=grouped_thousands,
            )
            candidates.append(candidate)
            if candidate[0] >= minimum_expected and candidate[1] >= 0.80:
                break
        valid = [candidate for candidate in candidates if candidate[0] >= minimum_expected]
        if not valid:
            return first
        return max(valid, key=lambda candidate: (candidate[1], len(str(candidate[0]))))

    def _read_regions(self, image, regions) -> BattleRewards:
        values = {}
        confidence = {}
        raw_text = {}
        for name, region in regions.items():
            value, score, raw = self._read_number(
                image,
                region,
                self.LIMITS[name],
                grouped_thousands=True,
                minimum_expected=(self.MINIMUM_CREDITS if name == "credits" else 1),
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

    @staticmethod
    def _looks_like_port_reward_card(image) -> bool:
        """Conservatively gate the extra three OCR calls for the port card."""
        if image is None or image.size == 0:
            return False
        height, width = image.shape[:2]
        if width / max(height, 1) < 1.65:
            return False
        title = image[
            int(height * 0.32) : int(height * 0.39),
            int(width * 0.83) : int(width * 0.93),
        ]
        if title.size == 0:
            return False
        title_hsv = cv2.cvtColor(title, cv2.COLOR_BGR2HSV)
        victory_green = (
            (title_hsv[:, :, 0] >= 35)
            & (title_hsv[:, :, 0] <= 95)
            & (title_hsv[:, :, 1] >= 70)
            & (title_hsv[:, :, 2] >= 80)
        )
        defeat_red = (
            ((title_hsv[:, :, 0] <= 15) | (title_hsv[:, :, 0] >= 165))
            & (title_hsv[:, :, 1] >= 80)
            & (title_hsv[:, :, 2] >= 90)
        )
        # The ordinary port's ship-stat panel can contain just as much green
        # as a reward card, but it has no green/red battle outcome heading in
        # this narrow row.  Requiring the heading prevents normal port frames
        # from being OCRed and counted as completed battles.
        if float(np.mean(victory_green | defeat_red)) < 0.01:
            return False
        panel = image[
            int(height * 0.38) : int(height * 0.63),
            int(width * 0.82) : int(width * 0.995),
        ]
        if panel.size == 0 or float(panel.std()) < 15.0:
            return False
        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
        vivid_green = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 70)
            & (hsv[:, :, 2] >= 80)
        )
        return float(np.mean(vivid_green)) >= 0.012

    def read(self, image) -> BattleRewards:
        height, width = image.shape[:2]
        # Layout follows the captured client aspect ratio.  A maximized 16:9
        # game often arrives as 2560x1494; it needs its own vertical profile.
        aspect_ratio = width / max(height, 1)
        if aspect_ratio >= 1.82:
            regions = WIDE_RESULT_REWARD_REGIONS
        elif aspect_ratio >= 1.65:
            regions = BORDERLESS_RESULT_REWARD_REGIONS
        else:
            regions = RESULT_REWARD_REGIONS
        rewards = self._read_regions(image, regions)
        if rewards.recognized or not self._looks_like_port_reward_card(image):
            return rewards

        port_rewards = self._read_regions(image, PORT_REWARD_REGIONS)
        if not port_rewards.recognized:
            return rewards
        return BattleRewards(
            **port_rewards.resource_values(),
            recognized=True,
            provider=port_rewards.provider,
            confidence=port_rewards.confidence,
            raw_text=port_rewards.raw_text,
            outcome=self._read_port_outcome(image),
        )
