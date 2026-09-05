import numpy as np
import cv2

from core.ocr import OcrToken
from core.results import (
    RESULT_REWARD_REGIONS,
    WIDE_RESULT_REWARD_REGIONS,
    ResultRewardReader,
)


class FakeRewardBackend:
    execution_provider = "CUDAExecutionProvider"

    def __init__(self):
        self.calls = 0

    def recognize(self, _image):
        values = (
            [
                OcrToken("258", 0.99, ((10, 0),)),
                OcrToken("088", 0.98, ((80, 0),)),
            ],
            [OcrToken("897☆", 0.91, ((10, 0),))],
            [OcrToken("540☆", 0.96, ((10, 0),))],
        )
        result = values[self.calls]
        self.calls += 1
        return result


def test_result_reward_reader_extracts_credits_and_experience():
    image = np.full((1600, 2560, 3), 80, dtype=np.uint8)
    rewards = ResultRewardReader(FakeRewardBackend()).read(image)

    assert rewards.recognized
    assert rewards.credits == 258088
    assert rewards.ship_xp == 897
    assert rewards.free_xp == 540
    assert rewards.provider == "CUDAExecutionProvider"


def test_16_by_10_result_fixture_uses_normal_three_column_layout():
    image = cv2.imread("tests/fixtures/results.png")

    rewards = ResultRewardReader().read(image)

    assert rewards.recognized
    assert rewards.credits == 258_088
    assert rewards.ship_xp == 897
    assert rewards.free_xp == 540


def test_result_reader_classifies_coloured_result_headline_without_reward_ocr():
    image = np.zeros((900, 1800, 3), dtype=np.uint8)
    # BGR gold matching the victory headline in the upper-left result area.
    image[100:165, 130:410] = (50, 190, 250)
    assert ResultRewardReader.read_outcome(image) == "victory"

    image[:] = 0
    image[100:165, 130:410] = (70, 75, 240)
    assert ResultRewardReader.read_outcome(image) == "defeat"


def test_result_reward_reader_rejects_missing_credit_value():
    class EmptyBackend:
        def recognize(self, _image):
            return []

    rewards = ResultRewardReader(EmptyBackend()).read(
        np.full((1600, 2560, 3), 80, dtype=np.uint8)
    )

    assert not rewards.recognized


def test_defeat_headline_ocr_overrides_gold_ship_text_and_background():
    from types import SimpleNamespace
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[100:200, 100:480] = (50, 190, 250)
    backend = SimpleNamespace(recognize=lambda _: [
        OcrToken("失败！", .99, ((150, 100), (290, 100), (290, 160), (150, 160))),
        OcrToken("胜利", .99, ((150, 220), (180, 220), (180, 232), (150, 232))),
    ])
    assert ResultRewardReader.read_outcome(image, backend) == "defeat"
    backend.recognize = lambda _: []
    assert ResultRewardReader.read_outcome(image, backend) == "unknown"


def test_result_reward_reader_rejects_clipped_one_digit_credit_fragment():
    class ClippedBackend:
        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [OcrToken("8", 0.99, ((10, 0),))],
                [OcrToken("503", 0.99, ((10, 0),))],
                [OcrToken("63", 0.99, ((10, 0),))],
            )
            value = values[self.calls]
            self.calls += 1
            return value

    rewards = ResultRewardReader(ClippedBackend()).read(
        np.full((1600, 2560, 3), 80, dtype=np.uint8)
    )

    assert rewards.credits == 8
    assert not rewards.recognized


def test_result_reward_reader_rejects_icon_and_neighbour_column_digits():
    class NoisyBackend:
        execution_provider = "CUDAExecutionProvider"

        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [
                    OcrToken("68", 0.99, ((10, 0),)),
                    OcrToken("9440", 0.86, ((80, 0),)),
                ],
                [
                    OcrToken("503☆", 0.93, ((10, 0),)),
                    OcrToken("6", 0.92, ((180, 0),)),
                ],
                [OcrToken("63☆", 0.94, ((10, 0),))],
            )
            result = values[self.calls]
            self.calls += 1
            return result

    rewards = ResultRewardReader(NoisyBackend()).read(
        np.full((1600, 2560, 3), 80, dtype=np.uint8)
    )

    assert rewards.recognized
    assert rewards.credits == 68_944
    assert rewards.ship_xp == 503
    assert rewards.free_xp == 63


def test_result_reward_reader_joins_grouped_xp_and_ignores_icon_digit():
    class GroupedBackend:
        execution_provider = "CUDAExecutionProvider"

        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [
                    OcrToken("198", 0.99, ((10, 0),)),
                    OcrToken("363", 0.98, ((80, 0),)),
                    OcrToken("0", 0.91, ((145, 0),)),
                ],
                [
                    OcrToken("1", 0.97, ((10, 0),)),
                    OcrToken("602", 0.96, ((45, 0),)),
                ],
                [OcrToken("198☆", 0.95, ((10, 0),))],
            )
            result = values[self.calls]
            self.calls += 1
            return result

    rewards = ResultRewardReader(GroupedBackend()).read(
        np.full((1600, 2560, 3), 80, dtype=np.uint8)
    )

    assert rewards.recognized
    assert rewards.credits == 198_363
    assert rewards.ship_xp == 1_602
    assert rewards.free_xp == 198


def test_wide_result_profile_moves_rewards_left_and_lower():
    standard = RESULT_REWARD_REGIONS["credits"]
    wide = WIDE_RESULT_REWARD_REGIONS["credits"]

    assert wide.left < standard.left
    assert wide.top > standard.top


def test_ship_xp_region_keeps_four_digit_group_and_accepts_soft_trailing_group():
    class FourDigitBackend:
        execution_provider = "CUDAExecutionProvider"

        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [OcrToken("102", 0.99, ((10, 0),)), OcrToken("692", 0.98, ((80, 0),))],
                [OcrToken("1", 0.98, ((10, 0),)), OcrToken("143☆", 0.52, ((48, 0),))],
                [OcrToken("136☆", 0.96, ((10, 0),))],
            )
            value = values[self.calls]
            self.calls += 1
            return value

    rewards = ResultRewardReader(FourDigitBackend()).read(
        np.full((1600, 2560, 3), 80, dtype=np.uint8)
    )

    assert rewards.recognized
    assert rewards.ship_xp == 1_143
    assert RESULT_REWARD_REGIONS["ship_xp"].right >= 0.37


def test_overlapping_partial_ocr_hypothesis_does_not_prefix_value():
    class OverlappingBackend:
        execution_provider = "CUDAExecutionProvider"

        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [OcrToken("198", 0.99, ((5, 2), (45, 2), (45, 22), (5, 22))),
                 OcrToken("363", 0.98, ((52, 2), (102, 2), (102, 22), (52, 22)))],
                [OcrToken("1", 0.97, ((5, 2), (15, 2), (15, 22), (5, 22))),
                 OcrToken("143", 0.96, ((22, 2), (62, 2), (62, 22), (22, 22)))],
                [OcrToken("19", 0.92, ((0, 2), (42, 2), (42, 22), (0, 22))),
                 OcrToken("198", 0.89, ((12, 2), (146, 2), (146, 22), (12, 22)))],
            )
            value = values[self.calls]
            self.calls += 1
            return value

    rewards = ResultRewardReader(OverlappingBackend()).read(
        np.full((870, 1827, 3), 80, dtype=np.uint8)
    )

    assert rewards.recognized
    assert rewards.credits == 198_363
    assert rewards.ship_xp == 1_143
    assert rewards.free_xp == 198


def test_star_icon_separates_ship_xp_from_green_free_xp_suffix():
    class StarSeparatedBackend:
        execution_provider = "CUDAExecutionProvider"

        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [OcrToken("69 276⚓", 0.99, ((10, 0),))],
                [OcrToken("217☆44☆", 0.98, ((10, 0),))],
                [OcrToken("44☆", 0.99, ((10, 0),))],
            )
            result = values[self.calls]
            self.calls += 1
            return result

    rewards = ResultRewardReader(StarSeparatedBackend()).read(
        np.full((1600, 2560, 3), 80, dtype=np.uint8)
    )

    assert rewards.recognized
    assert rewards.credits == 69_276
    assert rewards.ship_xp == 217
    assert rewards.free_xp == 44


def test_numeric_reward_ocr_retries_enhanced_crop_after_missing_original():
    from core.ocr import RapidOcrBackend

    class RetryBackend(RapidOcrBackend):
        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            self.calls += 1
            if self.calls == 1:
                return []
            return [
                OcrToken("198", 0.93, ((5, 2),)),
                OcrToken("363", 0.91, ((60, 2),)),
            ]

    reader = ResultRewardReader(RetryBackend())
    image = np.full((870, 1827, 3), 80, dtype=np.uint8)
    value, confidence, _raw = reader._read_number(
        image,
        RESULT_REWARD_REGIONS["credits"],
        reader.LIMITS["credits"],
        grouped_thousands=True,
        minimum_expected=reader.MINIMUM_CREDITS,
    )

    assert value == 198_363
    assert confidence >= 0.90
    assert reader.backend.calls == 2


def test_free_xp_does_not_join_star_icon_read_as_two():
    class IconPrefixBackend:
        execution_provider = "CUDAExecutionProvider"

        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [OcrToken("145", 0.99, ((10, 0),)), OcrToken("360", 0.98, ((80, 0),))],
                [OcrToken("1", 0.98, ((10, 0),)), OcrToken("719", 0.97, ((40, 0),))],
                [OcrToken("2", 0.91, ((5, 0),)), OcrToken("295☆", 0.97, ((35, 0),))],
            )
            value = values[self.calls]
            self.calls += 1
            return value

    rewards = ResultRewardReader(IconPrefixBackend()).read(
        np.full((1494, 2560, 3), 80, dtype=np.uint8)
    )

    assert rewards.ship_xp == 1_719
    assert rewards.free_xp == 295


def test_free_xp_repairs_icon_merged_into_one_token():
    class MergedIconBackend:
        execution_provider = "CUDAExecutionProvider"

        def __init__(self):
            self.calls = 0

        def recognize(self, _image):
            values = (
                [OcrToken("132 776", 0.99, ((10, 0),))],
                [OcrToken("1 425", 0.98, ((10, 0),))],
                [OcrToken("2295☆", 0.96, ((10, 0),))],
            )
            value = values[self.calls]
            self.calls += 1
            return value

    rewards = ResultRewardReader(MergedIconBackend()).read(
        np.full((1494, 2560, 3), 80, dtype=np.uint8)
    )

    assert rewards.ship_xp == 1_425
    assert rewards.free_xp == 295
