import numpy as np

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


def test_result_reward_reader_rejects_missing_credit_value():
    class EmptyBackend:
        def recognize(self, _image):
            return []

    rewards = ResultRewardReader(EmptyBackend()).read(
        np.full((1600, 2560, 3), 80, dtype=np.uint8)
    )

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
