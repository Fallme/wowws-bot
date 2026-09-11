from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from core.ocr import OcrToken
from core.results import ResultRewardReader
from port_navigator import _capture_ship_cards


def test_partially_overlapping_prefix_does_not_add_fifteen_million():
    tokens = [OcrToken('15', .99, ((37,9),(86,9),(85,61),(36,60))),
              OcrToken('156', .98, ((61,3),(139,3),(139,69),(61,69))),
              OcrToken('224', .99, ((148,0),(265,2),(264,70),(147,69)))]
    reader = ResultRewardReader(SimpleNamespace(recognize=lambda _: tokens))
    value, _, _ = reader._read_number_once(np.zeros((80,280,3),dtype=np.uint8), 100000000, grouped_thousands=True)
    assert value == 156224


def test_cursor_is_parked_before_ship_capture():
    calls = []
    with patch('port_navigator.park_port_cursor', side_effect=lambda _: calls.append('park')), patch('port_navigator._capture', side_effect=lambda _: calls.append('capture')):
        _capture_ship_cards(123)
    assert calls == ['park', 'capture']
