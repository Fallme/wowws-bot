import os
import unittest
from unittest.mock import patch

from config_loader import load_ship_config, ship_key_from_env


class ConfigLoaderTests(unittest.TestCase):
    def test_loads_known_ship(self):
        ship = load_ship_config("pommern")
        self.assertEqual(ship["name"], "Pommern")
        self.assertGreater(ship["secondary"]["range"], 0)

    def test_unknown_ship_lists_available_choices(self):
        with self.assertRaisesRegex(KeyError, "available ships"):
            load_ship_config("missing")

    def test_ship_key_is_normalized(self):
        with patch.dict(os.environ, {"WOWS_SHIP": " Napoli "}, clear=True):
            self.assertEqual(ship_key_from_env(), "napoli")

    def test_empty_ship_key_is_rejected(self):
        with patch.dict(os.environ, {"WOWS_SHIP": "  "}, clear=True):
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                ship_key_from_env()

    def test_custom_ship_uses_exact_name_and_secondary_range(self):
        with patch.dict(
            os.environ,
            {
                "WOWS_CUSTOM_SHIP_NAME": "大选帝侯",
                "WOWS_CUSTOM_SECONDARY_RANGE": "12.1",
            },
            clear=True,
        ):
            ship = load_ship_config("custom")
        self.assertEqual(ship["name"], "大选帝侯")
        self.assertEqual(ship["display_name"], "大选帝侯")
        self.assertEqual(ship["secondary"]["range"], 12.1)
        self.assertFalse(ship["has_torpedoes"])
        self.assertFalse(ship["has_smoke"])
        self.assertLess(
            ship["strategy"]["secondary_target_distance_km"],
            ship["secondary"]["range"],
        )

    def test_custom_ship_rejects_missing_name(self):
        with patch.dict(
            os.environ,
            {"WOWS_CUSTOM_SECONDARY_RANGE": "10.0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "完整名称"):
                load_ship_config("custom")

    def test_custom_ship_rejects_invalid_secondary_range(self):
        with patch.dict(
            os.environ,
            {
                "WOWS_CUSTOM_SHIP_NAME": "大选帝侯",
                "WOWS_CUSTOM_SECONDARY_RANGE": "50",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "1.0 到 30.0"):
                load_ship_config("custom")


if __name__ == "__main__":
    unittest.main()
