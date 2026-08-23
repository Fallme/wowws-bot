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


if __name__ == "__main__":
    unittest.main()
