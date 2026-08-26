import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from control_server import (
    MODES,
    load_custom_ship,
    load_ships,
    save_custom_ship,
    validate_custom_ship,
)


class ControlOptionsTests(unittest.TestCase):
    def test_only_requested_modes_are_available(self):
        self.assertEqual(set(MODES), {"cooperative", "asymmetric"})

    def test_only_requested_ships_are_available(self):
        ships = load_ships()
        self.assertEqual(set(ships), {"pommern", "napoli"})
        self.assertEqual(ships["pommern"]["name"], "波美拉尼亚")
        self.assertEqual(ships["napoli"]["name"], "那不勒斯")

    def test_custom_ship_settings_survive_control_server_reload(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom_ship.json"
            with patch("control_server.CUSTOM_SHIP_PATH", path):
                save_custom_ship("石见", 10.0)
                self.assertEqual(
                    load_custom_ship(),
                    {"name": "石见", "secondary_range": 10.0},
                )

    def test_custom_ship_payload_validation(self):
        self.assertEqual(
            validate_custom_ship(
                {
                    "custom_ship_name": " 石见 ",
                    "custom_secondary_range": "10.0",
                }
            ),
            ("石见", 10.0),
        )
        with self.assertRaisesRegex(ValueError, "副炮射程"):
            validate_custom_ship(
                {
                    "custom_ship_name": "石见",
                    "custom_secondary_range": 31,
                }
            )


if __name__ == "__main__":
    unittest.main()
