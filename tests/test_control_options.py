import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from control_server import (
    MODES,
    delete_custom_ship_preset,
    load_custom_ship,
    load_custom_ship_presets,
    load_ships,
    resolve_custom_ship_for_run,
    save_custom_ship,
    save_custom_ship_preset,
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

    def test_custom_ship_presets_are_reusable_and_update_by_name(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom_ship_presets.json"
            with patch("control_server.CUSTOM_SHIP_PRESETS_PATH", path):
                first, presets = save_custom_ship_preset("石见", 10.0)
                self.assertEqual(len(presets), 1)
                updated, presets = save_custom_ship_preset("石见", 10.5)

                self.assertEqual(updated["id"], first["id"])
                self.assertEqual(len(presets), 1)
                self.assertEqual(
                    load_custom_ship_presets(),
                    [
                        {
                            "id": first["id"],
                            "name": "石见",
                            "secondary_range": 10.5,
                        }
                    ],
                )

    def test_custom_ship_preset_can_be_deleted_by_id(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom_ship_presets.json"
            with patch("control_server.CUSTOM_SHIP_PRESETS_PATH", path):
                first, _ = save_custom_ship_preset("石见", 10.0)
                save_custom_ship_preset("大选帝侯", 12.5)

                remaining = delete_custom_ship_preset(first["id"])

                self.assertEqual([item["name"] for item in remaining], ["大选帝侯"])
                self.assertEqual(load_custom_ship_presets(), remaining)

    def test_run_uses_selected_preset_id_instead_of_stale_custom_form(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom_ship_presets.json"
            with patch("control_server.CUSTOM_SHIP_PRESETS_PATH", path):
                preset, _ = save_custom_ship_preset("冯·祖克霍夫", 12.0)

                resolved = resolve_custom_ship_for_run(
                    {
                        "custom_ship_preset_id": preset["id"],
                        "custom_ship_name": "石见",
                        "custom_secondary_range": 10.0,
                    }
                )

                self.assertEqual(resolved, ("冯·祖克霍夫", 12.0))


if __name__ == "__main__":
    unittest.main()
