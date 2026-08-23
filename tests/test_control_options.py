import unittest

from control_server import MODES, load_ships


class ControlOptionsTests(unittest.TestCase):
    def test_only_requested_modes_are_available(self):
        self.assertEqual(set(MODES), {"cooperative", "asymmetric"})

    def test_only_requested_ships_are_available(self):
        ships = load_ships()
        self.assertEqual(set(ships), {"pommern", "napoli"})
        self.assertEqual(ships["pommern"]["name"], "波美拉尼亚")
        self.assertEqual(ships["napoli"]["name"], "那不勒斯")


if __name__ == "__main__":
    unittest.main()
