import unittest
from pathlib import Path


class CustomShipWebTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1] / "frontend"

    def test_custom_ship_form_and_local_persistence_are_present(self):
        html = (self.ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (self.ROOT / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="customShipName"', html)
        self.assertIn('id="customSecondaryRange"', html)
        self.assertIn("localStorage.setItem", javascript)
        self.assertIn("custom_ship_name", javascript)
        self.assertIn("custom_secondary_range", javascript)


if __name__ == "__main__":
    unittest.main()
