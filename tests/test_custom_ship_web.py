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

    def test_custom_ship_uses_tabs_and_saved_presets_scroll_inside_panel(self):
        html = (self.ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (self.ROOT / "app.js").read_text(encoding="utf-8")
        stylesheet = (self.ROOT / "custom_ship.css").read_text(encoding="utf-8")

        self.assertIn('data-ship-tab="presets"', html)
        self.assertIn('data-ship-tab="custom"', html)
        self.assertIn('id="saveCustomPresetBtn"', html)
        self.assertIn("/api/custom-ship-presets", javascript)
        self.assertIn("custom_ship_presets", javascript)
        self.assertIn(".ship-preset-scroll", stylesheet)
        self.assertIn("overflow-y: auto", stylesheet)

    def test_saved_presets_have_delete_action_and_stop_is_terminal(self):
        html = (self.ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (self.ROOT / "app.js").read_text(encoding="utf-8")
        stylesheet = (self.ROOT / "custom_ship.css").read_text(encoding="utf-8")

        self.assertIn("终止作战", html)
        self.assertIn("deleteCustomPreset", javascript)
        self.assertIn("/api/custom-ship-presets/delete", javascript)
        self.assertIn("state.terminating", javascript)
        self.assertIn(".delete-preset-button", stylesheet)

    def test_live_layout_is_aligned_and_log_follow_respects_user_scroll(self):
        html = (self.ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (self.ROOT / "app.js").read_text(encoding="utf-8")
        stylesheet = (self.ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn("operation-copy operation-banner", html)
        self.assertIn("initializeLogFollowing", javascript)
        self.assertIn("renderLiveLog(data.log)", javascript)
        self.assertIn("logIsAtBottom", javascript)
        self.assertIn("if(text===logViewState.lastText)return", javascript)
        self.assertIn(".operation-banner", stylesheet)
        self.assertIn("grid-template-rows:minmax(0,1fr) auto", stylesheet)


if __name__ == "__main__":
    unittest.main()
