from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_console_exposes_client_selector_and_sends_selection():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert 'id="launcherGrid"' in html
    assert "Steam" in javascript
    assert "WG Game Center" in javascript
    assert "launcher_client:state.launcher" in javascript
    assert "wowws.launcherClient" in javascript
