from types import SimpleNamespace

import pytest

import control_server


def test_screenshot_cleanup_is_scoped_to_project_image_root(tmp_path, monkeypatch):
    root = tmp_path / "runtime" / "screenshots"
    run = root / "runs" / "run_1"
    run.mkdir(parents=True)
    (run / "frame.png").write_bytes(b"frame")
    (run / "events.jsonl").write_text("{}\n", encoding="utf-8")
    reference = tmp_path / "training_assets" / "reference.png"
    reference.parent.mkdir()
    reference.write_bytes(b"keep")
    monkeypatch.setattr(control_server, "BASE_DIR", tmp_path)
    monkeypatch.setattr(control_server, "SCREENSHOT_ROOT", root)
    monkeypatch.setattr(
        control_server,
        "RUNNER",
        SimpleNamespace(status=lambda: {"running": False}),
    )

    result = control_server.clear_generated_screenshots()

    assert result["removed"] == 1
    assert not (run / "frame.png").exists()
    assert (run / "events.jsonl").exists()
    assert reference.exists()


def test_screenshot_cleanup_is_blocked_during_active_run(tmp_path, monkeypatch):
    monkeypatch.setattr(control_server, "SCREENSHOT_ROOT", tmp_path)
    monkeypatch.setattr(
        control_server,
        "RUNNER",
        SimpleNamespace(status=lambda: {"running": True}),
    )

    with pytest.raises(RuntimeError, match="任务运行中"):
        control_server.clear_generated_screenshots()
