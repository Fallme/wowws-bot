"""Replay readiness markers from the real October 12 cold-start failure."""

from pathlib import Path

from core import wgc_readiness


def test_recorded_initialization_at_failed_retry_time_is_not_ready(tmp_path, monkeypatch):
    recorded = (Path(__file__).parent / "fixtures" / "wgc_cold_start_20261012.log").read_text(encoding="utf-8")
    center = tmp_path / "GameCenter" / "wgc.exe"
    logs = center.parent / "logs"
    logs.mkdir(parents=True)
    log = logs / "wgc_recorded.log"
    monkeypatch.setattr(wgc_readiness, "running_wgc_process_ids", lambda: {30600})
    monkeypatch.setattr(wgc_readiness, "_session_matches_process", lambda *_args: True)
    directory = Path("E:/Games/World_of_Warships")

    # Retry was dispatched at 12:39:34.988 and exited at 12:39:35.121,
    # before the final postload marker in this recorded trace.
    log.write_text("\n".join(recorded.splitlines()[:3]), encoding="utf-8")
    assert not wgc_readiness.wgc_backend_ready(center, directory)

    log.write_text(recorded, encoding="utf-8")
    assert wgc_readiness.wgc_backend_ready(center, directory)
