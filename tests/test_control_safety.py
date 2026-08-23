from unittest.mock import Mock, patch

from control_server import RunnerManager
def test_runner_starts_without_manual_calibration(tmp_path):
    manager = RunnerManager(Mock())
    process = Mock(pid=1234)
    process.poll.return_value = None
    with (
        patch("control_server.LOG_PATH", tmp_path / "runtime.log"),
        patch("control_server.STATE_PATH", tmp_path / "state.json"),
        patch("control_server.STOP_PATH", tmp_path / "stop.request"),
        patch("control_server.subprocess.Popen", return_value=process),
    ):
        result = manager.start(
            {
                "ship": "pommern",
                "mode": "cooperative",
                "limit_type": "rounds",
                "limit_value": 1,
            }
        )

    assert result["run_id"]
    assert manager.process is process
