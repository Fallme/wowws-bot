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


def test_runner_passes_selected_launcher_to_worker(tmp_path):
    manager = RunnerManager(Mock())
    process = Mock(pid=1234)
    process.poll.return_value = None
    with (
        patch("control_server.LOG_PATH", tmp_path / "runtime.log"),
        patch("control_server.STATE_PATH", tmp_path / "state.json"),
        patch("control_server.STOP_PATH", tmp_path / "stop.request"),
        patch("control_server.PAUSE_PATH", tmp_path / "pause.request"),
        patch("control_server.RESUME_PATH", tmp_path / "resume.request"),
        patch("control_server.subprocess.Popen", return_value=process) as popen,
    ):
        manager.start(
            {
                "ship": "pommern",
                "mode": "cooperative",
                "launcher_client": "wgc",
                "limit_type": "rounds",
                "limit_value": 1,
            }
        )

    assert popen.call_args.kwargs["env"]["WOWS_LAUNCHER_CLIENT"] == "wgc"


def test_pause_and_resume_never_create_a_stop_request(tmp_path):
    manager = RunnerManager(Mock())
    process = Mock(pid=1234)
    process.poll.return_value = None
    manager.process = process
    pause = tmp_path / "pause.request"
    resume = tmp_path / "resume.request"
    stop = tmp_path / "stop.request"

    with (
        patch("control_server.PAUSE_PATH", pause),
        patch("control_server.RESUME_PATH", resume),
        patch("control_server.STOP_PATH", stop),
    ):
        assert manager.pause()
        assert pause.exists()
        assert not resume.exists()
        assert not stop.exists()

        assert manager.resume()
        assert not pause.exists()
        assert resume.exists()
        assert not stop.exists()
        assert manager.process is process
