from unittest.mock import Mock, patch

from control_server import RunnerManager, ensure_elevated_control_server


def test_control_server_keeps_running_when_already_elevated():
    shell = Mock()
    shell.IsUserAnAdmin.return_value = 1
    with (
        patch("control_server.os.name", "nt"),
        patch("control_server.ctypes.windll.shell32", shell),
        patch.dict("control_server.os.environ", {"WOWS_PANEL_SKIP_ELEVATION": "0"}),
    ):
        assert ensure_elevated_control_server()


def test_control_server_restarts_itself_elevated_once():
    shell = Mock()
    shell.IsUserAnAdmin.return_value = 0
    shell.ShellExecuteW.return_value = 33
    with (
        patch("control_server.os.name", "nt"),
        patch("control_server.ctypes.windll.shell32", shell),
        patch("control_server.sys.executable", r"E:\aimemo\wowws-bot\.venv\Scripts\python.exe"),
        patch.dict("control_server.os.environ", {"WOWS_PANEL_SKIP_ELEVATION": "0"}),
    ):
        assert not ensure_elevated_control_server()

    shell.ShellExecuteW.assert_called_once()


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
