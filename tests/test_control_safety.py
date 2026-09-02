from unittest.mock import Mock, patch

import pytest

from control_server import (
    ElevationRequiredError,
    RunnerManager,
    ensure_elevated_control_server,
    running_game_requires_elevation,
)


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
        patch.dict("control_server.os.environ", {"WOWS_PANEL_FORCE_ELEVATION": "1"}),
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
        patch("control_server.running_game_requires_elevation", return_value=False),
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
        patch("control_server.running_game_requires_elevation", return_value=False),
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


def test_runner_requests_elevation_only_for_verified_game_mismatch():
    manager = RunnerManager(Mock())
    with patch("control_server.running_game_requires_elevation", return_value=True):
        with pytest.raises(ElevationRequiredError):
            manager.start(
                {
                    "ship": "pommern",
                    "mode": "cooperative",
                    "limit_type": "rounds",
                    "limit_value": 1,
                }
            )


def test_game_elevation_check_uses_verified_largest_game_window():
    user32 = Mock()

    def write_process_id(_hwnd, pointer):
        pointer._obj.value = 4321
        return 1

    user32.GetWindowThreadProcessId.side_effect = write_process_id
    windows = [
        (10, "helper", (0, 0, 200, 100)),
        (20, "World of Warships", (0, 0, 2560, 1440)),
    ]
    reader = Mock(return_value=True)
    with (
        patch("control_server.os.name", "nt"),
        patch("control_server.current_process_is_elevated", return_value=False),
        patch("control_server.ctypes.windll.user32", user32),
    ):
        assert running_game_requires_elevation(
            window_finder=lambda: windows,
            elevation_reader=reader,
        )

    user32.GetWindowThreadProcessId.assert_called_once()
    assert user32.GetWindowThreadProcessId.call_args.args[0] == 20
    reader.assert_called_once_with(4321)


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
