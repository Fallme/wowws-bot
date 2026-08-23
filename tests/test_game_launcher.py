from pathlib import Path
from unittest.mock import Mock, patch

from core.launcher import STEAM_APP_ID, find_steam_executable, launch_game


def test_find_steam_executable_uses_explicit_override(tmp_path, monkeypatch):
    steam = tmp_path / "Steam.exe"
    steam.touch()
    monkeypatch.setenv("WOWS_STEAM_EXE", str(steam))

    assert find_steam_executable() == steam.resolve()


def test_launch_game_uses_steam_applaunch(tmp_path, monkeypatch):
    steam = tmp_path / "Steam.exe"
    steam.touch()
    monkeypatch.setenv("WOWS_STEAM_EXE", str(steam))
    popen = Mock()

    result = launch_game(popen=popen, startfile=Mock())

    assert result.started
    assert result.method == "steam"
    assert popen.call_args.args[0] == [str(steam.resolve()), "-applaunch", STEAM_APP_ID]
