from pathlib import Path
from unittest.mock import Mock, patch

from core.launcher import (
    STEAM_APP_ID,
    find_steam_executable,
    find_wgc_game_executable,
    launch_game,
)


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


def test_find_wgc_game_executable_searches_common_game_folder(
    tmp_path, monkeypatch
):
    executable = tmp_path / "Games" / "World_of_Warships" / "WorldOfWarships.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.delenv("WOWS_WGC_GAME_EXE", raising=False)
    monkeypatch.delenv("WOWS_GAME_EXE", raising=False)

    with patch("core.launcher._uninstall_entries", return_value=()):
        found = find_wgc_game_executable(search_roots=[tmp_path])

    assert found == executable.resolve()


def test_launch_game_uses_selected_wgc_install(tmp_path, monkeypatch):
    executable = tmp_path / "World_of_Warships" / "WorldOfWarships.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setenv("WOWS_WGC_GAME_EXE", str(executable))
    monkeypatch.delenv("WOWS_GAME_EXE", raising=False)
    popen = Mock()

    result = launch_game(client="wgc", popen=popen, startfile=Mock())

    assert result.started
    assert result.method == "wgc_game"
    assert popen.call_args.args[0] == [str(executable.resolve())]
    assert popen.call_args.kwargs["cwd"] == str(executable.resolve().parent)


def test_wgc_selection_never_falls_back_to_steam(monkeypatch):
    monkeypatch.delenv("WOWS_GAME_EXE", raising=False)
    popen = Mock()
    startfile = Mock()
    with (
        patch("core.launcher.find_wgc_game_executable", return_value=None),
        patch("core.launcher.find_wgc_executable", return_value=None),
    ):
        result = launch_game(client="wgc", popen=popen, startfile=startfile)

    assert not result.started
    assert result.method == "wgc_game"
    popen.assert_not_called()
    startfile.assert_not_called()
