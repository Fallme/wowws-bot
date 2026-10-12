from pathlib import Path
from unittest.mock import Mock, patch
import pytest


@pytest.fixture(autouse=True)
def isolated_launcher_history(tmp_path, monkeypatch):
    # A stale legacy environment setting must not reopen the launcher.
    monkeypatch.setenv("WOWS_WGC_LAUNCH_MODE", "launcher")
    monkeypatch.setattr("core.launcher.LAST_CLIENT_PATH", tmp_path / "launcher.txt")
    for name in ("WOWS_GAME_EXE", "WOWS_WGC_GAME_EXE", "WOWS_LAUNCHER_CLIENT"):
        monkeypatch.delenv(name, raising=False)
    center = tmp_path / "GameCenter" / "wgc.exe"
    center.parent.mkdir()
    center.touch()
    monkeypatch.setattr("core.launcher.find_wgc_executable", lambda: center)


def write_game_info(directory):
    (directory / "game_info.xml").write_text(
        '<protocol><game><id>WOWS.WW.PRODUCTION</id><part_versions>'
        '<version name="client" installed="15.8.0.0.13187581"/>'
        '</part_versions></game></protocol>', encoding="utf-8")
    native = directory / "bin" / "13187581" / "bin64" / "WorldOfWarships64.exe"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.touch()


def test_default_launch_reuses_last_client(tmp_path, monkeypatch):
    from core.launcher import remember_launch_client, last_launch_client
    game = tmp_path / "WorldOfWarships.exe"
    game.touch()
    write_game_info(game.parent)
    monkeypatch.delenv("WOWS_LAUNCHER_CLIENT", raising=False)
    monkeypatch.delenv("WOWS_GAME_EXE", raising=False)
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: game)
    remember_launch_client("wgc")
    result = launch_game(popen=Mock())
    assert result.method == "wgc_direct"
    assert last_launch_client() == "wgc"

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
    write_game_info(executable.parent)
    monkeypatch.delenv("WOWS_GAME_EXE", raising=False)
    popen = Mock()

    result = launch_game(client="wgc", popen=popen, startfile=Mock())

    assert result.started
    assert result.method == "wgc_direct"
    native = executable.parent / "bin" / "13187581" / "bin64" / "WorldOfWarships64.exe"
    assert popen.call_args.args[0] == [str(native.resolve())]
    assert popen.call_args.kwargs["cwd"] == str(executable.parent.resolve())
    assert result.launcher_path == ""
    assert result.game_directory == str(executable.parent.resolve())
    assert "等待真实游戏窗口" in result.detail


def test_wgc_binary_override_resolves_install_root(tmp_path, monkeypatch):
    game = tmp_path / "game" / "bin" / "123" / "bin64" / "WorldOfWarships64.exe"
    game.parent.mkdir(parents=True)
    game.touch()
    write_game_info(tmp_path / "game")
    monkeypatch.setenv("WOWS_GAME_EXE", str(game))
    result = launch_game(client="wgc", popen=Mock())
    assert result.method == "wgc_direct"
    assert result.game_directory == str((tmp_path / "game").resolve())


def test_wgc_uses_registered_install_on_nonstandard_path(tmp_path):
    folder = tmp_path / "custom_games" / "ships"
    folder.mkdir(parents=True)
    (folder / "WorldOfWarships.exe").touch()
    record = tmp_path / "GameCenter" / "apps" / "wows.ww.production" / "record"
    record.parent.mkdir(parents=True)
    record.write_text(str(folder), encoding="utf-8")
    with patch("core.launcher._uninstall_entries", return_value=()):
        assert find_wgc_game_executable(search_roots=[]) == (folder / "WorldOfWarships.exe").resolve()


def test_wgc_invalid_metadata_never_dispatches_bootstrapper(tmp_path, monkeypatch):
    game = tmp_path / "WorldOfWarships.exe"
    game.touch()
    monkeypatch.setenv("WOWS_WGC_GAME_EXE", str(game))
    popen = Mock()
    assert not launch_game(client="wgc", popen=popen).started
    popen.assert_not_called()


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
