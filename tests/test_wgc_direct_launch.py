"""Direct WG startup selects the installed game build and never its bootstrapper."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET
import subprocess

import pytest

from core.launcher import find_wgc_direct_game_executable, launch_game


CURRENT_BUILD = "13187581"
FUTURE_BUILD = "13357625"


def metadata(directory, *, installed=f"15.8.0.0.{CURRENT_BUILD}",
             available=f"15.9.0.0.{FUTURE_BUILD}", game_id="WOWS.WW.PRODUCTION"):
    root = ET.Element("protocol")
    game = ET.SubElement(root, "game")
    ET.SubElement(game, "id").text = game_id
    if installed is not None:
        parts = ET.SubElement(game, "part_versions")
        ET.SubElement(parts, "version", name="client", installed=installed, available=available)
    ET.ElementTree(root).write(directory / "game_info.xml", encoding="utf-8")


def binary(directory, build=CURRENT_BUILD):
    path = directory / "bin" / build / "bin64" / "WorldOfWarships64.exe"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


def current_version(directory, build):
    root = ET.Element("current_version")
    ET.SubElement(root, "bin").text = build
    ET.ElementTree(root).write(directory / "current_version.xml", encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated_launcher_state(tmp_path, monkeypatch):
    monkeypatch.setattr("core.launcher.LAST_CLIENT_PATH", tmp_path / "last_launcher.txt")
    for name in ("WOWS_WGC_LAUNCH_MODE", "WOWS_WGC_GAME_EXE", "WOWS_GAME_EXE", "WOWS_LAUNCHER_CLIENT"):
        monkeypatch.delenv(name, raising=False)


def test_installed_client_build_wins_over_available_and_newest_directory(tmp_path):
    metadata(tmp_path)
    expected = binary(tmp_path)
    binary(tmp_path, FUTURE_BUILD)
    binary(tmp_path, "99999999")
    assert find_wgc_direct_game_executable(tmp_path) == expected


def test_installed_build_wins_over_conflicting_current_version_file(tmp_path):
    metadata(tmp_path)
    expected = binary(tmp_path)
    binary(tmp_path, FUTURE_BUILD)
    current_version(tmp_path, FUTURE_BUILD)
    assert find_wgc_direct_game_executable(tmp_path) == expected


def test_current_build_bin64_wins_over_bin32_or_root_bootstrapper(tmp_path):
    metadata(tmp_path)
    expected = binary(tmp_path)
    bin32 = tmp_path / "bin" / CURRENT_BUILD / "bin32" / "WorldOfWarships.exe"
    bin32.parent.mkdir()
    bin32.touch()
    (tmp_path / "WorldOfWarships.exe").touch()
    assert find_wgc_direct_game_executable(tmp_path) == expected


def test_missing_installed_binary_does_not_choose_an_old_or_future_build(tmp_path):
    metadata(tmp_path)
    binary(tmp_path, "13015811")
    binary(tmp_path, FUTURE_BUILD)
    (tmp_path / "WorldOfWarships.exe").touch()
    assert find_wgc_direct_game_executable(tmp_path) is None


def test_root_executables_are_not_direct_versioned_game_binaries(tmp_path):
    metadata(tmp_path)
    (tmp_path / "WorldOfWarships.exe").touch()
    (tmp_path / "WorldOfWarships64.exe").touch()
    assert find_wgc_direct_game_executable(tmp_path) is None


def test_missing_installed_metadata_does_not_guess_the_highest_build(tmp_path):
    metadata(tmp_path, installed=None)
    binary(tmp_path)
    binary(tmp_path, FUTURE_BUILD)
    assert find_wgc_direct_game_executable(tmp_path) is None


def test_valid_current_version_bin_is_an_explicit_metadata_fallback(tmp_path):
    metadata(tmp_path, installed=None)
    expected = binary(tmp_path)
    binary(tmp_path, FUTURE_BUILD)
    current_version(tmp_path, CURRENT_BUILD)
    assert find_wgc_direct_game_executable(tmp_path) == expected


@pytest.mark.parametrize("build", ["../13187581", "..\\13187581", "bin/13187581", "latest"])
def test_current_version_bin_must_be_a_numeric_build_not_a_path(tmp_path, build):
    metadata(tmp_path, installed=None)
    binary(tmp_path)
    current_version(tmp_path, build)
    assert find_wgc_direct_game_executable(tmp_path) is None


@pytest.mark.parametrize("contents", [None, "<protocol><game>", "<protocol><game><id>WOT.WW.PRODUCTION</id></game></protocol>"])
def test_unverified_game_metadata_never_authorizes_direct_launch(tmp_path, contents):
    binary(tmp_path)
    current_version(tmp_path, CURRENT_BUILD)
    if contents is not None:
        (tmp_path / "game_info.xml").write_text(contents, encoding="utf-8")
    assert find_wgc_direct_game_executable(tmp_path) is None


def test_default_wg_launch_dispatches_real_binary_without_game_center(tmp_path, monkeypatch):
    metadata(tmp_path)
    expected = binary(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: bootstrapper)
    monkeypatch.setattr("core.launcher.find_wgc_executable", lambda: None)
    popen = Mock()
    result = launch_game(client="wgc", popen=popen, startfile=Mock())
    assert result.started
    assert result.method == "wgc_direct"
    assert Path(popen.call_args.args[0][0]) == expected
    assert Path(popen.call_args.kwargs["cwd"]) == tmp_path.resolve()
    assert result.game_directory == str(tmp_path.resolve())


def test_direct_launch_matches_wg_app_type_and_shared_profile_directory(tmp_path, monkeypatch):
    metadata(tmp_path)
    expected = binary(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    (tmp_path / "app_type.xml").write_text(
        '<protocol><app_type>high</app_type></protocol>', encoding="utf-8")
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: bootstrapper)
    popen = Mock()
    assert launch_game(client="wgc", popen=popen).started
    assert popen.call_args.args[0] == [str(expected), "--apptype=high"]
    assert Path(popen.call_args.kwargs["cwd"]) == tmp_path.resolve()


def test_cold_native_exit_retries_engine_once_without_launcher_command(tmp_path, monkeypatch):
    metadata(tmp_path)
    native = binary(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: bootstrapper)
    monkeypatch.setattr("core.launcher.wait_for_wgc_ready", Mock(return_value=True))
    early = Mock()
    early.wait.return_value = 0
    running = Mock()
    running.wait.side_effect = subprocess.TimeoutExpired(str(native), 4)
    popen = Mock(side_effect=[early, running])
    result = launch_game(client="wgc", popen=popen)
    assert result.started
    assert "已重试1次" in result.detail
    assert popen.call_count == 2
    assert all(call.args[0] == [str(native)] for call in popen.call_args_list)
    assert all(call.kwargs["cwd"] == str(tmp_path.resolve()) for call in popen.call_args_list)


def test_repeated_native_early_exits_stop_instead_of_waiting_for_missing_window(tmp_path, monkeypatch):
    metadata(tmp_path)
    binary(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: bootstrapper)
    monkeypatch.setattr("core.launcher.wait_for_wgc_ready", Mock(return_value=True))
    exited = Mock()
    exited.wait.return_value = 1
    popen = Mock(return_value=exited)
    result = launch_game(client="wgc", popen=popen)
    assert not result.started
    assert "连续提前退出" in result.detail
    assert popen.call_count == 2


def test_missing_direct_binary_never_automatically_opens_game_center(tmp_path, monkeypatch):
    metadata(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    center = tmp_path / "wgc.exe"
    center.touch()
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: bootstrapper)
    monkeypatch.setattr("core.launcher.find_wgc_executable", lambda: center)
    popen = Mock()
    result = launch_game(client="wgc", popen=popen, startfile=Mock())
    assert not result.started
    assert result.method != "wgc_ui"
    popen.assert_not_called()


@pytest.mark.parametrize("variable", ["WOWS_WGC_GAME_EXE", "WOWS_GAME_EXE"])
@pytest.mark.parametrize("override_build", ["13015811", FUTURE_BUILD])
def test_explicit_bin64_override_only_locates_install_and_cannot_pin_version(
        tmp_path, monkeypatch, variable, override_build):
    metadata(tmp_path)
    expected = binary(tmp_path)
    explicit = binary(tmp_path, override_build)
    monkeypatch.setenv(variable, str(explicit))
    monkeypatch.setattr("core.launcher.find_wgc_executable", lambda: None)
    popen = Mock()
    result = launch_game(client="wgc", popen=popen, startfile=Mock())
    assert result.started
    assert result.method == "wgc_direct"
    assert Path(popen.call_args.args[0][0]) == expected


def test_explicit_root_bootstrapper_resolves_the_installed_direct_binary(tmp_path, monkeypatch):
    metadata(tmp_path)
    expected = binary(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    monkeypatch.setenv("WOWS_WGC_GAME_EXE", str(bootstrapper))
    monkeypatch.setattr("core.launcher.find_wgc_executable", lambda: None)
    popen = Mock()
    result = launch_game(client="wgc", popen=popen, startfile=Mock())
    assert result.started
    assert result.method == "wgc_direct"
    assert Path(popen.call_args.args[0][0]) == expected


@pytest.mark.parametrize("mode", ["launcher", "direct", "obsolete"])
def test_old_launch_mode_cannot_open_a_launcher(tmp_path, monkeypatch, mode):
    metadata(tmp_path)
    expected = binary(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    center = tmp_path / "wgc.exe"
    center.touch()
    monkeypatch.setenv("WOWS_WGC_LAUNCH_MODE", mode)
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: bootstrapper)
    monkeypatch.setattr("core.launcher.find_wgc_executable", lambda: center)
    popen, startfile = Mock(), Mock()
    result = launch_game(client="wgc", popen=popen, startfile=startfile)
    assert result.started and result.method == "wgc_direct"
    assert popen.call_args.args[0] == [str(expected)]
    assert result.launcher_path == ""
    startfile.assert_not_called()


def test_launch_rereads_installed_version_after_update_even_with_old_override(tmp_path, monkeypatch):
    metadata(tmp_path)
    before, after = binary(tmp_path), binary(tmp_path, FUTURE_BUILD)
    monkeypatch.setenv("WOWS_GAME_EXE", str(before))
    popen = Mock()
    assert launch_game(client="wgc", popen=popen).started
    assert popen.call_args.args[0] == [str(before)]
    metadata(tmp_path, installed=f"15.9.0.0.{FUTURE_BUILD}")
    assert launch_game(client="wgc", popen=popen).started
    assert popen.call_args.args[0] == [str(after)]
    assert popen.call_args.kwargs["cwd"] == str(tmp_path.resolve())


def test_removed_override_binary_after_update_still_resolves_verified_install(tmp_path, monkeypatch):
    metadata(tmp_path, installed=f"15.9.0.0.{FUTURE_BUILD}")
    expected = binary(tmp_path, FUTURE_BUILD)
    missing_old = tmp_path / "bin" / CURRENT_BUILD / "bin64" / "WorldOfWarships64.exe"
    monkeypatch.setenv("WOWS_WGC_GAME_EXE", str(missing_old))
    popen = Mock()
    assert launch_game(client="wgc", popen=popen).started
    assert popen.call_args.args[0] == [str(expected)]


def test_stale_override_cannot_launch_when_current_binary_is_missing(tmp_path, monkeypatch):
    metadata(tmp_path, installed=f"15.9.0.0.{FUTURE_BUILD}")
    old = binary(tmp_path)
    monkeypatch.setenv("WOWS_GAME_EXE", str(old))
    monkeypatch.setenv("WOWS_WGC_LAUNCH_MODE", "launcher")
    popen, startfile = Mock(), Mock()
    result = launch_game(client="wgc", popen=popen, startfile=startfile)
    assert not result.started
    assert result.method == "wgc_direct"
    popen.assert_not_called()
    startfile.assert_not_called()


def test_runtime_entry_waits_for_game_after_direct_dispatch_with_legacy_environment(tmp_path, monkeypatch):
    import main

    class ReachedGameControl(Exception):
        pass

    metadata(tmp_path)
    expected = binary(tmp_path)
    bootstrapper = tmp_path / "WorldOfWarships.exe"
    bootstrapper.touch()
    monkeypatch.setattr("core.launcher.find_wgc_game_executable", lambda: bootstrapper)
    monkeypatch.setenv("WOWS_LAUNCHER_CLIENT", "wgc")
    monkeypatch.setenv("WOWS_WGC_LAUNCH_MODE", "launcher")
    monkeypatch.setenv("WOWS_MODE", "cooperative")
    config = {"name": "test_ship", "secondary": {"range": 6}}
    limits = SimpleNamespace(quick_battle=False, stop_requested=lambda: False)
    monkeypatch.setattr(main, "configure_dpi_awareness", Mock())
    monkeypatch.setattr(main, "configure_logging", Mock())
    monkeypatch.setattr(main, "ship_key_from_env", lambda: "test_ship")
    monkeypatch.setattr(main, "load_ship_config", lambda _: config)
    monkeypatch.setattr(main.RunLimits, "from_env", lambda: limits)
    monkeypatch.setattr(main, "RuntimeReporter", Mock())
    game_window = (42, "World of Warships", (0, 0, 1920, 1080))
    waits = Mock(side_effect=[None, game_window])
    monkeypatch.setattr(main, "wait_for_game_window", waits)
    popen = Mock()
    monkeypatch.setattr(main, "launch_game", lambda **kwargs: launch_game(popen=popen, **kwargs))
    control = Mock(side_effect=ReachedGameControl)
    monkeypatch.setattr(main, "BattleBot", control)
    # Stop at the first game-control handoff; no real window/input API runs.
    with pytest.raises(ReachedGameControl):
        main.run()
    assert popen.call_args.args[0] == [str(expected)]
    assert waits.call_count == 2
    control.assert_called_once_with(42, config)
