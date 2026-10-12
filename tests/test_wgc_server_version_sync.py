"""WG direct startup follows the selected realm's server version, not preloads."""

from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

import pytest

from core import launcher
from core import wgc_version


OLD_VERSION = "15.8.0.13187581"
NEW_VERSION = "15.9.0.13357625"
REALMS = {
    "asia": ("WOWS ASIA", "https://asia.example.test/csis", "asia-master.example.test:20020"),
    "eu": ("WOWS EU", "https://eu.example.test/csis", "eu-master.example.test:20020"),
}
FETCH_CSIS_VERSION = wgc_version.fetch_csis_version


def write_install(directory, *, installed="15.9.0.0.13357625"):
    root = ET.Element("protocol")
    game = ET.SubElement(root, "game")
    ET.SubElement(game, "id").text = "WOWS.WW.PRODUCTION"
    ET.SubElement(game, "installed").text = "true"
    versions = ET.SubElement(game, "part_versions")
    ET.SubElement(versions, "version", name="client", installed=installed,
                  available="15.9.0.0.13357625")
    ET.ElementTree(root).write(directory / "game_info.xml", encoding="utf-8")


def write_engine(directory, version, *, realms=("asia",)):
    build = version.rsplit(".", 1)[-1]
    native = directory / "bin" / build / "bin64" / "WorldOfWarships64.exe"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.touch()
    config = ET.Element("scripts_config")
    server = ET.SubElement(config, "server")
    for realm in realms:
        alias, endpoint, master = REALMS[realm]
        host = ET.SubElement(server, "host", alias=alias, realm=realm, csis=endpoint)
        ET.SubElement(host, "master", addr=master)
    resources = native.parent.parent / "res"
    resources.mkdir(exist_ok=True)
    ET.ElementTree(config).write(resources / "scripts_config.xml", encoding="utf-8")
    return native.resolve()


def write_preferences(directory, *, active="WOWS ASIA", version=OLD_VERSION):
    root = ET.Element("preferences")
    scripts = ET.SubElement(root, "scriptsPreferences")
    if active is not None:
        ET.SubElement(scripts, "active_server").text = active
    if version is not None:
        ET.SubElement(scripts, "last_server_version").text = version
    ET.ElementTree(root).write(directory / "preferences.xml", encoding="utf-8")


def seed_cache(directory, values):
    wgc_version.VERSION_CACHE_PATH.write_text(
        json.dumps({str(directory.resolve()).casefold(): values}), encoding="utf-8")


@pytest.fixture(autouse=True)
def isolate_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(wgc_version, "VERSION_CACHE_PATH", tmp_path / "server_versions.json")
    monkeypatch.setattr(launcher, "LAST_CLIENT_PATH", tmp_path / "last_launcher.txt")
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(
        side_effect=AssertionError("Unexpected network access")))
    for name in ("WOWS_WGC_LAUNCH_MODE", "WOWS_WGC_GAME_EXE", "WOWS_GAME_EXE", "WOWS_LAUNCHER_CLIENT"):
        monkeypatch.delenv(name, raising=False)


def multiversion_install(directory, *, realms=("asia",), last_version=OLD_VERSION):
    write_install(directory)
    old = write_engine(directory, OLD_VERSION, realms=realms)
    new = write_engine(directory, NEW_VERSION, realms=realms)
    write_preferences(directory, version=last_version)
    return old, new


def test_server_enabled_old_build_wins_over_downloaded_new_client(tmp_path, monkeypatch):
    old, _ = multiversion_install(tmp_path)
    fetch = Mock(return_value=OLD_VERSION)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", fetch)

    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable == old
    assert (selected.version, selected.realm, selected.source) == (OLD_VERSION, "asia", "csis")
    endpoint = urlsplit(fetch.call_args.args[0])
    assert endpoint.hostname == "asia.example.test"
    assert parse_qs(endpoint.query)["periphery"] == [REALMS["asia"][2]]


def test_each_launch_refreshes_build_when_server_switches_version(tmp_path, monkeypatch):
    old, new = multiversion_install(tmp_path)
    fetch = Mock(side_effect=[OLD_VERSION, NEW_VERSION])
    monkeypatch.setattr(wgc_version, "fetch_csis_version", fetch)

    before = wgc_version.resolve_wgc_version(tmp_path, refresh=True)
    after = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert before.executable == old
    assert after.executable == new
    assert after.version == NEW_VERSION
    assert fetch.call_count == 2
    assert wgc_version.resolve_wgc_version(tmp_path).executable == new


def test_missing_server_build_never_falls_back_to_another_local_build(tmp_path, monkeypatch):
    write_install(tmp_path)
    write_engine(tmp_path, OLD_VERSION)
    write_preferences(tmp_path)
    seed_cache(tmp_path, {"asia": OLD_VERSION})
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(return_value=NEW_VERSION))

    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable is None
    assert selected.version == NEW_VERSION
    assert selected.source == "csis"
    assert "游戏本体缺失" in selected.detail


def test_selected_realm_controls_endpoint_and_cache(tmp_path, monkeypatch):
    old, new = multiversion_install(tmp_path, realms=("asia", "eu"))
    fetch = Mock(side_effect=[OLD_VERSION, NEW_VERSION])
    monkeypatch.setattr(wgc_version, "fetch_csis_version", fetch)
    assert wgc_version.resolve_wgc_version(tmp_path, refresh=True).executable == old

    write_preferences(tmp_path, active="WOWS EU", version=None)
    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable == new
    assert selected.realm == "eu"
    assert urlsplit(fetch.call_args.args[0]).hostname == "eu.example.test"
    cache = json.loads(wgc_version.VERSION_CACHE_PATH.read_text(encoding="utf-8"))
    assert cache[str(tmp_path.resolve()).casefold()] == {"asia": OLD_VERSION, "eu": NEW_VERSION}
    write_preferences(tmp_path, active="WOWS ASIA", version=None)
    assert wgc_version.resolve_wgc_version(tmp_path).executable == old


def test_network_failure_never_reuses_other_realm_cache(tmp_path, monkeypatch):
    multiversion_install(tmp_path, realms=("asia", "eu"), last_version=None)
    seed_cache(tmp_path, {"asia": OLD_VERSION})
    write_preferences(tmp_path, active="WOWS EU", version=None)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(side_effect=TimeoutError()))

    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable is None
    assert selected.realm == "eu"
    assert selected.version == ""


def test_cache_is_scoped_to_installation_directory(tmp_path, monkeypatch):
    first = tmp_path / "first_install"
    second = tmp_path / "second_install"
    first.mkdir()
    second.mkdir()
    multiversion_install(first, last_version=None)
    multiversion_install(second, last_version=None)
    seed_cache(first, {"asia": OLD_VERSION})
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(side_effect=TimeoutError()))

    assert wgc_version.resolve_wgc_version(first, refresh=True).version == OLD_VERSION
    assert wgc_version.resolve_wgc_version(second, refresh=True).executable is None


@pytest.mark.parametrize("active", [None, "", "unknown alias"])
def test_multiple_realms_without_matching_active_server_do_not_guess(tmp_path, active):
    multiversion_install(tmp_path, realms=("asia", "eu"))
    seed_cache(tmp_path, {"asia": OLD_VERSION, "eu": NEW_VERSION})
    write_preferences(tmp_path, active=active, version=OLD_VERSION)

    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable is None
    assert selected.realm == ""
    assert "区服" in selected.detail


def test_explicit_unknown_active_server_does_not_reuse_single_realm_cache(tmp_path):
    multiversion_install(tmp_path)
    seed_cache(tmp_path, {"asia": OLD_VERSION})
    write_preferences(tmp_path, active="WOWS EU", version=OLD_VERSION)

    selected = wgc_version.resolve_wgc_version(tmp_path)

    assert selected.executable is None
    assert selected.realm == ""


@pytest.mark.parametrize("fallback", ["cache", "preferences"])
def test_network_failure_uses_same_realm_known_version(tmp_path, monkeypatch, fallback):
    old, _ = multiversion_install(tmp_path)
    if fallback == "cache":
        seed_cache(tmp_path, {"asia": OLD_VERSION})
        write_preferences(tmp_path, version=NEW_VERSION)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(side_effect=TimeoutError()))

    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable == old
    assert selected.source == fallback
    assert "查询失败" in selected.detail


def test_configured_install_with_unknown_server_version_never_uses_installed(tmp_path, monkeypatch):
    multiversion_install(tmp_path, last_version=None)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(side_effect=OSError()))

    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable is None
    assert selected.version == ""
    assert selected.source == ""


@pytest.mark.parametrize("version", [
    "15.9.0.../../../outside", "15.9.0.-13357625", "15.9.0.1.13357625",
    "15.9.13357625", "15.9.0.13357625/../other", "15.9.0.13357625.exe",
])
def test_invalid_server_version_cannot_select_an_executable(tmp_path, monkeypatch, version):
    multiversion_install(tmp_path, last_version=None)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(return_value=version))

    selected = wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    assert selected.executable is None
    assert selected.version == ""
    assert not wgc_version.VERSION_CACHE_PATH.exists()


def test_offline_status_reads_preferences_without_requesting_server(tmp_path, monkeypatch):
    old, _ = multiversion_install(tmp_path)
    monkeypatch.setattr(launcher, "find_steam_executable", lambda: None)
    monkeypatch.setattr(launcher, "find_steam_game_executable", lambda: None)
    monkeypatch.setattr(launcher, "find_wgc_executable", lambda: None)
    monkeypatch.setattr(launcher, "find_wgc_game_executable", lambda: old)

    status = launcher.launcher_statuses()["wgc"]

    assert status["available"] is True
    assert Path(status["game_path"]) == old
    assert status["version"] == OLD_VERSION
    assert status["realm"] == "asia"
    assert status["version_source"] == "preferences"
    wgc_version.fetch_csis_version.assert_not_called()


def test_startup_launches_refreshed_engine_with_shared_installation_profile(tmp_path, monkeypatch):
    old, new = multiversion_install(tmp_path)
    seed_cache(tmp_path, {"asia": OLD_VERSION})
    monkeypatch.setattr(launcher, "find_wgc_game_executable", lambda: old)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(return_value=NEW_VERSION))
    popen = Mock()
    startfile = Mock()

    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile)

    assert result.started is True
    assert popen.call_args.args[0] == [str(new)]
    assert popen.call_args.kwargs["cwd"] == str(tmp_path.resolve())
    startfile.assert_not_called()


@pytest.mark.parametrize("app_type,expected_options", [
    ("high", ["--apptype=high"]),
    ("high --launch_wgc", []),
])
def test_direct_startup_preserves_valid_native_application_type(tmp_path, monkeypatch,
                                                              app_type, expected_options):
    old, new = multiversion_install(tmp_path)
    root = ET.Element("application")
    ET.SubElement(root, "app_type").text = app_type
    ET.ElementTree(root).write(tmp_path / "app_type.xml", encoding="utf-8")
    monkeypatch.setattr(launcher, "find_wgc_game_executable", lambda: old)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(return_value=NEW_VERSION))
    popen = Mock()

    result = launcher.launch_game(client="wgc", popen=popen, startfile=Mock())

    assert result.started is True
    assert popen.call_args.args[0] == [str(new), *expected_options]
    assert popen.call_args.kwargs["cwd"] == str(tmp_path.resolve())


def test_startup_does_not_dispatch_when_required_engine_is_missing(tmp_path, monkeypatch):
    write_install(tmp_path)
    old = write_engine(tmp_path, OLD_VERSION)
    write_preferences(tmp_path)
    monkeypatch.setattr(launcher, "find_wgc_game_executable", lambda: old)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", Mock(return_value=NEW_VERSION))
    popen = Mock()
    startfile = Mock()

    result = launcher.launch_game(client="wgc", popen=popen, startfile=startfile)

    assert result.started is False
    popen.assert_not_called()
    startfile.assert_not_called()


def test_endpoint_preserves_other_queries_and_uses_each_master_once(tmp_path, monkeypatch):
    multiversion_install(tmp_path)
    config_path = tmp_path / "bin" / NEW_VERSION.rsplit(".", 1)[-1] / "res" / "scripts_config.xml"
    config = ET.parse(config_path)
    host = config.getroot().find("server/host")
    host.set("csis", REALMS["asia"][1] + "?mode=version&periphery=stale#discard")
    ET.SubElement(host, "master", addr="another-master.example.test:20020")
    config.write(config_path, encoding="utf-8")
    fetch = Mock(return_value=OLD_VERSION)
    monkeypatch.setattr(wgc_version, "fetch_csis_version", fetch)

    wgc_version.resolve_wgc_version(tmp_path, refresh=True)

    url = urlsplit(fetch.call_args.args[0])
    assert parse_qs(url.query) == {
        "mode": ["version"],
        "periphery": [REALMS["asia"][2], "another-master.example.test:20020"],
    }
    assert url.fragment == ""


def fetch_payload(payload, monkeypatch):
    calls = []

    class Response(BytesIO):
        def read(self, size=-1):
            calls.append(size)
            return super().read(size)

    open_mock = Mock(return_value=Response(payload))
    monkeypatch.setattr(wgc_version, "urlopen", open_mock)
    return open_mock, calls


def test_csis_parser_uses_numeric_largest_periphery_version(monkeypatch):
    open_mock, reads = fetch_payload(
        b"<cluster><items><item><version>15.9.0.13357625</version></item>"
        b"<item><version>15.10.0.14000000</version></item>"
        b"<item><version>bad</version></item></items></cluster>", monkeypatch)

    assert FETCH_CSIS_VERSION("https://asia.example.test/csis") == "15.10.0.14000000"
    assert reads == [256 * 1024 + 1]
    assert open_mock.call_args.kwargs["timeout"] == 4


@pytest.mark.parametrize("payload", [
    b"<other><items><item><version>15.8.0.13187581</version></item></items></other>",
    b"<cluster><items><item><version>15.8.0.../../../outside</version></item></items></cluster>",
    b"<cluster><items /></cluster>",
    b"x" * (256 * 1024 + 1),
], ids=["wrong_root", "invalid_version", "empty_versions", "oversized"])
def test_csis_parser_rejects_invalid_or_excessive_payloads(monkeypatch, payload):
    fetch_payload(payload, monkeypatch)

    with pytest.raises(ValueError):
        FETCH_CSIS_VERSION("https://asia.example.test/csis")
