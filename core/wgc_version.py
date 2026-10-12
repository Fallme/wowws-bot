"""Select the same regional server build as WG's multiversion client runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen
import xml.etree.ElementTree as ET


VERSION_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "wgc_server_versions.json"


@dataclass(frozen=True)
class WGVersionSelection:
    executable: Path | None
    version: str = ""
    realm: str = ""
    source: str = ""
    detail: str = ""


def _version(value: str) -> str:
    value = str(value).strip().replace(",", ".")
    if not re.fullmatch(r"\d+(?:\.\d+){3,4}", value):
        return ""
    parts = value.split(".")
    # WGC's installed part version has an extra patch field (15.9.0.0.build).
    if len(parts) == 5:
        if parts[3] != "0":
            return ""
        del parts[3]
    return ".".join(str(int(part)) for part in parts)


def fetch_csis_version(url: str) -> str:
    """Read the public version endpoint used by the game runner; no login data."""
    with urlopen(url, timeout=4) as response:
        payload = response.read(256 * 1024 + 1)
    if len(payload) > 256 * 1024:
        raise ValueError("服务器版本响应过大")
    root = ET.fromstring(payload)
    if root.tag != "cluster":
        raise ValueError("服务器版本响应格式错误")
    versions = {_version(node.text or "") for node in root.findall("items/item/version")}
    versions.discard("")
    if not versions:
        raise ValueError("服务器未返回有效版本")
    # The native client runner uses the largest version if peripheries disagree.
    return max(versions, key=lambda value: tuple(map(int, value.split("."))))


def _parse(path: Path) -> ET.Element | None:
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None


def _native(directory: Path, version: str) -> Path | None:
    version = _version(version)
    if not version:
        return None
    path = (directory / "bin" / version.rsplit(".", 1)[-1] / "bin64" / "WorldOfWarships64.exe").resolve()
    return path if path.is_relative_to(directory) and path.is_file() else None


def _read_cache(directory: Path, realm: str) -> str:
    try:
        values = json.loads(VERSION_CACHE_PATH.read_text(encoding="utf-8"))
        return _version(values[str(directory).casefold()][realm])
    except (OSError, ValueError, KeyError, TypeError):
        return ""


def _save_cache(directory: Path, realm: str, version: str) -> None:
    try:
        try:
            values = json.loads(VERSION_CACHE_PATH.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                values = {}
        except (OSError, ValueError):
            values = {}
        key = str(directory).casefold()
        if not isinstance(values.get(key), dict):
            values[key] = {}
        values[key][realm] = version
        VERSION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = VERSION_CACHE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(VERSION_CACHE_PATH)
    except OSError:
        # A read-only cache cannot prevent a verified live selection.
        pass


def _csis_url(host: ET.Element) -> str:
    url = host.get("csis", "").strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        return ""
    masters = [node.get("addr", "").strip() for node in host.findall("master")]
    if not masters or not all(masters):
        return ""
    query = [(key, value) for key, value in parse_qsl(parts.query) if key != "periphery"]
    query.extend(("periphery", master) for master in masters)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def resolve_wgc_version(directory: Path, *, refresh: bool = False) -> WGVersionSelection:
    """Refresh server version for launches; status checks only read local data."""
    directory = Path(directory).resolve()
    info = _parse(directory / "game_info.xml")
    if info is None or not (info.findtext("game/id") or "").upper().startswith("WOWS."):
        return WGVersionSelection(None, detail="无法确认 WG 游戏安装记录")
    if (info.findtext("game/installed") or "").strip().lower() == "false":
        return WGVersionSelection(None, detail="WG 游戏尚未安装完成")

    # Read configs from every installed engine, rather than assuming the
    # highest downloaded client is the one currently enabled on this realm.
    hosts: dict[str, ET.Element] = {}
    configured = False
    for config in sorted(directory.glob("bin/[0-9]*/res/scripts_config.xml")):
        if not config.parent.parent.name.isdigit():
            continue
        scripts = _parse(config)
        if scripts is None:
            continue
        configured = True
        for host in scripts.findall("server/host"):
            realm = host.get("realm", "").strip().lower()
            if realm:
                hosts[realm] = host

    if not configured:
        # Older single-version layouts do not expose regional server configs.
        part = info.find("game/part_versions/version[@name='client']")
        installed = part.get("installed", "").strip() if part is not None else ""
        version = _version(installed)
        if installed and not version:
            return WGVersionSelection(None, detail="安装版本格式错误")
        if version:
            return WGVersionSelection(_native(directory, version), version, source="installed",
                                      detail=f"安装版本 {version}")
        current = _parse(directory / "current_version.xml")
        build = (current.findtext(".//bin") or "").strip() if current is not None else ""
        if re.fullmatch(r"\d+", build):
            native = (directory / "bin" / build / "bin64" / "WorldOfWarships64.exe").resolve()
            if native.is_relative_to(directory) and native.is_file():
                return WGVersionSelection(native, source="current_version", detail=f"安装构建 {build}")
        return WGVersionSelection(None, detail="安装版本或游戏本体缺失")

    preferences = _parse(directory / "preferences.xml")
    active = (preferences.findtext(".//active_server") or "").strip() if preferences is not None else ""
    matching = [realm for realm, host in hosts.items() if host.get("alias", "").strip() == active]
    realm = matching[0] if len(matching) == 1 else next(iter(hosts)) if not active and len(hosts) == 1 else ""
    if not realm:
        return WGVersionSelection(None, detail="无法确定当前游戏区服，不能自动选择版本")

    source = ""
    version = ""
    warning = ""
    if refresh:
        url = _csis_url(hosts[realm])
        if url:
            try:
                version = _version(fetch_csis_version(url))
                if not version:
                    raise ValueError("无效服务器版本")
                source = "csis"
                _save_cache(directory, realm, version)
            except (OSError, ValueError, ET.ParseError):
                warning = "服务器版本查询失败；"
        else:
            warning = "服务器版本查询地址缺失；"

    if not version:
        version = _read_cache(directory, realm)
        if version:
            source = "cache"
        elif preferences is not None and len(matching) == 1:
            version = _version(preferences.findtext(".//last_server_version") or "")
            if version:
                source = "preferences"
    if not version:
        return WGVersionSelection(None, realm=realm, detail=warning + "当前区服的服务器版本未知")
    native = _native(directory, version)
    label = "服务器实时版本" if source == "csis" else "服务器缓存版本"
    detail = f"{warning}{realm} {label} {version}"
    if native is None:
        detail += " 的游戏本体缺失，请完成客户端更新"
    return WGVersionSelection(native, version, realm, source, detail)
