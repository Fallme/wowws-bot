from types import SimpleNamespace

import main
from core import window


def _visible_window_api(monkeypatch, titles):
    monkeypatch.setattr(window.win32gui, "IsWindow", lambda hwnd: hwnd in titles)
    monkeypatch.setattr(
        window.win32gui,
        "IsWindowVisible",
        lambda hwnd: hwnd in titles,
    )
    monkeypatch.setattr(window.win32gui, "GetWindowText", lambda hwnd: titles[hwnd])


def test_game_like_video_title_is_rejected_by_process_identity(monkeypatch):
    _visible_window_api(monkeypatch, {101: "World of Warships 战斗录像.mp4"})
    monkeypatch.setattr(
        window,
        "window_process_identity",
        lambda _hwnd: (5001, "vlc.exe", r"C:\Program Files\VideoLAN\VLC\vlc.exe"),
    )

    assert not window.is_game_window(101)


def test_real_game_process_is_accepted_even_with_localized_title(monkeypatch):
    _visible_window_api(monkeypatch, {202: "任意本地化客户端标题"})
    monkeypatch.setattr(
        window,
        "window_process_identity",
        lambda _hwnd: (
            6002,
            "worldofwarships64.exe",
            r"D:\Games\World_of_Warships\bin\WorldOfWarships64.exe",
        ),
    )

    assert window.is_game_window(202)


def test_find_game_window_filters_every_visible_window_by_process(monkeypatch):
    titles = {
        101: "World of Warships 攻略视频",
        202: "战舰世界",
        303: "浏览器",
    }
    _visible_window_api(monkeypatch, titles)
    identities = {
        101: (5001, "mpv.exe", r"C:\Tools\mpv.exe"),
        202: (
            6002,
            "worldofwarships64.exe",
            r"D:\Games\World_of_Warships\bin\WorldOfWarships64.exe",
        ),
        303: (7003, "chrome.exe", r"C:\Chrome\chrome.exe"),
    }
    monkeypatch.setattr(
        window,
        "window_process_identity",
        lambda hwnd: identities[hwnd],
    )
    monkeypatch.setattr(
        window.win32gui,
        "EnumWindows",
        lambda callback, argument: [callback(hwnd, argument) for hwnd in titles],
    )
    monkeypatch.setattr(
        window,
        "get_window_rect",
        lambda _hwnd: {
            "left": 0,
            "top": 0,
            "right": 2560,
            "bottom": 1440,
        },
    )

    assert window.find_game_window() == [
        (202, "战舰世界", (0, 0, 2560, 1440))
    ]


def test_custom_regional_process_name_can_be_added(monkeypatch):
    monkeypatch.setenv("WOWS_GAME_PROCESS_NAMES", "WorldOfWarshipsAsia.exe")
    _visible_window_api(monkeypatch, {404: "区域客户端"})
    monkeypatch.setattr(
        window,
        "window_process_identity",
        lambda _hwnd: (
            8004,
            "worldofwarshipsasia.exe",
            r"E:\Games\WorldOfWarshipsAsia.exe",
        ),
    )

    assert window.is_game_window(404)


def test_completion_close_force_terminates_only_the_verified_game(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "is_game_window", lambda hwnd: hwnd == 202)
    monkeypatch.setattr(
        main, "window_process_identity",
        lambda hwnd: (4242, "worldofwarships64.exe", r"E:\Games\wows64.exe"),
    )
    monkeypatch.setattr(
        main, "force_terminate_process_tree",
        lambda pid: calls.append(pid) or True,
    )

    assert main.close_game_window_after_plan(202)
    assert calls == [4242]
    # An unverified window handle never reaches the process lookup.
    assert not main.close_game_window_after_plan(101)
    assert calls == [4242]


def test_completion_close_refuses_a_non_game_process(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "is_game_window", lambda hwnd: True)
    monkeypatch.setattr(
        main, "window_process_identity",
        lambda hwnd: (777, "chrome.exe", r"C:\chrome.exe"),
    )
    monkeypatch.setattr(
        main, "force_terminate_process_tree",
        lambda pid: calls.append(pid) or True,
    )

    assert not main.close_game_window_after_plan(202)
    assert calls == []


def test_completion_close_reports_a_failed_termination(monkeypatch):
    monkeypatch.setattr(main, "is_game_window", lambda hwnd: True)
    monkeypatch.setattr(
        main, "window_process_identity",
        lambda hwnd: (4242, "worldofwarships64.exe", r"E:\Games\wows64.exe"),
    )
    monkeypatch.setattr(
        main, "force_terminate_process_tree", lambda pid: False,
    )

    assert not main.close_game_window_after_plan(202)


def test_platform_close_terminates_each_known_image(monkeypatch):
    calls = []

    def run(args, **_kwargs):
        calls.append(list(args))
        return SimpleNamespace(returncode=0 if args[-1] == "wgc.exe" else 128)

    monkeypatch.setattr(
        main, "platform_process_names",
        lambda: ("wgc.exe", "wgc_renderer_host.exe"),
    )
    monkeypatch.setattr(main.subprocess, "run", run)

    assert main.close_wargaming_platform_after_plan()
    assert calls == [
        ["taskkill", "/F", "/T", "/IM", "wgc.exe"],
        ["taskkill", "/F", "/T", "/IM", "wgc_renderer_host.exe"],
    ]


def test_platform_close_is_a_no_op_when_absent(monkeypatch):
    monkeypatch.setattr(main, "platform_process_names", lambda: ("wgc.exe",))
    monkeypatch.setattr(
        main.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128),
    )

    assert not main.close_wargaming_platform_after_plan()
