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
