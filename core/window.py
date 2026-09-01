"""Windows game-window discovery and physical cursor operations."""

import ctypes
import ctypes.wintypes
import logging
import os
from pathlib import Path
import re
import time

import win32con
import win32gui
import win32process


logger = logging.getLogger("window")
DEFAULT_GAME_PROCESS_NAMES = frozenset(
    {
        "worldofwarships.exe",
        "worldofwarships64.exe",
        "worldofwarships32.exe",
    }
)
_INTERACTION_PAUSE_GUARD = None


def configured_game_process_names() -> frozenset[str]:
    """Return exact executable names allowed to own the game window.

    ``WOWS_GAME_PROCESS_NAMES`` can add regional/custom client names separated
    by commas or semicolons.  Window titles are deliberately not used: a video,
    browser tab or media player can contain "World of Warships" in its title.
    """

    configured = os.environ.get("WOWS_GAME_PROCESS_NAMES", "")
    additions = {
        Path(value.strip().strip('"')).name.lower()
        for value in re.split(r"[,;]", configured)
        if value.strip()
    }
    return frozenset(DEFAULT_GAME_PROCESS_NAMES | additions)


def window_process_identity(hwnd) -> tuple[int, str, str]:
    """Return ``(pid, executable_name, executable_path)`` for one HWND.

    Process lookup is fail-closed.  Automation must never focus or send input
    to an unverified window merely because its title resembles the game.
    """

    if not hwnd:
        return 0, "", ""
    process_handle = 0
    try:
        _thread_id, process_id = win32process.GetWindowThreadProcessId(int(hwnd))
        process_id = int(process_id or 0)
        if not process_id:
            return 0, "", ""
        kernel32 = ctypes.windll.kernel32
        open_process = kernel32.OpenProcess
        open_process.argtypes = [
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
        ]
        open_process.restype = ctypes.wintypes.HANDLE
        query_image = kernel32.QueryFullProcessImageNameW
        query_image.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.LPWSTR,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        ]
        query_image.restype = ctypes.wintypes.BOOL
        process_handle = open_process(
            0x1000,  # PROCESS_QUERY_LIMITED_INFORMATION
            False,
            process_id,
        )
        if not process_handle:
            return process_id, "", ""
        capacity = 32768
        buffer = ctypes.create_unicode_buffer(capacity)
        size = ctypes.wintypes.DWORD(capacity)
        if not query_image(
            process_handle,
            0,
            buffer,
            ctypes.byref(size),
        ):
            return process_id, "", ""
        executable_path = buffer.value.strip()
        return process_id, Path(executable_path).name.lower(), executable_path
    except Exception:
        logger.debug("无法读取窗口进程身份: hwnd=%s", hwnd, exc_info=True)
        return 0, "", ""
    finally:
        if process_handle:
            try:
                close_handle = ctypes.windll.kernel32.CloseHandle
                close_handle.argtypes = [ctypes.wintypes.HANDLE]
                close_handle.restype = ctypes.wintypes.BOOL
                close_handle(process_handle)
            except Exception:
                pass


def is_game_process_window(hwnd) -> bool:
    process_id, executable_name, executable_path = window_process_identity(hwnd)
    matched = bool(
        process_id and executable_name in configured_game_process_names()
    )
    logger.debug(
        "窗口进程校验: hwnd=%s pid=%s exe=%s path=%s matched=%s",
        hwnd,
        process_id,
        executable_name or "unknown",
        executable_path or "unknown",
        matched,
    )
    return matched


def set_interaction_pause_guard(guard=None) -> None:
    """Install the process-wide final interlock for focus and input actions.

    Scene code already checks pause before starting an operation, but a user
    keypress can arrive during a multi-attempt foreground activation or while
    the cursor is moving toward a verified control.  The low-level window
    module therefore rechecks this guard immediately before every focus,
    maximize, click and scroll side effect.
    """
    global _INTERACTION_PAUSE_GUARD
    _INTERACTION_PAUSE_GUARD = guard


def _interaction_paused() -> bool:
    guard = _INTERACTION_PAUSE_GUARD
    if guard is None:
        return False
    try:
        return bool(guard())
    except Exception:
        logger.exception("底层暂停门禁检查失败；按暂停处理")
        return True


def _enable_dpi_awareness():
    """Keep captures and clicks in the same physical-pixel coordinate space."""
    try:
        # The API returns FALSE instead of raising when the request is denied.
        # Treat only a successful call as configured so the older fallbacks
        # still get a chance on hosts that impose their own DPI context.
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)
        ):
            return
    except (AttributeError, OSError):
        pass
    try:
        # S_OK (0) means success. E_ACCESSDENIED means a context was already
        # chosen, in which case the physical cursor APIs below remain the
        # authoritative coordinate conversion path.
        if int(ctypes.windll.shcore.SetProcessDpiAwareness(2)) == 0:
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


_enable_dpi_awareness()


def _get_physical_cursor_pos(user32, point) -> bool:
    """Read the cursor in unvirtualized desktop pixels when Windows supports it."""
    getter = getattr(user32, "GetPhysicalCursorPos", None)
    if getter is not None:
        try:
            if getter(ctypes.byref(point)):
                return True
        except (AttributeError, OSError):
            pass
    return bool(user32.GetCursorPos(ctypes.byref(point)))


def _set_physical_cursor_pos(user32, x: int, y: int) -> bool:
    """Position the cursor using physical pixels, bypassing DPI virtualization."""
    setter = getattr(user32, "SetPhysicalCursorPos", None)
    if setter is not None:
        try:
            if setter(int(x), int(y)):
                return True
        except (AttributeError, OSError):
            pass
    return bool(user32.SetCursorPos(int(x), int(y)))


def _foreground_matches(hwnd) -> bool:
    """Treat a foreground child/owned game surface as the game window itself."""
    try:
        foreground = int(ctypes.windll.user32.GetForegroundWindow() or 0)
        if foreground == int(hwnd):
            return True
        if not foreground:
            return False
        if int(win32gui.GetAncestor(foreground, win32con.GA_ROOT)) == int(
            win32gui.GetAncestor(int(hwnd), win32con.GA_ROOT)
        ):
            return True
        # DirectX can replace the visible render surface without immediately
        # invalidating the previously bound top-level HWND. Treat any visible
        # surface owned by the exact same game process as foreground; matching
        # only roots made an already-active game look permanently background.
        _target_thread, target_pid = win32process.GetWindowThreadProcessId(
            int(hwnd)
        )
        _foreground_thread, foreground_pid = (
            win32process.GetWindowThreadProcessId(foreground)
        )
        return bool(target_pid and int(target_pid) == int(foreground_pid))
    except Exception:
        return False


def activate_window(hwnd):
    """Foreground the game and keep it maximized, without dragging it."""
    if _interaction_paused():
        logger.info("[USER] 暂停期间拒绝激活游戏窗口")
        return False
    try:
        # The user requested a consistent maximized game window whenever the
        # automation returns to it.  SW_MAXIMIZE changes only the standard
        # window state; no SetWindowPos/MoveWindow/drag operation is used.
        if _interaction_paused():
            return False
        if win32gui.IsIconic(int(hwnd)) or not ctypes.windll.user32.IsZoomed(int(hwnd)):
            ctypes.windll.user32.ShowWindow(int(hwnd), win32con.SW_MAXIMIZE)
            time.sleep(0.18)
        # Windows blocks plain SetForegroundWindow when the automation process
        # is not the foreground owner.  Temporarily joining the foreground and
        # game input queues is the documented activation route.  This changes
        # activation only: it never posts a mouse event or changes geometry.
        user32 = ctypes.windll.user32
        foreground = int(user32.GetForegroundWindow() or 0)
        # GetCurrentThreadId is exported by kernel32, not user32.  Calling it
        # through user32 raised AttributeError and aborted every foreground
        # activation before SetForegroundWindow could run.
        current_thread = int(ctypes.windll.kernel32.GetCurrentThreadId() or 0)
        foreground_thread = 0
        game_thread = 0
        attached_foreground = False
        attached_game = False
        try:
            if foreground:
                foreground_thread, _ = win32process.GetWindowThreadProcessId(foreground)
            game_thread, _ = win32process.GetWindowThreadProcessId(int(hwnd))
            if foreground_thread and foreground_thread != current_thread:
                attached_foreground = bool(
                    user32.AttachThreadInput(
                        current_thread,
                        int(foreground_thread),
                        True,
                    )
                )
            if game_thread and game_thread != current_thread:
                attached_game = bool(
                    user32.AttachThreadInput(current_thread, int(game_thread), True)
                )
            try:
                # ASFW_ANY is advisory only.  The result is still checked below.
                user32.AllowSetForegroundWindow(0xFFFFFFFF)
            except Exception:
                pass
            if _interaction_paused():
                return False
            win32gui.BringWindowToTop(int(hwnd))
            user32.SetForegroundWindow(int(hwnd))
            user32.SetActiveWindow(int(hwnd))
            user32.SetFocus(int(hwnd))
            # Some full-screen DirectX windows ignore SetForegroundWindow
            # while another desktop app owns the foreground lock.  This Win32
            # activation fallback changes z-order only (SWP_NOMOVE/NOSIZE),
            # never the game's coordinates or size.
            if not _foreground_matches(hwnd):
                try:
                    user32.SwitchToThisWindow(int(hwnd), True)
                except Exception:
                    pass
                try:
                    win32gui.SetWindowPos(
                        int(hwnd),
                        win32con.HWND_TOP,
                        0,
                        0,
                        0,
                        0,
                        win32con.SWP_NOMOVE
                        | win32con.SWP_NOSIZE
                        | win32con.SWP_SHOWWINDOW,
                    )
                    win32gui.SetForegroundWindow(int(hwnd))
                except Exception:
                    logger.debug("DirectX foreground fallback failed", exc_info=True)
            if not _foreground_matches(hwnd) and not _interaction_paused():
                try:
                    # SetForegroundWindow can remain locked even after the
                    # input queues were joined. A short topmost transition is
                    # a z-order operation only and gives a maximized DirectX
                    # window one final activation path without synthetic Alt
                    # keys (which would be misreported as user intervention).
                    flags = (
                        win32con.SWP_NOMOVE
                        | win32con.SWP_NOSIZE
                        | win32con.SWP_SHOWWINDOW
                    )
                    user32.ShowWindowAsync(int(hwnd), win32con.SW_MAXIMIZE)
                    win32gui.SetWindowPos(
                        int(hwnd), win32con.HWND_TOPMOST, 0, 0, 0, 0, flags
                    )
                    win32gui.SetWindowPos(
                        int(hwnd), win32con.HWND_NOTOPMOST, 0, 0, 0, 0, flags
                    )
                    win32gui.BringWindowToTop(int(hwnd))
                    win32gui.SetForegroundWindow(int(hwnd))
                except Exception:
                    logger.debug("DirectX topmost activation fallback failed", exc_info=True)
        finally:
            if attached_game:
                user32.AttachThreadInput(current_thread, int(game_thread), False)
            if attached_foreground:
                user32.AttachThreadInput(current_thread, int(foreground_thread), False)
        time.sleep(0.14)
        return _foreground_matches(hwnd)
    except Exception:
        logger.debug("游戏窗口前台激活异常", exc_info=True)
        return False


def maximize_game_window(hwnd) -> bool:
    """Maximize on startup; foreground returns use the same standard state."""
    if _interaction_paused():
        logger.info("[USER] 暂停期间拒绝最大化游戏窗口")
        return False
    try:
        if not win32gui.IsWindow(int(hwnd)):
            return False
        # This is deliberately separate from ``activate_window``.  The caller
        # invokes it once while a run starts.  Later activation also keeps this
        # maximized state, but never uses positional window APIs.
        ctypes.windll.user32.ShowWindow(int(hwnd), win32con.SW_MAXIMIZE)
        time.sleep(0.18)
        return bool(ctypes.windll.user32.IsZoomed(int(hwnd)))
    except Exception:
        return False


def is_game_window(hwnd) -> bool:
    """Return whether ``hwnd`` is visible and owned by the game process."""
    if not hwnd:
        return False
    try:
        if not win32gui.IsWindow(int(hwnd)) or not win32gui.IsWindowVisible(int(hwnd)):
            return False
        return is_game_process_window(int(hwnd))
    except Exception:
        return False


def ensure_game_window_foreground(hwnd) -> bool:
    """Verify and foreground the game before any capture or input action."""
    if _interaction_paused():
        return False
    if not is_game_window(hwnd):
        logger.warning("目标窗口不是可见的战舰世界窗口: hwnd=%s", hwnd)
        return False
    for attempt in range(3):
        if _interaction_paused():
            logger.info("[USER] 前台切换重试期间检测到暂停，立即终止")
            return False
        if _foreground_matches(hwnd):
            return True
        logger.info("游戏不在前台，切换到《战舰世界》窗口 (%s/3)", attempt + 1)
        activate_window(hwnd)
        if _interaction_paused():
            return False
        time.sleep(0.10 * (attempt + 1))
    logger.warning("无法将《战舰世界》切换到前台")
    return False


def find_game_window():
    """Return visible game-process windows as ``(hwnd, title, rect)`` tuples."""
    result = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        if is_game_process_window(hwnd):
            title = win32gui.GetWindowText(hwnd)
            rect = get_window_rect(hwnd)
            result.append(
                (
                    hwnd,
                    title,
                    (rect["left"], rect["top"], rect["right"], rect["bottom"]),
                )
            )
        return True

    win32gui.EnumWindows(callback, None)
    # A DirectX client can briefly expose more than one visible top-level
    # surface while changing display modes.  The rendering window is the
    # largest one; keep it first so lifecycle binding never selects a small
    # helper/dialog owned by the same verified process.
    result.sort(
        key=lambda item: max(0, item[2][2] - item[2][0])
        * max(0, item[2][3] - item[2][1]),
        reverse=True,
    )
    return result


def get_window_rect(hwnd):
    """Return DPI-aware physical window coordinates."""
    rect = ctypes.wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    return {
        "left": rect.left,
        "top": rect.top,
        "right": rect.right,
        "bottom": rect.bottom,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }


def get_client_rect(hwnd):
    """Return the physical screen rectangle of the game client area."""
    if not hwnd:
        raise ValueError("缺少游戏窗口句柄")
    left, top, right, bottom = win32gui.GetClientRect(int(hwnd))
    origin_x, origin_y = win32gui.ClientToScreen(int(hwnd), (left, top))
    width, height = int(right - left), int(bottom - top)
    if width <= 0 or height <= 0:
        raise RuntimeError("游戏客户区大小无效")
    return {
        "left": int(origin_x),
        "top": int(origin_y),
        "right": int(origin_x + width),
        "bottom": int(origin_y + height),
        "width": width,
        "height": height,
    }


def physical_click(
    screen_x, screen_y, extra_delay=0.0, *, hwnd=None, button="left"
):
    """Click a physical coordinate and restore the original cursor.

    ``button`` is explicit so the port workflow can open the selected ship's
    context menu without introducing a second cursor-positioning path.
    """
    if _interaction_paused():
        return False
    if hwnd and not ensure_game_window_foreground(hwnd):
        return False
    user32 = ctypes.windll.user32
    original = ctypes.wintypes.POINT()
    if not _get_physical_cursor_pos(user32, original):
        return False

    try:
        # Captured frames and OCR boxes are physical client pixels. Legacy
        # absolute mouse_event movement is DPI-virtualized when Codex/the web
        # launcher runs at a different scale than the game, which caused the
        # 2K claim target (998, 976) to land around (1275, 756). Windows'
        # physical cursor API accepts the same unvirtualized pixels as the
        # capture pipeline, so use it as the primary path.
        if not _set_physical_cursor_pos(user32, screen_x, screen_y):
            return False
        time.sleep(0.15 + max(0.0, extra_delay))
        moved = ctypes.wintypes.POINT()
        if not _get_physical_cursor_pos(user32, moved):
            return False
        if abs(moved.x - screen_x) > 3 or abs(moved.y - screen_y) > 3:
            logger.info(
                "物理鼠标首次定位有偏差: target=(%s,%s) actual=(%s,%s)；重试",
                screen_x,
                screen_y,
                moved.x,
                moved.y,
            )
            positioned = False
            for _ in range(2):
                if not _set_physical_cursor_pos(user32, screen_x, screen_y):
                    continue
                time.sleep(0.04)
                if not _get_physical_cursor_pos(user32, moved):
                    continue
                if (
                    abs(moved.x - screen_x) <= 3
                    and abs(moved.y - screen_y) <= 3
                ):
                    positioned = True
                    break
            if not positioned:
                logger.warning(
                    "无法将鼠标定位到物理坐标: target=(%s,%s) actual=(%s,%s)",
                    screen_x,
                    screen_y,
                    moved.x,
                    moved.y,
                )
                return False
        if button not in {"left", "right"}:
            raise ValueError(f"unsupported mouse button: {button}")
        down_flag = (
            win32con.MOUSEEVENTF_LEFTDOWN
            if button == "left"
            else win32con.MOUSEEVENTF_RIGHTDOWN
        )
        up_flag = (
            win32con.MOUSEEVENTF_LEFTUP
            if button == "left"
            else win32con.MOUSEEVENTF_RIGHTUP
        )
        if _interaction_paused():
            logger.info("[USER] 鼠标移动后检测到暂停，取消本次点击")
            return False
        user32.mouse_event(down_flag, 0, 0, 0, 0)
        time.sleep(0.05)
        user32.mouse_event(up_flag, 0, 0, 0, 0)
        time.sleep(0.1)
        return True
    finally:
        _set_physical_cursor_pos(user32, original.x, original.y)


def window_message_click(
    hwnd, screen_x, screen_y, extra_delay=0.0, *, button="left"
):
    """Deliver a verified click to a specific game window without moving the cursor.

    Some multi-monitor/remote desktops block global cursor warps from the
    worker process.  In that situation a global down event is unsafe because
    it could land on a different monitor.  The port UI still receives normal
    client mouse messages, so use this narrowly-scoped fallback only after
    screenshot recognition identified the target control.
    """
    if _interaction_paused():
        return False
    if not hwnd or not ensure_game_window_foreground(hwnd):
        return False
    try:
        client_x, client_y = win32gui.ScreenToClient(
            int(hwnd), (int(screen_x), int(screen_y))
        )
        left, top, right, bottom = win32gui.GetClientRect(int(hwnd))
        if not (left <= client_x < right and top <= client_y < bottom):
            logger.warning(
                "窗口消息点击坐标不在客户区: hwnd=%s screen=(%s,%s) client=(%s,%s)",
                hwnd,
                screen_x,
                screen_y,
                client_x,
                client_y,
            )
            return False
        lparam = (int(client_y) & 0xFFFF) << 16 | (int(client_x) & 0xFFFF)
        win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
        if button not in {"left", "right"}:
            raise ValueError(f"unsupported mouse button: {button}")
        down_message = (
            win32con.WM_LBUTTONDOWN
            if button == "left"
            else win32con.WM_RBUTTONDOWN
        )
        up_message = (
            win32con.WM_LBUTTONUP
            if button == "left"
            else win32con.WM_RBUTTONUP
        )
        button_mask = win32con.MK_LBUTTON if button == "left" else win32con.MK_RBUTTON
        if _interaction_paused():
            logger.info("[USER] 窗口消息派发前检测到暂停，取消本次点击")
            return False
        win32gui.PostMessage(hwnd, down_message, button_mask, lparam)
        time.sleep(0.05 + max(0.0, extra_delay))
        win32gui.PostMessage(hwnd, up_message, 0, lparam)
        logger.info(
            "已通过窗口消息派发点击: hwnd=%s client=(%s,%s)",
            hwnd,
            client_x,
            client_y,
        )
        return True
    except Exception as error:
        logger.warning("窗口消息点击派发失败: %s", error)
        return False


def physical_scroll(screen_x, screen_y, notches, *, hwnd=None):
    """Scroll at a physical coordinate and restore the user's cursor."""
    if _interaction_paused():
        return False
    if hwnd and not ensure_game_window_foreground(hwnd):
        return False
    user32 = ctypes.windll.user32
    original = ctypes.wintypes.POINT()
    if not _get_physical_cursor_pos(user32, original):
        return False
    try:
        if not _set_physical_cursor_pos(user32, screen_x, screen_y):
            return False
        time.sleep(0.10)
        moved = ctypes.wintypes.POINT()
        if not _get_physical_cursor_pos(user32, moved):
            return False
        if abs(moved.x - screen_x) > 3 or abs(moved.y - screen_y) > 3:
            logger.warning(
                "无法将滚轮定位到物理坐标: target=(%s,%s) actual=(%s,%s)",
                screen_x,
                screen_y,
                moved.x,
                moved.y,
            )
            return False
        if _interaction_paused():
            logger.info("[USER] 滚轮派发前检测到暂停，取消本次滚动")
            return False
        user32.mouse_event(
            win32con.MOUSEEVENTF_WHEEL,
            0,
            0,
            int(notches) * 120,
            0,
        )
        time.sleep(0.35)
        return True
    finally:
        _set_physical_cursor_pos(user32, original.x, original.y)


def click_center(hwnd):
    rect = get_client_rect(hwnd)
    return physical_click(
        rect["left"] + rect["width"] // 2,
        rect["top"] + rect["height"] // 2,
        hwnd=hwnd,
    )
