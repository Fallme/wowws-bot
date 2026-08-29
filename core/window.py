"""Windows game-window discovery and physical cursor operations."""

import ctypes
import ctypes.wintypes
import logging
import time

import win32con
import win32gui
import win32process


logger = logging.getLogger("window")
GAME_WINDOW_TOKENS = ("world of warships", "战舰世界", "wows")
_INTERACTION_PAUSE_GUARD = None


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
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


_enable_dpi_awareness()


def _foreground_matches(hwnd) -> bool:
    """Treat a foreground child/owned game surface as the game window itself."""
    try:
        foreground = int(ctypes.windll.user32.GetForegroundWindow() or 0)
        if foreground == int(hwnd):
            return True
        if not foreground:
            return False
        return int(win32gui.GetAncestor(foreground, win32con.GA_ROOT)) == int(
            win32gui.GetAncestor(int(hwnd), win32con.GA_ROOT)
        )
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
    """Return whether ``hwnd`` is a visible World of Warships window."""
    if not hwnd:
        return False
    try:
        if not win32gui.IsWindow(int(hwnd)) or not win32gui.IsWindowVisible(int(hwnd)):
            return False
        title = win32gui.GetWindowText(int(hwnd)).strip().lower()
        return any(token in title for token in GAME_WINDOW_TOKENS)
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
    """Return visible matching windows as ``(hwnd, title, rect)`` tuples."""
    result = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if any(target in title.lower() for target in GAME_WINDOW_TOKENS):
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
    original = ctypes.wintypes.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(original)):
        return False

    try:
        virtual_left = ctypes.windll.user32.GetSystemMetrics(76)
        virtual_top = ctypes.windll.user32.GetSystemMetrics(77)
        virtual_width = ctypes.windll.user32.GetSystemMetrics(78)
        virtual_height = ctypes.windll.user32.GetSystemMetrics(79)
        if virtual_width <= 1 or virtual_height <= 1:
            return False

        normalized_x = int(
            (screen_x - virtual_left) * 65535 / (virtual_width - 1)
        )
        normalized_y = int(
            (screen_y - virtual_top) * 65535 / (virtual_height - 1)
        )
        normalized_x = max(0, min(normalized_x, 65535))
        normalized_y = max(0, min(normalized_y, 65535))
        move_flags = (
            win32con.MOUSEEVENTF_MOVE
            | win32con.MOUSEEVENTF_ABSOLUTE
            | 0x4000  # MOUSEEVENTF_VIRTUALDESK
        )
        ctypes.windll.user32.mouse_event(
            move_flags, normalized_x, normalized_y, 0, 0
        )
        time.sleep(0.15 + max(0.0, extra_delay))
        moved = ctypes.wintypes.POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(moved)):
            return False
        if abs(moved.x - screen_x) > 3 or abs(moved.y - screen_y) > 3:
            # ``mouse_event(MOVE|ABSOLUTE)`` can still be DPI-virtualized when
            # the worker is launched by a 150% scaled desktop host.  Captures
            # and GetWindowRect then use physical pixels (for example
            # 2560x1600) while the injected move lands at the corresponding
            # logical coordinate (1707x1067).  Fall back to SetCursorPos and
            # verify the physical destination before sending *one* click.  No
            # button event has been emitted at this point, so this cannot
            # duplicate a UI action.
            logger.info(
                "绝对鼠标移动发生坐标缩放: target=(%s,%s) actual=(%s,%s)；"
                "改用物理坐标定位",
                screen_x,
                screen_y,
                moved.x,
                moved.y,
            )
            positioned = False
            for _ in range(2):
                if not ctypes.windll.user32.SetCursorPos(
                    int(screen_x), int(screen_y)
                ):
                    continue
                time.sleep(0.04)
                if not ctypes.windll.user32.GetCursorPos(ctypes.byref(moved)):
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
        ctypes.windll.user32.mouse_event(down_flag, 0, 0, 0, 0)
        time.sleep(0.05)
        ctypes.windll.user32.mouse_event(up_flag, 0, 0, 0, 0)
        time.sleep(0.1)
        return True
    finally:
        ctypes.windll.user32.SetCursorPos(original.x, original.y)


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
    original = ctypes.wintypes.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(original)):
        return False
    try:
        ctypes.windll.user32.SetCursorPos(int(screen_x), int(screen_y))
        time.sleep(0.10)
        if _interaction_paused():
            logger.info("[USER] 滚轮派发前检测到暂停，取消本次滚动")
            return False
        ctypes.windll.user32.mouse_event(
            win32con.MOUSEEVENTF_WHEEL,
            0,
            0,
            int(notches) * 120,
            0,
        )
        time.sleep(0.35)
        return True
    finally:
        ctypes.windll.user32.SetCursorPos(original.x, original.y)


def click_center(hwnd):
    rect = get_client_rect(hwnd)
    return physical_click(
        rect["left"] + rect["width"] // 2,
        rect["top"] + rect["height"] // 2,
        hwnd=hwnd,
    )
