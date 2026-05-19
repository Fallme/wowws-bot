"""窗口操作模块 - 找游戏窗口 + 截图 + 后台点击"""

import time
import ctypes
import ctypes.wintypes
import win32gui
import win32con
import win32api
import numpy as np
import mss


def find_game_window():
    """查找战舰世界窗口"""
    targets = ["World of Warships", "战舰世界", "wows"]
    result = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            for t in targets:
                if t.lower() in title.lower():
                    result.append((hwnd, title))
        return True

    win32gui.EnumWindows(callback, None)
    return result


def get_window_rect(hwnd):
    """获取窗口矩形"""
    rect = win32gui.GetWindowRect(hwnd)
    return {
        "left": rect[0], "top": rect[1],
        "right": rect[2], "bottom": rect[3],
        "width": rect[2] - rect[0],
        "height": rect[3] - rect[1],
    }


def capture_game(sct, hwnd):
    """用mss截取游戏窗口区域 (后台截图)"""
    rect = get_window_rect(hwnd)
    monitor = {
        "left": rect["left"],
        "top": rect["top"],
        "width": rect["width"],
        "height": rect["height"],
    }
    img = np.array(sct.grab(monitor))
    return img[:, :, :3]  # BGRA -> BGR


def click_window(hwnd, rel_x, rel_y):
    """向窗口发送点击消息 (不移动鼠标!)"""
    try:
        point = win32api.MAKELONG(int(rel_x), int(rel_y))
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, point)
        time.sleep(0.05)
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, point)
    except Exception:
        # 如果PostMessage失败(权限问题)，用ctypes尝试
        try:
            lparam = (int(rel_y) << 16) | (int(rel_x) & 0xFFFF)
            ctypes.windll.user32.SendMessageW(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
            time.sleep(0.05)
            ctypes.windll.user32.SendMessageW(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
        except Exception as e:
            print(f"点击失败: {e}")
