"""查找游戏窗口 + 截图测试 (不控制鼠标)"""

import sys
sys.path.insert(0, "E:/aimemo/wowws-bot")

from core.window import find_game_window, capture_window, get_window_rect
import cv2
import os

SAVE_DIR = "E:/aimemo/wowws-bot/snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)

print("搜索战舰世界窗口...")
windows = find_game_window()

if not windows:
    print("未找到战舰世界窗口!")
    print("请确保游戏已启动")
else:
    for hwnd, title in windows:
        print(f"找到窗口: [{hwnd}] {title}")
        rect = get_window_rect(hwnd)
        print(f"  位置: {rect['width']}x{rect['height']} @ ({rect['left']},{rect['top']})")

        # 截图
        img = capture_window(hwnd)
        if img is not None:
            path = f"{SAVE_DIR}/game_window.jpg"
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  截图已保存: {path} ({img.shape[1]}x{img.shape[0]})")
        else:
            print("  截图失败!")
