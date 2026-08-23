"""Read-only utility: locate the game window and save one screenshot."""

from pathlib import Path

import cv2

from core.window import find_game_window
from dxgi_capture import ScreenCapture

SNAPSHOT_DIR = Path(r"E:\aimemo\docs\screenshots\wowws_bot\manual")


def main():
    print("搜索战舰世界窗口...")
    windows = find_game_window()
    if not windows:
        print("未找到战舰世界窗口，请确保游戏已启动")
        return 1

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    capture = ScreenCapture()
    for index, (hwnd, title, rect) in enumerate(windows, start=1):
        left, top, right, bottom = rect
        print(
            f"找到窗口: [{hwnd}] {title} "
            f"{right-left}x{bottom-top} @ ({left},{top})"
        )
        image = capture.capture_window(hwnd)
        if image is None:
            print("  截图失败")
            continue
        destination = SNAPSHOT_DIR / f"game_window_{index}.jpg"
        cv2.imwrite(
            str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        print(f"  截图已保存: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
