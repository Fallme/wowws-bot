"""每秒截屏分析 - 支持后台+副屏幕"""

import time
import mss
import cv2
import numpy as np
import os

SAVE_DIR = "E:/aimemo/wowws-bot/snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)

def list_monitors():
    """列出所有显示器"""
    with mss.mss() as sct:
        monitors = sct.monitors
        print("可用显示器:")
        for i, m in enumerate(monitors):
            print(f"  [{i}] {m['width']}x{m['height']} offset({m['left']},{m['top']})")
        return monitors

def analyze_minimap(minimap_img):
    """分析小地图"""
    hsv = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2HSV)
    red_lower1 = np.array([0, 120, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 120, 100])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    red_pixels = np.count_nonzero(red_mask)
    return {"red_enemies": red_pixels, "has_enemies": red_pixels > 20}

def analyze_reload_bar(bar_img):
    """分析装填条"""
    hsv = cv2.cvtColor(bar_img, cv2.COLOR_BGR2HSV)
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    total = green_mask.shape[0] * green_mask.shape[1]
    green_ratio = np.count_nonzero(green_mask) / max(total, 1)
    return {"green_ratio": round(green_ratio, 3), "ready": green_ratio > 0.7}

def main():
    import sys

    # 列出显示器
    monitors = list_monitors()

    # 默认用第二个显示器 (副屏), index=2 (index=0是所有屏幕合并)
    monitor_index = 2
    if len(sys.argv) > 1:
        monitor_index = int(sys.argv[1])

    print(f"\n使用显示器: [{monitor_index}]")
    print("开始截屏分析... 按 Ctrl+C 停止")

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        print(f"截图区域: {monitor}")

        for i in range(120):
            # mss 截图 - 后台也能截，不需要窗口在前台
            full = np.array(sct.grab(monitor))
            full_bgr = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)
            h, w = full_bgr.shape[:2]

            # 保存全屏
            cv2.imwrite(f"{SAVE_DIR}/full_{i:03d}.jpg", full_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # 裁剪小地图 (右下角)
            mm_x = max(0, w - 280)
            mm_y = max(0, h - 280)
            minimap = full_bgr[mm_y:h-30, mm_x:w-30]
            cv2.imwrite(f"{SAVE_DIR}/minimap_{i:03d}.jpg", minimap)
            mm_info = analyze_minimap(minimap)

            # 装填条 (屏幕下方中间)
            rl_y = max(0, h - 120)
            rl_x = max(0, w // 2 - 100)
            reload_area = full_bgr[rl_y:h-80, rl_x:rl_x+200]
            cv2.imwrite(f"{SAVE_DIR}/reload_{i:03d}.jpg", reload_area)
            rl_info = analyze_reload_bar(reload_area)

            # 血条
            hp_y = max(0, h - 150)
            hp_x = max(0, w // 2 - 120)
            health_area = full_bgr[hp_y:h-130, hp_x:hp_x+240]
            cv2.imwrite(f"{SAVE_DIR}/health_{i:03d}.jpg", health_area)

            print(f"[{i:03d}] {w}x{h} | 敌人{'有' if mm_info['has_enemies'] else '无'}(红:{mm_info['red_enemies']}) | "
                  f"装填{'OK' if rl_info['ready'] else '中'}({rl_info['green_ratio']})")

            time.sleep(1)

    print(f"\n截图已保存到 {SAVE_DIR}/")

if __name__ == "__main__":
    main()
